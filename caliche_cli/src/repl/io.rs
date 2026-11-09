//! Input/output implementations for REPL mode.

use crate::cancellation::CancellationToken;
use crate::core::traits::InputSource;
use crate::input::{CompletionData, InputHelper};
use anyhow::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use reedline::{
    ColumnarMenu, Completer, EditCommand, Emacs, ExternalPrinter, FileBackedHistory, HistoryItem,
    KeyCode, KeyModifiers, MenuBuilder, Prompt, PromptEditMode, PromptHistorySearch, Reedline,
    ReedlineEvent, ReedlineMenu, Signal, Span, Suggestion,
};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

/// Custom prompt that renders only the provided left segment.
///
/// We intentionally suppress reedline's built-in prompt indicator so the CLI
/// controls the full prompt shape (`[provider/model] > `) without an extra
/// trailing marker like `〉`.
struct ReplPrompt {
    left: String,
}

/// Bridges `InputHelper` completion logic into reedline.
struct ReplCompleter {
    helper: InputHelper,
}

impl ReplCompleter {
    fn new(data: Arc<RwLock<CompletionData>>) -> Self {
        Self {
            helper: InputHelper::with_data(data),
        }
    }
}

impl Completer for ReplCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let (start, pairs) = self.helper.complete(line, pos);
        pairs
            .into_iter()
            .map(|pair| Suggestion {
                value: pair.replacement,
                display_override: None,
                description: Some(pair.display),
                style: None,
                extra: None,
                span: Span::new(start, pos),
                append_whitespace: false,
                match_indices: None,
            })
            .collect()
    }
}

impl Prompt for ReplPrompt {
    fn render_prompt_left(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.left)
    }

    fn render_prompt_right(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed("\x1b[90m... \x1b[0m")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Owned(format!("(reverse-search: {}) ", history_search.term))
    }
}

/// Reedline-based input provider.
///
/// Uses `tokio::task::spawn_blocking` to avoid blocking the async runtime
/// during readline operations.
///
/// The editor is wrapped in `Arc<parking_lot::Mutex<…>>` because:
/// - `spawn_history_load` moves a clone into a `spawn_blocking` closure
///   (requires `'static + Send`), so an `Arc` is unavoidable.
/// - `parking_lot::Mutex` is used instead of `std::sync::Mutex` because it
///   is faster (no OS-level poisoning), non-poisoning (no `expect()` needed),
///   and has a smaller memory footprint.
/// All other accesses (`add_history`, `save_history`, `read_line`) are
/// sequential with respect to the REPL loop and never contend with each other.
#[derive(Clone)]
pub struct ReedlineInput {
    editor: Arc<Mutex<Reedline>>,
    history_path: PathBuf,
    /// Shared completion data — clone this Arc into the REPL engine so it can
    /// refresh the model list and configured-provider list at runtime.
    completion_data: Arc<RwLock<CompletionData>>,
    /// Buffer content to seed into the next prompt render.
    prefill_buffer: Arc<Mutex<Option<String>>>,
    /// Shared printer used to emit streamed output while the prompt stays live.
    external_printer: ExternalPrinter<String>,
    /// Optional cancellation token to trigger when Ctrl+C is pressed in reedline.
    interrupt_token: Arc<Mutex<Option<CancellationToken>>>,
    /// When true, the next `read_line` drops buffered stdin before the prompt.
    stdin_drain_pending: Arc<AtomicBool>,
}

impl ReedlineInput {
    fn is_reedline_cursor_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("cursor position could not be read")
            || e.contains("cursor") && e.contains("could not be read")
    }

    fn fallback_stdin_read_line(prompt: &str) -> Result<Option<String>> {
        use std::io::{self, Write};
        print!("{}", prompt);
        io::stdout().flush()?;
        let mut line = String::new();
        let n = io::stdin().read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        while line.ends_with('\n') || line.ends_with('\r') {
            line.pop();
        }
        Ok(Some(line))
    }

    /// Drop any bytes already buffered on stdin before starting a new prompt.
    ///
    /// This prevents "ghost" submissions when users press Enter while a turn is
    /// still streaming and those bytes are later consumed as the next prompt input.
    #[cfg(unix)]
    #[expect(
        unsafe_code,
        reason = "libc fcntl and read drain bytes already queued on stdin"
    )]
    fn drain_pending_stdin_bytes() -> usize {
        use std::io;
        let fd = libc::STDIN_FILENO;
        // SAFETY: fcntl/read operate on the process stdin file descriptor only.
        // We temporarily set O_NONBLOCK and restore the original flags before return.
        unsafe {
            let original_flags = libc::fcntl(fd, libc::F_GETFL);
            if original_flags < 0 {
                return 0;
            }

            let mut switched_nonblocking = false;
            if (original_flags & libc::O_NONBLOCK) == 0
                && libc::fcntl(fd, libc::F_SETFL, original_flags | libc::O_NONBLOCK) >= 0
            {
                switched_nonblocking = true;
            }

            let mut drained = 0usize;
            let mut buf = [0_u8; 512];
            loop {
                let n = libc::read(fd, buf.as_mut_ptr().cast(), buf.len());
                if n > 0 {
                    drained += n as usize;
                    continue;
                }
                if n == 0 {
                    break;
                }
                let errno = io::Error::last_os_error()
                    .raw_os_error()
                    .unwrap_or_default();
                if errno == libc::EAGAIN || errno == libc::EWOULDBLOCK {
                    break;
                }
                break;
            }

            if switched_nonblocking {
                let _ = libc::fcntl(fd, libc::F_SETFL, original_flags);
            }
            drained
        }
    }

    /// Non-Unix stub: reedline handles buffered stdin natively on Windows, so
    /// there is no equivalent fcntl/read trick to perform here.
    #[cfg(not(unix))]
    fn drain_pending_stdin_bytes() -> usize {
        0
    }

    /// Create a new reedline input provider with default (empty) completion data.
    pub fn new() -> Result<Self> {
        let data = Arc::new(RwLock::new(CompletionData::default()));
        Self::with_completion_data(data, true)
    }

    /// Create with pre-populated completion data.
    /// When `load_history_now` is false, history is not loaded in the constructor;
    /// call `spawn_history_load()` after creation to load it in the background for faster REPL startup.
    pub fn with_completion_data(
        data: Arc<RwLock<CompletionData>>,
        _load_history_now: bool,
    ) -> Result<Self> {
        let history_path = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("caliche")
            .join("history.txt");
        if let Some(parent) = history_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let history = FileBackedHistory::with_file(10_000, history_path.clone())
            .map_err(|e| anyhow::anyhow!("failed to initialize history: {}", e))?;
        let completer = Box::new(ReplCompleter::new(data.clone()));
        let completion_menu = Box::new(ColumnarMenu::default().with_name("completion_menu"));

        let mut keybindings = reedline::default_emacs_keybindings();
        keybindings.add_binding(
            KeyModifiers::NONE,
            KeyCode::Tab,
            ReedlineEvent::UntilFound(vec![
                ReedlineEvent::Menu("completion_menu".to_string()),
                ReedlineEvent::MenuNext,
            ]),
        );
        let edit_mode = Box::new(Emacs::new(keybindings));

        let external_printer = ExternalPrinter::default();
        // A paste arrives as one buffer. Enter sends it. Without this mode,
        // each pasted line submits as its own turn.
        let editor = Reedline::create()
            .use_bracketed_paste(true)
            .with_history(Box::new(history))
            .with_completer(completer)
            .with_menu(ReedlineMenu::EngineCompleter(completion_menu))
            .with_edit_mode(edit_mode)
            .with_external_printer(external_printer.clone());

        Ok(Self {
            editor: Arc::new(Mutex::new(editor)),
            history_path,
            completion_data: data,
            prefill_buffer: Arc::new(Mutex::new(None)),
            external_printer,
            interrupt_token: Arc::new(Mutex::new(None)),
            stdin_drain_pending: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Load history from disk in a blocking task so REPL prompt can show immediately.
    /// Call once after creation when `with_completion_data(..., false)` was used.
    pub fn spawn_history_load(&self) {
        // History is file-backed and loaded eagerly at construction time.
    }

    /// Path to the history file (for deferred load).
    pub fn history_path(&self) -> &PathBuf {
        &self.history_path
    }

    /// Return a clone of the completion data handle so the engine can update it.
    pub fn completion_data(&self) -> Arc<RwLock<CompletionData>> {
        self.completion_data.clone()
    }

    pub fn external_printer(&self) -> ExternalPrinter<String> {
        self.external_printer.clone()
    }

    pub fn set_interrupt_token(&mut self, token: Option<CancellationToken>) {
        *self.interrupt_token.lock() = token;
    }

    /// Get mutable editor access (for synchronous operations).
    pub fn with_editor<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Reedline) -> R,
    {
        // parking_lot::Mutex::lock() is infallible — no poisoning, no expect().
        let mut guard = self.editor.lock();
        f(&mut guard)
    }
}

#[async_trait]
impl InputSource for ReedlineInput {
    async fn read_line(&mut self, prompt: &str) -> Result<Option<String>> {
        let editor: Arc<Mutex<Reedline>> = self.editor.clone();
        let prefill_buffer = self.prefill_buffer.clone();
        let interrupt_token = self.interrupt_token.clone();
        let stdin_drain_pending = self.stdin_drain_pending.clone();
        let prompt = prompt.to_string();
        let prompt_for_reedline = prompt.clone();

        // Run the blocking readline operation in a dedicated thread pool
        // This prevents blocking the async runtime
        let result = tokio::task::spawn_blocking(move || {
            if stdin_drain_pending.swap(false, Ordering::AcqRel) {
                let drained = Self::drain_pending_stdin_bytes();
                if drained > 0 {
                    tracing::debug!(
                        target: "repl::io",
                        "discarded {} buffered stdin byte(s) before prompt read",
                        drained
                    );
                }
            }
            // parking_lot::Mutex::lock() is infallible.
            let mut guard = editor.lock();
            if let Some(prefill) = prefill_buffer.lock().take() {
                guard.run_edit_commands(&[EditCommand::Clear, EditCommand::InsertString(prefill)]);
            }
            let prompt = ReplPrompt {
                left: prompt_for_reedline,
            };
            guard.read_line(&prompt)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?;

        match result {
            Ok(Signal::Success(line)) => Ok(Some(line)),
            Ok(Signal::CtrlC) => {
                if let Some(token) = interrupt_token.lock().clone() {
                    token.cancel();
                    Ok(Some(String::new()))
                } else {
                    Ok(None)
                }
            }
            Ok(Signal::CtrlD) => Ok(None),
            Ok(_) => Ok(Some(String::new())),
            Err(e) => {
                let msg = e.to_string();
                if Self::is_reedline_cursor_error(&msg) {
                    tracing::warn!(
                        target: "repl::io",
                        "reedline cursor read failed; falling back to plain stdin input: {}",
                        msg
                    );
                    return tokio::task::spawn_blocking({
                        let prompt = prompt.clone();
                        move || Self::fallback_stdin_read_line(&prompt)
                    })
                    .await
                    .map_err(|je| anyhow::anyhow!("stdin fallback join error: {}", je))?;
                }
                Err(anyhow::anyhow!("reedline read failed: {}", e))
            }
        }
    }

    fn add_history(&mut self, line: &str) {
        self.with_editor(|editor| {
            // Persist to history backend without mutating the current edit buffer.
            if let Err(err) = editor
                .history_mut()
                .save(HistoryItem::from_command_line(line))
            {
                tracing::warn!(target: "repl::io", "failed to save history entry: {}", err);
            }
        });
    }

    fn set_prefill_buffer(&mut self, text: String) {
        *self.prefill_buffer.lock() = Some(text);
    }

    fn request_stdin_drain(&mut self) {
        self.stdin_drain_pending.store(true, Ordering::Release);
    }

    fn save_history(&mut self) -> Result<()> {
        // FileBackedHistory persists automatically.
        Ok(())
    }
}

/// Standard terminal output handler.
pub trait OutputHandler: Send + Sync {
    /// Print a message.
    fn print(&mut self, message: &str);

    /// Print an error.
    fn print_error(&mut self, message: &str);

    /// Print a success message.
    fn print_success(&mut self, message: &str);

    /// Clear the screen.
    fn clear(&mut self);

    /// Whether output is routed via reedline's external printer.
    fn uses_external_printer(&self) -> bool {
        false
    }
}

/// Terminal output implementation.
pub struct TerminalOutput {
    /// Whether to use colors.
    use_colors: bool,
    external_printer: Option<ExternalPrinter<String>>,
}

impl Default for TerminalOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalOutput {
    /// Create a new terminal output handler.
    pub fn new() -> Self {
        let ansi_ok = crate::repl::terminal::supports_ansi_terminal_control();
        Self {
            use_colors: std::io::stdout().is_terminal() && ansi_ok,
            external_printer: None,
        }
    }

    pub fn with_external_printer(printer: ExternalPrinter<String>) -> Self {
        Self {
            use_colors: true,
            external_printer: Some(printer),
        }
    }

    /// Create without colors.
    pub fn plain() -> Self {
        Self {
            use_colors: false,
            external_printer: None,
        }
    }

    fn colorize<'a>(&self, text: &'a str, code: &str) -> beef::Cow<'a, str> {
        if self.use_colors {
            beef::Cow::owned(format!("\x1b[{}m{}\x1b[0m", code, text))
        } else {
            beef::Cow::borrowed(text)
        }
    }
}

impl OutputHandler for TerminalOutput {
    fn print(&mut self, message: &str) {
        // Always write directly to stdout. The external printer routes through a
        // bounded crossbeam channel consumed by the reedline event loop, which is
        // not running while we render agent output. Using it deadlocks once the
        // channel fills.
        crate::repl::stdout_sync::with_stdout(|out| {
            use std::io::Write;
            if let Err(err) = writeln!(out, "{}", message) {
                tracing::warn!(target: "repl::io", "failed to write output: {}", err);
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(target: "repl::io", "failed to flush output: {}", err);
            }
        });
    }

    fn print_error(&mut self, message: &str) {
        crate::repl::stdout_sync::with_stdout(|out| {
            use std::io::Write;
            if let Err(err) = writeln!(out, "{}", self.colorize(message, "31")) {
                tracing::warn!(target: "repl::io", "failed to write error output: {}", err);
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(target: "repl::io", "failed to flush error output: {}", err);
            }
        });
    }

    fn print_success(&mut self, message: &str) {
        crate::repl::stdout_sync::with_stdout(|out| {
            use std::io::Write;
            if let Err(err) = writeln!(out, "{}", self.colorize(message, "32")) {
                tracing::warn!(
                    target: "repl::io",
                    "failed to write success output: {}",
                    err
                );
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(target: "repl::io", "failed to flush success output: {}", err);
            }
        });
    }

    fn clear(&mut self) {
        crate::repl::stdout_sync::with_stdout(|out| {
            use std::io::Write;
            if let Err(err) = out.write_all(b"\x1b[2J\x1b[H") {
                tracing::warn!(target: "repl::io", "failed to clear terminal: {}", err);
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(target: "repl::io", "failed to flush clear operation: {}", err);
            }
        });
    }

    fn uses_external_printer(&self) -> bool {
        self.external_printer.is_some()
    }
}
