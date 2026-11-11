//! Bash tool: execute shell commands with timeout and output limits.
//!
//! Fine-grained policy (allow / prompt / forbidden per pattern) is enforced by
//! [`ExecPolicyHook`](crate::agent::hooks::ExecPolicyHook) when registered for the `"bash"` tool;
//! configure rules via `permissions.bash` and `permissions.bash_default` in caliche config.

use crate::agent::tools::common::params;
#[cfg(feature = "sandbox")]
use crate::agent::tools::common::sandbox_exec;
use crate::agent::tools::process_tracker;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::json;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command as TokioCommand;
use tokio::time::timeout;

/// Default execution time for bash commands (10 minutes).
const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Hard capture cap (10 MB). Larger outputs fail rather than exhausting memory.
const MAX_OUTPUT_SIZE: usize = 10_048_576;
/// Soft cap for model-facing content (pi/goose-style). Oversized output spills to a temp file.
const MAX_MODEL_OUTPUT_BYTES: usize = 50 * 1024;
const MAX_MODEL_OUTPUT_LINES: usize = 2000;
const FORBIDDEN_FILE_COMMANDS: &[&str] = &[
    "cat", "find", "grep", "head", "ls", "rg", "tail", "tree", "wc",
];
const MAX_DEDUPED_COMMANDS: usize = 256;

const NAME: &str = "bash";
/// Bare shell names allowed for argv0 (paths and arbitrary binaries are rejected).
const ALLOWED_SHELLS: &[&str] = &["bash", "sh", "zsh", "dash"];
const DESCRIPTION: &str = "Execute shell commands (timeout and output limits apply). \
ONLY for project commands: git, build/test/lint/install (cargo, bun, npm, pnpm, yarn, make, go, python), \
and running scripts from package.json/Makefile/README. \
FORBIDDEN: find, grep, rg, head, tail, cat, wc, ls, tree — use fs tool instead (op=list, op=read, op=grep). \
Any bash call that reads, lists, or searches files is a bug.";

#[derive(Debug)]
pub struct BashTool;

impl ToolMeta for BashTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (e.g. 'bun test', 'npm run build', 'cargo check'). Can be a single command or multiple commands separated by && or ;"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for the command (default: current directory)"
                },
                "timeout_secs": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 600, configurable with CALICHE_BASH_TIMEOUT_SECS)",
                    "default": 600
                },
                "shell": {
                    "type": "string",
                    "description": "Bare shell name to use: bash, sh, zsh, or dash (default: bash). Paths are rejected.",
                    "default": "bash",
                    "enum": ["bash", "sh", "zsh", "dash"]
                }
            },
            "required": ["command"]
        })
    }
}

#[async_trait]
impl Tool for BashTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let args = match parse_bash_args(&arguments) {
            Ok(a) => a,
            Err(e) => return *e,
        };
        #[cfg(feature = "sandbox")]
        return run_bash_command(args, None).await;
        #[cfg(not(feature = "sandbox"))]
        run_bash_command(args).await
    }

    #[cfg(feature = "sandbox")]
    async fn execute_sandboxed(
        &self,
        arguments: serde_json::Value,
        sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> ToolResult {
        let args = match parse_bash_args(&arguments) {
            Ok(a) => a,
            Err(e) => return *e,
        };
        run_bash_command(args, sandbox).await
    }
}

/// Parsed bash tool arguments (shared by execute and `execute_sandboxed`).
struct ParsedBashArgs {
    command_str: String,
    working_dir: Option<PathBuf>,
    timeout_secs: u64,
    shell: String,
}

/// Parse and validate bash tool arguments once. Returns error `ToolResult` on validation failure.
fn parse_bash_args(
    arguments: &serde_json::Value,
) -> std::result::Result<ParsedBashArgs, Box<ToolResult>> {
    let command_str = params::extract_string(arguments, "command", None)
        .map_err(|msg| Box::new(ToolResult::error(msg.to_string())))?;
    if command_str.trim().is_empty() {
        return Err(Box::new(ToolResult::error(
            "Missing required parameter 'command'. Please provide a shell command to execute."
                .to_string(),
        )));
    }
    if let Some(command) = forbidden_file_command(&command_str) {
        return Err(Box::new(ToolResult::error(format!(
            "Bash command '{command}' is reserved for project commands only. Use the fs tool instead: op=list for directory listing, op=read for file contents, or op=grep for searching."
        ))));
    }
    let working_dir = arguments["working_directory"]
        .as_str()
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok());
    let timeout_secs = params::extract_number_or(arguments, "timeout_secs", default_timeout_secs());
    let shell = match normalize_shell_argv0(&params::extract_string_or(arguments, "shell", "bash"))
    {
        Ok(shell) => shell,
        Err(message) => return Err(Box::new(ToolResult::error(message))),
    };
    if should_enforce_duplicate_guard() && is_duplicate_command(&command_str, working_dir.as_ref())
    {
        return Err(Box::new(ToolResult::error(
            "Exact duplicate bash command already ran in this process. Reuse the previous result or change the command.".to_string(),
        )));
    }
    Ok(ParsedBashArgs {
        command_str,
        working_dir,
        timeout_secs,
        shell,
    })
}

/// Validate bash tool `shell` argv0: bare allowlisted name only (no paths).
fn normalize_shell_argv0(shell: &str) -> Result<String, String> {
    let trimmed = shell.trim();
    if trimmed.is_empty() {
        return Err("Shell argv0 must be a non-empty bare name (bash, sh, zsh, dash).".to_string());
    }
    if trimmed.contains('/') || trimmed.contains('\\') || trimmed.contains('\0') {
        return Err(format!(
            "Shell '{trimmed}' must be a bare name (bash, sh, zsh, dash), not a path or binary."
        ));
    }
    let canonical = trimmed.to_ascii_lowercase();
    if ALLOWED_SHELLS.iter().any(|allowed| *allowed == canonical) {
        return Ok(canonical);
    }
    Err(format!(
        "Shell '{trimmed}' is not allowed. Use one of: {}.",
        ALLOWED_SHELLS.join(", ")
    ))
}

fn forbidden_file_command(command: &str) -> Option<&'static str> {
    let first = command
        .split_whitespace()
        .next()
        .and_then(|token| token.rsplit('/').next())
        .unwrap_or_default();
    FORBIDDEN_FILE_COMMANDS
        .iter()
        .copied()
        .find(|forbidden| first == *forbidden)
}

struct DedupeCache {
    order: VecDeque<String>,
    seen: HashSet<String>,
}

fn is_duplicate_command(command: &str, working_dir: Option<&PathBuf>) -> bool {
    static SEEN: OnceLock<Mutex<DedupeCache>> = OnceLock::new();
    let normalized = format!(
        "{}\n{}",
        working_dir
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        command.trim()
    );
    // The cache holds only a VecDeque and a HashSet, so recovering the inner value after a
    // poison is safe; the worst case is one missed duplicate.
    let mut cache = SEEN
        .get_or_init(|| {
            Mutex::new(DedupeCache {
                order: VecDeque::new(),
                seen: HashSet::new(),
            })
        })
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if cache.seen.contains(&normalized) {
        return true;
    }
    cache.seen.insert(normalized.clone());
    cache.order.push_back(normalized);
    if cache.order.len() > MAX_DEDUPED_COMMANDS
        && let Some(oldest) = cache.order.pop_front()
    {
        cache.seen.remove(&oldest);
    }
    false
}

fn should_enforce_duplicate_guard() -> bool {
    std::env::var_os("CALICHE_BASH_DEDUPE").is_some()
}

/// Bound bash output for the model: keep a tail window and spill the full text to a temp file.
fn bound_bash_output_for_model(output: &str) -> (String, Option<serde_json::Value>) {
    let line_count = output.lines().count();
    let needs_bound = output.len() > MAX_MODEL_OUTPUT_BYTES || line_count > MAX_MODEL_OUTPUT_LINES;
    if !needs_bound {
        return (output.to_string(), None);
    }

    let spill_path = match spill_bash_output(output) {
        Ok(path) => path,
        Err(err) => {
            tracing::warn!("Failed to spill oversized bash output: {err}");
            let truncated = tail_bash_output(output);
            return (
                format!(
                    "{truncated}\n\n[Output truncated: {line_count} lines / {} bytes. Spill failed: {err}]",
                    output.len()
                ),
                Some(json!({
                    "truncated": true,
                    "original_bytes": output.len(),
                    "original_lines": line_count,
                })),
            );
        }
    };

    let truncated = tail_bash_output(output);
    let content = format!(
        "{truncated}\n\n[Output truncated: showing last ~{MAX_MODEL_OUTPUT_LINES} lines / {MAX_MODEL_OUTPUT_BYTES} bytes of {line_count} lines / {} bytes]\nFull output: {spill_path}\nRe-read with fs op=read path=\"{spill_path}\" if needed.",
        output.len()
    );
    (
        content,
        Some(json!({
            "truncated": true,
            "original_bytes": output.len(),
            "original_lines": line_count,
            "full_output_path": spill_path,
        })),
    )
}

/// Write `output` to a new spill file and return its path.
///
/// The name holds the time, the process id, and a process-wide counter, and
/// the file is created with `create_new`. Two spills in the same millisecond
/// get different files, and a spill never overwrites another.
fn spill_bash_output(output: &str) -> Result<String> {
    use std::io::Write as _;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);
    let path = spill_dir()?.join(format!(
        "bash-{}-{}-{}.log",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
        std::process::id(),
        SPILL_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options
        .open(&path)
        .and_then(|mut file| file.write_all(output.as_bytes()))
        .context("write bash spill file")?;
    Ok(path.display().to_string())
}

/// This process's spill directory: a new `caliche_bash-*` directory in the
/// user's temp dir, created exclusively and owner-only (0700 on unix), so
/// another local user can neither read the spills nor plant the directory.
fn spill_dir() -> Result<&'static std::path::Path> {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    if let Some(dir) = DIR.get() {
        return Ok(dir);
    }
    #[cfg_attr(
        not(unix),
        expect(unused_mut, reason = "unix sets the mode through it")
    )]
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    // A random v7 name and `create_dir` (never an existing directory) keep
    // another user from planting the path first.
    let dir = std::env::temp_dir().join(format!("caliche_bash-{}", uuid::Uuid::now_v7()));
    builder
        .create(&dir)
        .with_context(|| format!("create bash spill directory {}", dir.display()))?;
    Ok(DIR.get_or_init(|| dir))
}

fn tail_bash_output(output: &str) -> String {
    if output.len() <= MAX_MODEL_OUTPUT_BYTES && output.lines().count() <= MAX_MODEL_OUTPUT_LINES {
        return output.to_string();
    }
    let lines: Vec<&str> = output.lines().collect();
    let line_tail = if lines.len() > MAX_MODEL_OUTPUT_LINES {
        lines[lines.len() - MAX_MODEL_OUTPUT_LINES..].join("\n")
    } else {
        output.to_string()
    };
    if line_tail.len() <= MAX_MODEL_OUTPUT_BYTES {
        return line_tail;
    }
    let start =
        line_tail.floor_char_boundary(line_tail.len().saturating_sub(MAX_MODEL_OUTPUT_BYTES));
    line_tail[start..].to_string()
}

fn bash_success_result(output: String) -> ToolResult {
    let (content, payload) = bound_bash_output_for_model(&output);
    match payload {
        Some(details) => ToolResult::success_with_payload(content, details),
        None => {
            if content.is_empty() {
                ToolResult::success("Command executed successfully with no output.")
            } else {
                ToolResult::success(content)
            }
        }
    }
}

fn bash_error_result(exit_code: i64, output: &str) -> ToolResult {
    let (bounded, payload) = bound_bash_output_for_model(output);
    let message = if bounded.is_empty() {
        format!("Command failed with exit code {exit_code}. No output captured.")
    } else {
        format!("Command failed with exit code {exit_code}:\n{bounded}")
    };
    let mut result = ToolResult::error(message);
    let mut details = match payload {
        Some(value) if value.is_object() => value,
        _ => json!({}),
    };
    details["exit_code"] = json!(exit_code);
    result.typed_payload = Some(details);
    // Put the bounded text in content too so providers that only surface `content` still see it.
    if result.content.is_empty() {
        result.content = result.error.clone().unwrap_or_default();
    }
    result
}

#[cfg(feature = "sandbox")]
pub(crate) fn sandbox_env_from_host() -> std::collections::HashMap<String, String> {
    caliche_contracts::child_env::host_child_env()
        .into_iter()
        .collect()
}

fn default_timeout_secs() -> u64 {
    std::env::var("CALICHE_BASH_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
}

/// Single execution path: run with optional sandbox. When sandbox is None, runs via raw process.
#[cfg(feature = "sandbox")]
async fn run_bash_command(
    args: ParsedBashArgs,
    sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
) -> ToolResult {
    tracing::info!("Executing bash command: {}", args.command_str);
    if let Some(ref wd) = args.working_dir {
        tracing::debug!("Working directory: {}", wd.display());
    }
    if let Some(h) = sandbox {
        let working_dir_str = args
            .working_dir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());
        let timeout = std::time::Duration::from_secs(args.timeout_secs);
        let cmd: Vec<&str> = vec![args.shell.as_str(), "-c", args.command_str.as_str()];
        // Forward a minimal, allowlisted host environment so PATH-dependent commands work
        // without leaking API keys and other secrets into the sandbox.
        let host_env = sandbox_env_from_host();
        let tracker_id = process_tracker::register_process(
            "bash",
            &args.command_str,
            args.working_dir.as_deref(),
            None,
            None,
        );
        let _cancel_guard = process_tracker::cancel_on_drop(tracker_id);
        process_tracker::process_recorded(tracker_id).await;
        let result = h
            .exec(&cmd, working_dir_str.as_deref(), Some(&host_env), timeout)
            .await;
        match result {
            Ok(exec_result) => {
                process_tracker::mark_process_finished(tracker_id, Some(exec_result.exit_code));
                let note = sandbox_exec::truncation_note(&exec_result);
                let mut out = exec_result.stdout;
                if !exec_result.stderr.is_empty() {
                    out.push_str("\n[stderr]\n");
                    out.push_str(&exec_result.stderr);
                }
                out.push_str(&note);
                if exec_result.exit_code != 0 {
                    return bash_error_result(exec_result.exit_code, &out);
                }
                bash_success_result(out)
            }
            Err(e) => {
                process_tracker::mark_process_finished(tracker_id, None);
                ToolResult::error(format!("Sandbox exec failed: {e}"))
            }
        }
    } else {
        match execute_bash_command(
            &args.command_str,
            args.working_dir.as_deref(),
            args.timeout_secs,
            &args.shell,
        )
        .await
        {
            Ok(outcome) => finish_bash_outcome(outcome),
            Err(e) => {
                tracing::warn!("Command execution failed: {}", e);
                ToolResult::error(format!("Command execution failed: {e}"))
            }
        }
    }
}

/// No-sandbox execution path (used when sandbox feature is disabled).
#[cfg(not(feature = "sandbox"))]
async fn run_bash_command(args: ParsedBashArgs) -> ToolResult {
    tracing::info!("Executing bash command: {}", args.command_str);
    if let Some(ref wd) = args.working_dir {
        tracing::debug!("Working directory: {}", wd.display());
    }
    match execute_bash_command(
        &args.command_str,
        args.working_dir.as_deref(),
        args.timeout_secs,
        &args.shell,
    )
    .await
    {
        Ok(outcome) => finish_bash_outcome(outcome),
        Err(e) => {
            tracing::warn!("Command execution failed: {}", e);
            ToolResult::error(format!("Command execution failed: {}", e))
        }
    }
}

struct BashCommandOutcome {
    exit_code: i64,
    output: String,
}

fn finish_bash_outcome(outcome: BashCommandOutcome) -> ToolResult {
    if outcome.exit_code == 0 {
        tracing::info!(
            "Command execution successful ({} bytes output)",
            outcome.output.len()
        );
        bash_success_result(outcome.output)
    } else {
        tracing::warn!(
            "Command execution failed with exit code {}",
            outcome.exit_code
        );
        bash_error_result(outcome.exit_code, &outcome.output)
    }
}

/// Execute bash command with timeout and output limits
async fn execute_bash_command(
    command: &str,
    working_dir: Option<&std::path::Path>,
    timeout_secs: u64,
    shell: &str,
) -> Result<BashCommandOutcome> {
    // Build command
    let mut cmd = TokioCommand::new(shell);
    cmd.arg("-c").arg(command);

    // Set working directory if provided
    if let Some(wd) = working_dir {
        cmd.current_dir(wd);
    }

    // Same allowlist as the sandboxed path: provider keys and tokens in this
    // process never reach the shell.
    cmd.env_clear()
        .envs(caliche_contracts::child_env::host_child_env());

    // Set up stdio
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Ensure the child process is killed when its owning future is dropped
        // (e.g. when the agent turn is cancelled via Ctrl+C).
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    // Register first so the ledger commits the task before the shell runs.
    let tracker_id = process_tracker::register_process("bash", command, working_dir, None, None);
    let _cancel_guard = process_tracker::cancel_on_drop(tracker_id);
    process_tracker::process_recorded(tracker_id).await;

    // Spawn the process
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            process_tracker::mark_process_finished(tracker_id, None);
            return Err(error).with_context(|| format!("Failed to spawn {shell} process"));
        }
    };
    let child_pid = child.id();
    #[cfg(unix)]
    let process_group_id = child_pid;
    #[cfg(not(unix))]
    let process_group_id = None;
    process_tracker::set_process_ids(tracker_id, child_pid, process_group_id);

    // Get stdout and stderr handles
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("Failed to capture stderr"))?;

    // Wait for process with timeout
    let result = timeout(Duration::from_secs(timeout_secs), async {
        // Read stdout and stderr
        let mut stdout_data = Vec::new();
        let mut stderr_data = Vec::new();

        // Read with size limits
        let stdout_read = stdout.read_to_end(&mut stdout_data);
        let stderr_read = stderr.read_to_end(&mut stderr_data);

        // Wait for both reads and process exit
        let (stdout_result, stderr_result, status) =
            tokio::join!(stdout_read, stderr_read, child.wait());

        // Check if reads succeeded
        stdout_result.context("Failed to read stdout")?;
        stderr_result.context("Failed to read stderr")?;

        // Check output size limits
        if stdout_data.len() > MAX_OUTPUT_SIZE {
            bail!(
                "Output too large ({} bytes, max {} bytes). Consider reducing output.",
                stdout_data.len(),
                MAX_OUTPUT_SIZE
            );
        }

        // Get exit status
        let status = status.context("Failed to get exit status")?;

        // Convert output to strings
        let stdout_str = String::from_utf8_lossy(&stdout_data).to_string();
        let stderr_str = String::from_utf8_lossy(&stderr_data).to_string();

        Ok((status, stdout_str, stderr_str))
    })
    .await;

    // Handle timeout
    let (status, stdout_str, stderr_str) = match result {
        Ok(Ok(data)) => {
            process_tracker::mark_process_finished(tracker_id, data.0.code().map(i64::from));
            data
        }
        Ok(Err(e)) => {
            process_tracker::mark_process_finished(tracker_id, None);
            return Err(e);
        }
        Err(_) => {
            if let Some(pgid) = process_group_id {
                process_tracker::terminate_process_group(pgid);
            }
            process_tracker::mark_process_timed_out(tracker_id);
            bail!(
                "Command execution timed out after {timeout_secs} seconds. Consider optimizing your command or increasing timeout."
            );
        }
    };

    // Format the output
    let mut output = String::new();

    if !stdout_str.is_empty() {
        output.push_str(&stdout_str);
    }

    if !stderr_str.is_empty() {
        if !output.is_empty() {
            output.push_str("\n\n");
        }
        output.push_str("STDERR:\n");
        output.push_str(&stderr_str);
    }

    Ok(BashCommandOutcome {
        exit_code: i64::from(
            status
                .code()
                .unwrap_or_else(|| if status.success() { 0 } else { -1 }),
        ),
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_command() {
        let tool = BashTool;
        let args = json!({
            "command": "echo 'Hello, World!'"
        });

        let result = tool.execute(args).await;
        assert!(result.success);
        assert!(result.content.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_multiple_commands() {
        let tool = BashTool;
        let args = json!({
            "command": "echo 'First' && echo 'Second'"
        });

        let result = tool.execute(args).await;
        assert!(result.success);
        assert!(result.content.contains("First"));
        assert!(result.content.contains("Second"));
    }

    #[tokio::test]
    async fn test_failing_command() {
        let tool = BashTool;
        let args = json!({
            "command": "false"
        });

        let result = tool.execute(args).await;
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn oversized_bash_output_spills_and_keeps_tail() {
        let (content, payload) = bound_bash_output_for_model(&"line\n".repeat(3_500));
        assert!(content.contains("Output truncated"));
        assert!(content.contains("Full output:"));
        let details = payload.expect("spill payload");
        assert_eq!(details["truncated"], true);
        assert!(details["full_output_path"].as_str().is_some());
        let path = details["full_output_path"].as_str().expect("path");
        let spilled = std::fs::read_to_string(path).expect("spill file readable");
        assert!(spilled.lines().count() >= 3_500);
    }

    #[cfg(unix)]
    #[test]
    fn spills_are_readable_only_by_their_owner() {
        use std::os::unix::fs::PermissionsExt as _;
        let path = spill_bash_output("secret\n").expect("spill");
        let mode = |path: &std::path::Path| {
            std::fs::metadata(path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777
        };
        let file = std::path::Path::new(&path);
        assert_eq!(mode(file.parent().expect("spill dir")), 0o700);
        assert_eq!(mode(file), 0o600);
        let _ = std::fs::remove_file(file);
    }

    #[test]
    fn spills_in_the_same_millisecond_keep_their_own_output() {
        let spills: Vec<(String, String)> = (0..50)
            .map(|idx| {
                let output = format!("spill {idx}\n");
                let path = spill_bash_output(&output).expect("spill");
                (path, output)
            })
            .collect();
        let paths: HashSet<&str> = spills.iter().map(|(path, _)| path.as_str()).collect();
        assert_eq!(paths.len(), spills.len());
        for (path, output) in &spills {
            assert_eq!(&std::fs::read_to_string(path).expect("spill file"), output);
            let _ = std::fs::remove_file(path);
        }
    }

    #[tokio::test]
    async fn test_empty_command() {
        let tool = BashTool;
        let args = json!({
            "command": ""
        });

        let result = tool.execute(args).await;
        assert!(!result.success);
    }

    #[test]
    fn normalize_shell_argv0_allowlists_bare_names() {
        assert_eq!(normalize_shell_argv0("bash").as_deref(), Ok("bash"));
        assert_eq!(normalize_shell_argv0("Zsh").as_deref(), Ok("zsh"));
        assert_eq!(normalize_shell_argv0(" dash ").as_deref(), Ok("dash"));
    }

    #[test]
    fn normalize_shell_argv0_rejects_paths_and_unknown() {
        assert!(normalize_shell_argv0("/bin/bash").is_err());
        assert!(normalize_shell_argv0("../evil").is_err());
        assert!(normalize_shell_argv0("python").is_err());
        assert!(normalize_shell_argv0("").is_err());
        assert!(normalize_shell_argv0("bash\0").is_err());
    }

    #[tokio::test]
    async fn shell_path_argv0_is_rejected_before_exec() {
        let tool = BashTool;
        let result = tool
            .execute(json!({
                "command": "echo should-not-run",
                "shell": "/bin/bash"
            }))
            .await;
        assert!(!result.success);
        let err = result.error.unwrap_or_default();
        assert!(err.contains("bare name"), "unexpected error: {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unsandboxed_bash_does_not_receive_host_tokens() {
        // The variables must be in this process's environment, which tests
        // may not mutate: re-run this one test in a child that has them.
        if std::env::var_os("CALICHE_SCRUB_FOO_TOKEN").is_none() {
            let name = "agent::tools::bash::tests::unsandboxed_bash_does_not_receive_host_tokens";
            let output =
                tokio::process::Command::new(std::env::current_exe().expect("test binary"))
                    .args(["--exact", name, "--nocapture"])
                    .env("CALICHE_BASH_SCRUB_PROBE", "kept")
                    .env("CALICHE_SCRUB_FOO_TOKEN", "planted")
                    .output()
                    .await
                    .expect("re-run test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let result = BashTool
            .execute(json!({
                "command": "printf '%s|%s' \"${CALICHE_BASH_SCRUB_PROBE-unset}\" \"${CALICHE_SCRUB_FOO_TOKEN-unset}\""
            }))
            .await;
        assert!(result.success, "{:?}", result.error);
        assert_eq!(result.content.trim(), "kept|unset");
    }

    /// Round-5 regression: when the bash tool runs through a sandbox backend
    /// that calls `env_clear()` (Noop, MacOS Seatbelt), the agent must still
    /// see PATH/HOME etc. so build/test commands like `bun test`, `cargo
    /// check`, `npm run build` can be located on disk. This test runs the
    /// bash tool against an explicit Noop sandbox and asserts that PATH
    /// survives the env wipe.
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn bash_tool_forwards_host_path_through_sandbox() {
        use caliche_sandbox::backends::noop::NoopBackend;
        use caliche_sandbox::config::{NamedPolicy, ToolPolicy, policy_to_config};
        use caliche_sandbox::handle::SandboxHandle;
        use caliche_sandbox::traits::SandboxBackend;

        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_BASH_FORWARD_PROBE", "round5_marker");

        let workspace = tempfile::tempdir().expect("workspace");
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            workspace.path(),
            None,
            caliche_sandbox::config::SandboxBackendType::Noop,
        );
        let backend = NoopBackend;
        let instance = backend
            .create(&config, workspace.path())
            .await
            .expect("noop sandbox should construct");
        let handle = SandboxHandle::new(instance, ToolPolicy::default());

        let tool = BashTool;
        let args = json!({
            "command": "echo \"PATH=$PATH\"; echo \"PROBE=$CALICHE_BASH_FORWARD_PROBE\"; \
                        if [ -n \"$PATH\" ]; then echo path_present; else echo path_missing; fi",
        });

        let result = tool.execute_sandboxed(args, Some(&handle)).await;

        assert!(
            result.success,
            "bash tool should succeed under sandbox; got error: {:?}",
            result.error
        );
        assert!(
            result.content.contains("path_present"),
            "PATH must be forwarded into the sandboxed shell; got output: {}",
            result.content
        );
        assert!(
            result.content.contains("PROBE=round5_marker"),
            "agent-side env vars must propagate through the sandbox; got output: {}",
            result.content
        );
    }

    #[cfg(feature = "sandbox")]
    mod sandbox_truncation {
        use super::*;
        use caliche_sandbox::{ExecResult, SandboxHandleOps};
        use std::collections::HashMap;

        struct CannedSandbox(ExecResult);

        #[async_trait]
        impl SandboxHandleOps for CannedSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> Result<ExecResult> {
                Ok(self.0.clone())
            }

            async fn read_file(&self, _path: &str) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> Result<()> {
                Ok(())
            }
        }

        async fn bash_content(
            stderr: &str,
            stdout_truncated: bool,
            stderr_truncated: bool,
        ) -> String {
            let sandbox = CannedSandbox(ExecResult {
                stdout: "out".to_string(),
                stderr: stderr.to_string(),
                stdout_truncated,
                stderr_truncated,
                exit_code: 0,
                duration: Duration::from_millis(1),
            });
            let result = BashTool
                .execute_sandboxed(json!({ "command": "produce" }), Some(&sandbox))
                .await;
            assert!(
                result.success,
                "canned exit 0 must succeed: {:?}",
                result.error
            );
            result.content
        }

        #[tokio::test]
        async fn stdout_cut_by_the_sandbox_is_named_in_the_tool_result() {
            assert_eq!(
                bash_content("", true, false).await,
                "out\n[stdout truncated at 4 MiB]"
            );
        }

        #[tokio::test]
        async fn stderr_cut_by_the_sandbox_is_named_in_the_tool_result() {
            assert_eq!(
                bash_content("err", false, true).await,
                "out\n[stderr]\nerr\n[stderr truncated at 4 MiB]"
            );
        }

        #[tokio::test]
        async fn uncut_streams_carry_no_truncation_note() {
            assert_eq!(
                bash_content("err", false, false).await,
                "out\n[stderr]\nerr"
            );
        }

        #[tokio::test]
        async fn stdout_note_survives_the_model_tail_bound() {
            let stderr = "e\n".repeat(MAX_MODEL_OUTPUT_LINES + 10);
            let content = bash_content(&stderr, true, false).await;
            assert!(
                content.contains("e\n\n[stdout truncated at 4 MiB]\n\n[Output truncated:"),
                "stdout note must sit in the kept tail: {}",
                &content[content.len().saturating_sub(400)..]
            );
        }
    }

    /// The bash tool's ledger wiring, against a real temp-dir ledger.
    #[cfg(feature = "sqlite-store")]
    mod ledger {
        use super::*;
        use crate::data::persistence::task_ledger::test_support::TestLedger;

        #[tokio::test]
        async fn raw_bash_runs_land_in_the_ledger() {
            let ledger = TestLedger::open().await;
            ledger
                .run(async {
                    BashTool.execute(json!({ "command": "echo unit-ok" })).await;
                    BashTool.execute(json!({ "command": "exit 4" })).await;
                    BashTool
                        .execute(json!({ "command": "sleep 30 # unit-timeout", "timeout_secs": 1 }))
                        .await;
                    let cut = tokio::time::timeout(
                        std::time::Duration::from_millis(300),
                        BashTool.execute(json!({ "command": "sleep 30 # unit-cancel" })),
                    )
                    .await;
                    assert!(cut.is_err(), "the cancel case must be cut short");
                })
                .await;

            let ok = ledger.task_named("echo unit-ok").await;
            assert_eq!(
                (ok.kind.as_str(), ok.state.as_str()),
                ("shell", "succeeded")
            );
            assert!(ok.child_pid.is_some(), "the shell pid must be stored");
            let failed = ledger.task_named("exit 4").await;
            assert_eq!(failed.terminal_reason.as_deref(), Some("exit code 4"));
            let slow = ledger.task_named("sleep 30 # unit-timeout").await;
            assert_eq!(
                (slow.state.as_str(), slow.terminal_reason.as_deref()),
                ("failed", Some("timeout"))
            );
            let cut = ledger.task_named("sleep 30 # unit-cancel").await;
            assert_eq!(
                (cut.state.as_str(), cut.terminal_reason.as_deref()),
                ("cancelled", Some("dropped"))
            );
        }

        #[cfg(feature = "sandbox")]
        #[tokio::test]
        async fn sandboxed_bash_runs_land_in_the_ledger() {
            use caliche_sandbox::backends::noop::NoopBackend;
            use caliche_sandbox::config::{NamedPolicy, ToolPolicy, policy_to_config};
            use caliche_sandbox::handle::SandboxHandle;
            use caliche_sandbox::traits::SandboxBackend;

            let workspace = tempfile::tempdir().expect("workspace");
            let config = policy_to_config(
                NamedPolicy::ReadOnly,
                workspace.path(),
                None,
                caliche_sandbox::config::SandboxBackendType::Noop,
            );
            let instance = NoopBackend
                .create(&config, workspace.path())
                .await
                .expect("noop sandbox should construct");
            let handle = SandboxHandle::new(instance, ToolPolicy::default());

            let ledger = TestLedger::open().await;
            ledger
                .run(async {
                    BashTool
                        .execute_sandboxed(json!({ "command": "echo sandboxed-ok" }), Some(&handle))
                        .await;
                    BashTool
                        .execute_sandboxed(json!({ "command": "exit 5" }), Some(&handle))
                        .await;
                })
                .await;

            let ok = ledger.task_named("echo sandboxed-ok").await;
            assert_eq!(ok.state, "succeeded");
            let failed = ledger.task_named("exit 5").await;
            assert_eq!(
                (failed.state.as_str(), failed.terminal_reason.as_deref()),
                ("failed", Some("exit code 5"))
            );
        }
    }
}
