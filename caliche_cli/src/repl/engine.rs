//! REPL engine v2 - using core module for unified state and commands.

use crate::core::state::InteractiveState;
use crate::core::traits::{InputSource, OutputRenderer};
use crate::custom_commands::CommandRegistry;
use anyhow::Result;
use caliche::AgentService;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

#[path = "message_turn.rs"]
mod message_turn;
#[path = "run_loop.rs"]
mod run_loop;
#[path = "stream_sink.rs"]
mod stream_sink;
#[path = "turn_quality.rs"]
mod turn_quality;

/// Result of processing REPL input
#[derive(Debug)]
pub enum InputResult {
    Continue,
    Exit,
    Message(String),
}

/// REPL engine v2 using core module
pub struct ReplEngine<I: InputSource + Clone, R: OutputRenderer> {
    input: I,
    renderer: R,
    state: InteractiveState,
    custom_commands: Arc<RwLock<CommandRegistry>>,
    /// Custom commands are loaded on first slash command to avoid pre-prompt disk I/O.
    /// Set to `true` once custom commands have been loaded from disk.
    /// Uses `AtomicBool` for zero-cost reads on the hot slash-command path.
    commands_loaded: Arc<AtomicBool>,
    /// Live completion data shared with the REPL input layer.
    completion_data: Option<Arc<std::sync::RwLock<crate::input::CompletionData>>>,
    /// Optional shell command to run after each turn completes.
    notify_command: Option<String>,
    /// Documentation/markdown files created so far this session (tracked for the limit).
    docs_created: Vec<std::path::PathBuf>,
    /// Maximum number of new documentation files allowed per session (0 = unlimited).
    max_new_doc_files: usize,
    /// Shared set of extra paths allowed beyond the workspace root for the current
    /// user turn.  Updated from the user's message text before each turn.
    extra_allowed: caliche::agent::hooks::AllowedPaths,
    /// When set, message turns execute via HTTP+SSE (`AgentClient`) instead of
    /// in-process `AgentService` / harness streaming (ROADMAP 2.5 thin client).
    thin_client: Option<crate::client::AgentClient>,
    /// Remote API session id paired with [`Self::thin_client`].
    thin_session_id: Option<String>,
    /// Provider and model the remote session already has.
    ///
    /// Set from the prompt label when the thin client attaches. A turn posts
    /// `provider` and `model` only when the label differs, then stores the
    /// posted pair. Matching turns omit both fields so the remote session keeps
    /// its startup model until the operator switches.
    remote_turn_selection: Option<(String, String)>,
    /// The `/context-window` value the remote session already has. A turn
    /// posts the REPL's window only when it differs.
    remote_context_window: Option<usize>,
    /// Which subagents an `@mention` may start. Empty means every subagent profile.
    mention_policy: caliche::orchestration::TaskInvocationPolicy,
    /// Workspace RAG attachment that must not block the first prompt.
    deferred_workspace: Option<crate::deferred::DeferredWorkspaceRag>,
}

impl<I: InputSource + Clone + 'static, R: OutputRenderer> ReplEngine<I, R> {
    /// Create a new REPL engine (custom commands are loaded on first slash command).
    ///
    /// `notify_command` defaults to `None`; callers that need it should use
    /// `new_with_completions` and pass a pre-loaded config value to avoid a
    /// redundant disk read at construction time.
    pub fn new(input: I, renderer: R, state: InteractiveState) -> Self {
        let custom_commands = Arc::new(RwLock::new(CommandRegistry::new()));
        Self {
            input,
            renderer,
            state,
            custom_commands,
            commands_loaded: Arc::new(AtomicBool::new(false)),
            completion_data: None,
            notify_command: None,
            docs_created: Vec::new(),
            max_new_doc_files: 3,
            extra_allowed: std::sync::Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::new(),
            )),
            thin_client: None,
            thin_session_id: None,
            remote_turn_selection: None,
            remote_context_window: None,
            mention_policy: caliche::orchestration::TaskInvocationPolicy::allow_all(),
            deferred_workspace: None,
        }
    }

    /// Restrict which subagents an `@mention` may start.
    pub fn with_mention_policy(
        mut self,
        policy: caliche::orchestration::TaskInvocationPolicy,
    ) -> Self {
        self.mention_policy = policy;
        self
    }

    /// Attach a thin-client transport so message turns go through remote SSE.
    pub fn with_thin_client(
        mut self,
        client: crate::client::AgentClient,
        session_id: impl Into<String>,
    ) -> Self {
        self.remote_turn_selection = Some((
            self.state.token_budget.provider.clone(),
            self.state.token_budget.model.clone(),
        ));
        self.thin_client = Some(client);
        self.thin_session_id = Some(session_id.into());
        self
    }

    /// Attach deferred workspace RAG discovery that must not block the prompt.
    pub fn with_deferred_workspace(
        mut self,
        deferred: crate::deferred::DeferredWorkspaceRag,
    ) -> Self {
        self.deferred_workspace = Some(deferred);
        self
    }

    /// Create with a live completion-data handle for dynamic Tab completions.
    /// If `notify_command` is `Some(cmd)`, that value is used directly.
    /// Pass `None` to disable the notify command (no disk I/O).
    pub fn new_with_completions(
        input: I,
        renderer: R,
        state: InteractiveState,
        completion_data: Arc<std::sync::RwLock<crate::input::CompletionData>>,
        notify_command: Option<String>,
        max_new_doc_files: usize,
        extra_allowed: caliche::agent::hooks::AllowedPaths,
    ) -> Self {
        let custom_commands = Arc::new(RwLock::new(CommandRegistry::new()));
        Self {
            input,
            renderer,
            state,
            custom_commands,
            commands_loaded: Arc::new(AtomicBool::new(false)),
            completion_data: Some(completion_data),
            notify_command,
            docs_created: Vec::new(),
            max_new_doc_files,
            extra_allowed,
            thin_client: None,
            thin_session_id: None,
            remote_turn_selection: None,
            remote_context_window: None,
            mention_policy: caliche::orchestration::TaskInvocationPolicy::allow_all(),
            deferred_workspace: None,
        }
    }

    /// Spawn a background task that fetches the full model list for `provider`
    /// from the provider API and writes the result into `completion_data`.
    ///
    /// Until the fetch completes the recommended-model list remains in place.
    /// Local runtimes intentionally start empty so Tab never suggests fake
    /// installed models.
    pub fn spawn_model_fetch(
        provider: String,
        completion_data: Arc<std::sync::RwLock<crate::input::CompletionData>>,
    ) {
        tokio::spawn(async move {
            use caliche::llm::{ModelCatalogOptions, ProviderType, discover_provider_models};

            let models = match provider.parse::<ProviderType>() {
                Ok(provider_type) => discover_provider_models(
                    provider_type,
                    &ModelCatalogOptions::from_environment(),
                )
                .await
                .map(|entry| entry.models.into_iter().map(|model| model.id).collect())
                .unwrap_or_default(),
                Err(_) => Vec::new(),
            };

            if !models.is_empty() {
                if let Ok(mut current) = completion_data.write() {
                    // Only update if provider hasn't changed while fetching.
                    if current.active_provider == provider {
                        current.current_models = models;
                    }
                }
            }
        });
    }

    /// If the service's active provider has changed, refresh the model completion list.
    ///
    /// Immediately swaps in the provider's baseline list, then spawns a
    /// background fetch for the full live model list.
    pub(crate) fn maybe_refresh_completions(&self, service: &AgentService) {
        let Some(ref data_arc) = self.completion_data else {
            return;
        };
        let current_provider = service.provider().to_string();

        let provider_changed = data_arc
            .read()
            .map(|d| d.active_provider != current_provider)
            .unwrap_or(false);

        if !provider_changed {
            return;
        }

        // Swap in provider baseline immediately.
        // Clone the current snapshot, update the provider, and store atomically.
        if let Ok(mut current) = data_arc.write() {
            current.set_active_provider(&current_provider);
        }

        // Background-fetch the full list
        Self::spawn_model_fetch(current_provider, data_arc.clone());
    }

    pub(crate) fn refresh_context_ref_completions(&self, service: &AgentService) {
        let Some(ref data_arc) = self.completion_data else {
            return;
        };
        let data_arc = Arc::clone(data_arc);
        let artifact_store = service.artifact_store();
        let mcp_clients = service.mcp_clients();
        tokio::spawn(async move {
            let mut artifact_refs = vec!["artifact:latest".to_string()];
            if let Some(store) = artifact_store {
                match store.list().await {
                    Ok(artifacts) => {
                        artifact_refs.extend(
                            artifacts
                                .into_iter()
                                .rev()
                                .take(25)
                                .map(|artifact| format!("artifact:{}", artifact.id)),
                        );
                    }
                    Err(error) => {
                        tracing::debug!(error = %error, "failed to list artifact completions");
                    }
                }
            }

            let mut resource_refs = Vec::new();
            for client in mcp_clients {
                match client.list_resources().await {
                    Ok(result) => {
                        resource_refs.extend(result.resources.into_iter().take(25).map(
                            |resource| format!("resource:{}::{}", client.name(), resource.uri),
                        ));
                    }
                    Err(error) => {
                        tracing::debug!(
                            server = %client.name(),
                            error = %error,
                            "failed to list MCP resource completions"
                        );
                    }
                }
            }

            if let Ok(mut data) = data_arc.write() {
                artifact_refs.sort();
                artifact_refs.dedup();
                resource_refs.sort();
                resource_refs.dedup();
                data.artifact_refs = artifact_refs;
                data.resource_refs = resource_refs;
            }
        });
    }

    /// Kick off a background model fetch for the provider the session is starting with.
    ///
    /// Call once after `new_with_completions` so `/model` Tab shows real models
    /// from the very first use.
    pub fn spawn_initial_model_fetch(&self, provider: &str) {
        if let Some(ref data_arc) = self.completion_data {
            Self::spawn_model_fetch(provider.to_string(), data_arc.clone());
        }
    }

    /// Load custom commands from disk on first slash command. Takes Arc clones by value so the future is Send (no &self across await).
    pub(crate) async fn ensure_commands_loaded(
        custom_commands: Arc<RwLock<CommandRegistry>>,
        commands_loaded: Arc<AtomicBool>,
    ) {
        // Relaxed load is sufficient here; the async await below provides
        // the necessary ordering before any subsequent reads of the registry.
        if commands_loaded.load(Ordering::Relaxed) {
            return;
        }
        let registry = CommandRegistry::load().await;
        if let Ok(reg) = registry {
            let mut guard = custom_commands.write().await;
            *guard = reg;
        }
        // Release store so the registry write above is visible to other threads.
        commands_loaded.store(true, Ordering::Release);
    }

    /// Build the prompt string based on current mode and provider/model context
    pub fn build_prompt(&self) -> String {
        let mode_indicator = match self.state.interaction_mode {
            caliche::agent::modes::InteractionMode::Agent => "\x1b[1;32m>\x1b[0m",
            caliche::agent::modes::InteractionMode::Ask => "\x1b[1;34m?\x1b[0m",
            caliche::agent::modes::InteractionMode::Plan => "\x1b[1;33m#\x1b[0m",
        };
        let ctx = format!(
            "\x1b[90m[{}/{}]\x1b[0m ",
            self.state.token_budget.provider, self.state.token_budget.model
        );
        let override_suffix = self
            .state
            .current_agent_override
            .as_deref()
            .map(|n| format!("\x1b[90m [agent: {}]\x1b[0m", n))
            .unwrap_or_default();
        format!("{}{}{} ", ctx, mode_indicator, override_suffix)
    }

    fn normalize_input_line_in_place(line: &mut String) {
        normalize_repl_input(line);
    }

    pub(crate) async fn read_logical_message(
        input: &mut I,
        prompt: &str,
    ) -> Result<Option<String>> {
        let Some(mut line) = input.read_line(prompt).await? else {
            return Ok(None);
        };

        let trimmed = line.trim_end();
        if trimmed.ends_with('\\') && !trimmed.trim_end_matches('\\').is_empty() {
            let mut buf = trimmed
                .strip_suffix('\\')
                .map(|s| s.trim_end())
                .unwrap_or(trimmed)
                .to_string();
            buf.push('\n');
            loop {
                match input.read_line("\x1b[90m... \x1b[0m").await {
                    Ok(Some(next)) => {
                        let next_trimmed = next.trim_end();
                        let stripped = next_trimmed
                            .strip_suffix('\\')
                            .map(|s| s.trim_end())
                            .unwrap_or(next_trimmed);
                        buf.push_str(stripped);
                        if !next_trimmed.ends_with('\\') {
                            break;
                        }
                        buf.push('\n');
                    }
                    _ => break,
                }
            }
            Self::normalize_input_line_in_place(&mut buf);
            Ok(Some(buf))
        } else {
            Self::normalize_input_line_in_place(&mut line);
            Ok(Some(line))
        }
    }

    /// Process a single input line
    pub fn process_input(&self, line: &str) -> InputResult {
        match prepared_repl_message(line) {
            Some(message) => InputResult::Message(message),
            None => InputResult::Continue,
        }
    }
}

/// Drop surrounding blank space.
///
/// A single line loses leading and trailing space. A block keeps the indent
/// on its first content line and drops only blank lines at the ends.
fn normalize_repl_input(line: &mut String) {
    if !line.contains('\n') {
        let end = line.trim_end().len();
        line.truncate(end);
        let start = line.len() - line.trim_start().len();
        line.drain(..start);
        return;
    }
    let end = line.trim_end().len();
    line.truncate(end);
    let start: usize = line
        .split_inclusive('\n')
        .take_while(|row| row.trim().is_empty())
        .map(|row| row.len())
        .sum();
    line.drain(..start);
}

fn prepared_repl_message(line: &str) -> Option<String> {
    let mut owned = line.to_string();
    normalize_repl_input(&mut owned);
    if owned.trim().is_empty() {
        return None;
    }
    if owned.contains('\n') {
        return Some(owned);
    }
    let trimmed = owned.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// A slash command is one line that starts with `/`.
///
/// A pasted block stays a chat message, including a block whose first line
/// starts with `/`.
pub(crate) fn is_slash_command(message: &str) -> bool {
    let mut lines = message.lines();
    let Some(first) = lines.next() else {
        return false;
    };
    first.starts_with('/') && lines.all(|line| line.trim().is_empty())
}

#[cfg(test)]
mod paste_tests {
    use super::{is_slash_command, prepared_repl_message};

    #[test]
    fn multiline_paste_keeps_indent_and_is_one_message() {
        let pasted = "\n    fn main() {\n        println!(\"hi\");\n    }\n";
        let message = prepared_repl_message(pasted).expect("message");
        assert!(message.starts_with("    fn main()"));
        assert!(message.contains("        println!"));
        assert!(!is_slash_command(&message));
    }

    #[test]
    fn pasted_comment_block_is_not_a_slash_command() {
        let message = prepared_repl_message("// keep this\nfn answer() {}\n").expect("message");
        assert!(!is_slash_command(&message));
        assert!(message.contains("fn answer()"));
    }

    #[test]
    fn single_line_slash_command_stays_a_command() {
        let message = prepared_repl_message("  /model haiku  ").expect("message");
        assert_eq!(message, "/model haiku");
        assert!(is_slash_command(&message));
        assert!(is_slash_command("/help\n"));
    }
}
