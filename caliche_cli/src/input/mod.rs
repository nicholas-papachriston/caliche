//! Interactive input handling with completions and context references.
//!
//! Provides:
//! - Slash command completion (e.g., /help, /undo, /save)
//! - @ symbol context attachment (e.g., @file.rs, @src/, @http://...)

mod arguments;
mod multiline;

pub use arguments::Pair;
pub use multiline::{MultiLineState, read_multiline};

pub use crate::context_refs::{ContextReference, parse_context_references};
pub use crate::slash_catalog::{
    QUICK_HELP_COMMANDS, slash_command_group_for, slash_command_groups, slash_commands,
};
use caliche_contracts::slash::{SlashSurface, find_slash_command};
use std::borrow::Cow;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CompletionData — shared, dynamically-updated completion lists
// ---------------------------------------------------------------------------

/// Dynamic state for argument completion, updated at runtime.
///
/// Stored behind an `Arc<RwLock<…>>` so the async REPL engine can refresh the
/// data (e.g., after `/provider` fires) while the synchronous `complete()`
/// method reads it lock-free.
#[derive(Debug, Default, Clone)]
pub struct CompletionData {
    /// Provider names that currently have credentials configured.
    /// These appear first (and are labelled) in `/provider` completions.
    pub configured_providers: Vec<String>,
    /// Models for the currently active provider (refreshed on provider switch).
    pub current_models: Vec<String>,
    /// The provider name whose model list is currently in `current_models`.
    pub active_provider: String,
    /// Cached generated artifact tags for `@artifact:` completion.
    pub artifact_refs: Vec<String>,
    /// Cached MCP resource tags for `@resource:` completion.
    pub resource_refs: Vec<String>,
}

impl CompletionData {
    /// Build the initial data for a given provider name.
    ///
    /// `configured` is the set of provider names that have keychain credentials.
    pub fn new(configured: Vec<String>, active_provider: &str) -> Self {
        let current_models = models_for_provider(active_provider);
        Self {
            configured_providers: configured,
            current_models,
            active_provider: active_provider.to_string(),
            artifact_refs: Vec::new(),
            resource_refs: Vec::new(),
        }
    }

    /// Refresh the model list for a new active provider.
    pub fn set_active_provider(&mut self, provider: &str) {
        if self.active_provider != provider {
            self.active_provider = provider.to_string();
            self.current_models = models_for_provider(provider);
        }
    }

    /// Set the list of configured providers (e.g. after async load from keychain).
    pub fn set_configured_providers(&mut self, providers: Vec<String>) {
        self.configured_providers = providers;
    }
}

/// Return the recommended / known models for a provider name.
/// Local runtimes are populated by async discovery instead of static guesses.
pub fn models_for_provider(provider: &str) -> Vec<String> {
    if matches!(provider, "lmstudio" | "ollama") {
        return Vec::new();
    }
    if provider == "local-inference" {
        #[cfg(feature = "local-inference-base")]
        {
            let mut models = caliche::llm::LocalInferenceProvider::discover_downloaded_models();
            models.push(caliche::llm::LocalInferenceProvider::default_model_name().to_string());
            models.sort();
            models.dedup();
            return models;
        }
        #[cfg(not(feature = "local-inference-base"))]
        {
            return Vec::new();
        }
    }

    crate::credentials::get_recommended_models(provider)
        .iter()
        .map(|(m, _)| m.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Static argument completions for slash commands
// ---------------------------------------------------------------------------

/// Commands whose first argument should complete from the configured provider list.
const PROVIDER_ARG_CMDS: &[&str] = &["/provider", "/models", "/subagent-provider"];

/// Commands whose first argument should complete from the current model list.
const MODEL_ARG_CMDS: &[&str] = &["/model", "/subagent-model"];

fn provider_argument_hint(active: &str) -> String {
    if active.is_empty() {
        "<provider>".to_owned()
    } else {
        format!("<provider>  (current: {active})")
    }
}

fn model_argument_hint(active: &str) -> String {
    if active.is_empty() {
        "<model>".to_owned()
    } else {
        format!("<model>  (Tab for {active} models)")
    }
}

fn static_argument_hint(cmd: &str) -> Option<&'static str> {
    if !crate::slash_catalog::command_in_this_build(cmd) {
        return None;
    }
    match cmd {
        "/help" | "/?" => Some("quick|all"),
        "/creds" => Some("status|setup [providers|gateway]|set|import|list|link|gateway ..."),
        "/gateway" => Some("setup|start|stop|restart|status|list|forward|route"),
        "/memory" => Some("buffer|summary|entity|vector|enhanced"),
        "/learning" | "/learning-loop" => {
            Some("status|on|off|interval <n>|min-tool-calls <n>|max-skills <n>")
        }
        "/autocompact" => Some("on|off|<threshold%>"),
        "/embedding" => Some("<provider>[/<model>]"),
        "/image-generation" | "/image" => Some(
            "<openai|google|openrouter|cloudflare-ai-gateway>[/<model>] | <provider> <model> | clear",
        ),
        "/image-generators" | "/images" => Some("[openai|google|openrouter|cloudflare-ai-gateway]"),
        "/artifact-dir" => Some("<path>|clear"),
        "/local-model" => Some("download|list|fit|backends"),
        _ => None,
    }
}

/// Resolve an unknown slash command to a canonical command via fuzzy matching.
/// Returns `Some(canonical)` when there is a unique clear match (prefix or edit distance <= 3).
/// Returns `None` when exact match, ambiguous, or no good match.
/// Used for auto-correcting typos like /hel -> /help before parsing.
pub fn resolve_slash_command(unknown: &str) -> Option<&'static str> {
    if let Some(spec) = find_slash_command(unknown, SlashSurface::Cli) {
        return (spec.primary_name != unknown).then_some(spec.primary_name);
    }

    let unknown_lower = unknown.to_lowercase();
    let candidates: Vec<&str> = slash_commands()
        .iter()
        .map(|(c, _)| canonical_slash_command(c.split_whitespace().next().unwrap_or(c)))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut matches: Vec<(&str, usize)> = Vec::new();
    for cmd in &candidates {
        let cl = cmd.to_lowercase();
        let dist = edit_distance_slash(&unknown_lower, &cl);
        if dist == 0 {
            return None; // Exact match - caller already checked, should not happen
        }
        let prefix_match =
            unknown_lower.starts_with(cl.as_str()) || cl.starts_with(unknown_lower.as_str());
        let score = if prefix_match {
            dist.saturating_sub(10)
        } else {
            dist
        };
        if score <= 3 {
            matches.push((*cmd, score));
        }
    }

    // Ambiguity check: if multiple candidates within threshold, do not auto-correct
    if matches.is_empty() {
        return None;
    }
    matches.sort_by_key(|(_, s)| *s);
    let best_score = matches[0].1;
    let best_matches: Vec<_> = matches.iter().filter(|(_, s)| *s == best_score).collect();
    if best_matches.len() == 1 {
        Some(best_matches[0].0)
    } else {
        None
    }
}

/// Returns the best-matching slash command for "did you mean?" suggestions.
pub fn suggest_slash_command(unknown: &str) -> Option<Cow<'static, str>> {
    resolve_slash_command(unknown).map(Cow::Borrowed)
}

fn canonical_slash_command(command: &str) -> &str {
    match command {
        "/show-plan" => "/showplan",
        "/revert" => "/undo",
        _ => command,
    }
}

fn edit_distance_slash(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[n][m]
}

/// Static choices for commands that take a fixed set of arguments.

// ---------------------------------------------------------------------------
// InputHelper
// ---------------------------------------------------------------------------

/// Input helper with completion and hints.
pub struct InputHelper {
    /// Current working directory for file completion
    cwd: PathBuf,
    /// Dynamic completion data (configured providers, current models).
    /// Uses a `std::sync::RwLock` because `complete()` is synchronous.
    pub completion_data: std::sync::Arc<std::sync::RwLock<CompletionData>>,
}

impl InputHelper {
    pub fn new() -> Self {
        Self::with_data(std::sync::Arc::new(std::sync::RwLock::new(
            CompletionData::default(),
        )))
    }

    pub fn with_data(data: std::sync::Arc<std::sync::RwLock<CompletionData>>) -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            completion_data: data,
        }
    }

    // ------------------------------------------------------------------
    // Argument completion helpers
    // ------------------------------------------------------------------

    /// Complete the argument of a slash command.
    ///
    /// `cmd`     — the command token including the `/` (e.g. `"/provider"`)
    /// `partial` — what the user has typed so far for the argument
    fn complete_command_arg(&self, cmd: &str, partial: &str) -> (usize, Vec<Pair>) {
        // Whitespace such as U+3000 is wider than one byte.
        let token_start = partial
            .char_indices()
            .rev()
            .find(|(_, ch)| ch.is_whitespace())
            .map_or(0, |(idx, ch)| idx + ch.len_utf8());
        let token_partial = &partial[token_start..];
        let prior = partial[..token_start].trim_end();
        let prior_tokens: Vec<&str> = if prior.is_empty() {
            Vec::new()
        } else {
            prior.split_whitespace().collect()
        };
        if !crate::slash_catalog::command_in_this_build(cmd) {
            return (token_start, Vec::new());
        }
        let data = self.completion_data.read().ok();
        let data = data.as_deref();

        if cmd == "/embedding" {
            return arguments::complete_provider_slash_model(
                &prior_tokens,
                token_partial,
                token_start,
                &arguments::embedding_provider_choices(),
                arguments::embedding_models_for_provider,
            );
        }
        if cmd == "/image-generation" || cmd == "/image" {
            return arguments::complete_provider_slash_model(
                &prior_tokens,
                token_partial,
                token_start,
                arguments::IMAGE_PROVIDER_CHOICES,
                arguments::image_models_for_provider,
            );
        }
        if (cmd == "/image-generators" || cmd == "/images") && prior_tokens.is_empty() {
            let provider_choices =
                &arguments::IMAGE_PROVIDER_CHOICES[..arguments::IMAGE_PROVIDER_CHOICES.len() - 1];
            return (
                token_start,
                arguments::provider_choice_pairs(token_partial, provider_choices),
            );
        }
        if PROVIDER_ARG_CMDS.contains(&cmd) && prior_tokens.is_empty() {
            return (
                token_start,
                arguments::complete_provider_cmd_arg(data, token_partial),
            );
        }
        if MODEL_ARG_CMDS.contains(&cmd) && prior_tokens.is_empty() {
            return (
                token_start,
                arguments::complete_model_cmd_arg(data, token_partial),
            );
        }
        if cmd == "/local-model" {
            return (
                token_start,
                arguments::complete_local_model_arg(&prior_tokens, token_partial),
            );
        }
        if cmd == "/gateway" {
            return (
                token_start,
                arguments::complete_gateway_pairs(&prior_tokens, token_partial, data),
            );
        }
        if cmd == "/creds" {
            return (
                token_start,
                arguments::complete_creds_pairs(&prior_tokens, token_partial, data),
            );
        }
        if let Some(pairs) = arguments::complete_static_arg(cmd, &prior_tokens, token_partial) {
            return (token_start, pairs);
        }

        (token_start, Vec::new())
    }

    /// Get file completions for a partial path
    fn complete_path(&self, partial: &str) -> Vec<Pair> {
        let mut completions = Vec::new();

        let (dir, prefix) = if partial.contains('/') {
            let path = PathBuf::from(partial);
            if partial.ends_with('/') {
                (path, "".to_string())
            } else {
                let parent = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
                let prefix = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                (parent, prefix)
            }
        } else {
            (PathBuf::new(), partial.to_string())
        };

        let search_dir = if dir.as_os_str().is_empty() {
            self.cwd.clone()
        } else if dir.is_absolute() {
            dir.clone()
        } else {
            self.cwd.join(&dir)
        };

        if let Ok(entries) = std::fs::read_dir(&search_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(&prefix) && !name.starts_with('.') {
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    let suffix = if is_dir { "/" } else { "" };

                    let display = format!("{}{}", name, suffix);
                    let replacement = if dir.as_os_str().is_empty() {
                        display.clone()
                    } else {
                        format!("{}/{}", dir.display(), display)
                    };

                    completions.push(Pair {
                        display,
                        replacement,
                    });
                }
            }
        }

        completions.sort_by(|a, b| a.display.cmp(&b.display));
        completions
    }
}

impl Default for InputHelper {
    fn default() -> Self {
        Self::new()
    }
}

impl InputHelper {
    /// Compute completion candidates for the current line and cursor position.
    pub fn complete(&self, line: &str, pos: usize) -> (usize, Vec<Pair>) {
        let line_to_cursor = &line[..pos];

        if line_to_cursor.starts_with('/') {
            // Case A: still typing the command name (no space yet) — complete command names
            if !line_to_cursor.contains(' ') {
                let prefix = line_to_cursor;
                let matches: Vec<Pair> = slash_commands()
                    .iter()
                    .filter(|(cmd, _)| cmd.starts_with(prefix))
                    .map(|(cmd, desc)| Pair {
                        display: format!("{} - {}", cmd, desc),
                        replacement: format!("{} ", cmd),
                    })
                    .collect();
                return (0, matches);
            }

            // Case B: command name is complete, user is typing the argument
            if let Some(space_pos) = line_to_cursor.find(' ') {
                let cmd = &line_to_cursor[..space_pos];
                let partial_arg = &line_to_cursor[space_pos + 1..];

                let (arg_replace_start, pairs) = self.complete_command_arg(cmd, partial_arg);
                if !pairs.is_empty() {
                    return (space_pos + 1 + arg_replace_start, pairs);
                }
            }
        }

        // @ reference completion
        if let Some(at_pos) = line_to_cursor.rfind('@') {
            if at_pos == 0
                || line_to_cursor[..at_pos]
                    .chars()
                    .next_back()
                    .is_some_and(char::is_whitespace)
            {
                let partial = &line_to_cursor[at_pos + 1..];

                let mut completions = Vec::new();

                let specials = [
                    ("clipboard", "Paste clipboard content"),
                    ("diff", "Git diff"),
                    ("status", "Git status"),
                    ("artifact:latest", "Latest generated artifact"),
                    ("resource:", "MCP resource URI"),
                ];
                for (name, desc) in specials {
                    if name.starts_with(partial) {
                        completions.push(Pair {
                            display: format!("@{} - {}", name, desc),
                            replacement: format!("@{} ", name),
                        });
                    }
                }

                if let Ok(data) = self.completion_data.read() {
                    for artifact_ref in &data.artifact_refs {
                        if artifact_ref.starts_with(partial) {
                            completions.push(Pair {
                                display: format!("@{} - generated artifact", artifact_ref),
                                replacement: format!("@{} ", artifact_ref),
                            });
                        }
                    }
                    for resource_ref in &data.resource_refs {
                        if resource_ref.starts_with(partial) {
                            completions.push(Pair {
                                display: format!("@{} - MCP resource", resource_ref),
                                replacement: format!("@{} ", resource_ref),
                            });
                        }
                    }
                }

                let file_completions = self.complete_path(partial);
                for pair in file_completions {
                    completions.push(Pair {
                        display: format!("@{}", pair.display),
                        replacement: format!("@{}", pair.replacement),
                    });
                }

                return (at_pos, completions);
            }
        }

        (pos, Vec::new())
    }

    /// Compute a hint string for the current line/cursor position.
    pub fn hint(&self, line: &str, pos: usize) -> Option<String> {
        if line.starts_with('/') && pos == line.len() {
            // Still typing the command name — show the rest of the matching command
            if !line.contains(' ') {
                for (cmd, desc) in slash_commands() {
                    if cmd.starts_with(line) && *cmd != line {
                        let hint = &cmd[line.len()..];
                        return Some(format!("{} - {}", hint, desc));
                    }
                }
                return None;
            }

            // Command is complete; show argument placeholder
            if let Some(space_pos) = line.find(' ') {
                let cmd = &line[..space_pos];
                let after = &line[space_pos + 1..];

                // Only show hint when cursor is right after the space (empty arg)
                if after.is_empty()
                    && let Some(placeholder) = self.argument_hint(cmd)
                {
                    return Some(placeholder);
                }
            }
        }

        // Show hint for @ at end of line
        if line.ends_with('@') && pos == line.len() {
            return Some("<file|dir/|url|diff|status|artifact:|resource:>".to_string());
        }

        None
    }

    fn argument_hint(&self, cmd: &str) -> Option<String> {
        let active = self.active_provider_name();
        if PROVIDER_ARG_CMDS.contains(&cmd) {
            return Some(provider_argument_hint(&active));
        }
        if MODEL_ARG_CMDS.contains(&cmd) {
            return Some(model_argument_hint(&active));
        }
        static_argument_hint(cmd).map(str::to_owned)
    }

    fn active_provider_name(&self) -> String {
        let data = self.completion_data.read().ok();
        data.as_deref()
            .map(|data| data.active_provider.clone())
            .unwrap_or_default()
    }

    /// Apply lightweight syntax highlighting to an input line.
    pub fn highlight<'l>(&self, line: &'l str) -> Cow<'l, str> {
        if line.starts_with('/') {
            let cmd_end = line.find(' ').unwrap_or(line.len());
            let cmd = &line[..cmd_end];

            if slash_commands().iter().any(|(c, _)| *c == cmd) {
                return Cow::Owned(format!("\x1b[32m{}\x1b[0m{}", cmd, &line[cmd_end..]));
            }
            if slash_commands().iter().any(|(c, _)| c.starts_with(cmd)) {
                return Cow::Owned(format!("\x1b[33m{}\x1b[0m{}", cmd, &line[cmd_end..]));
            }
            return Cow::Owned(format!("\x1b[31m{}\x1b[0m{}", cmd, &line[cmd_end..]));
        }

        if line.contains('@') {
            let mut result = String::new();
            let mut in_ref = false;
            let chars = line.chars().peekable();
            let mut prev_was_space = true;

            for c in chars {
                if c == '@' && prev_was_space {
                    in_ref = true;
                    result.push_str("\x1b[36m@");
                } else if in_ref && c.is_whitespace() {
                    in_ref = false;
                    result.push_str("\x1b[0m");
                    result.push(c);
                } else {
                    result.push(c);
                }
                prev_was_space = c.is_whitespace();
            }

            if in_ref {
                result.push_str("\x1b[0m");
            }

            return Cow::Owned(result);
        }

        Cow::Borrowed(line)
    }
}

/// Multi-line input mode

/// Print the list of available slash commands
pub fn print_slash_commands() {
    println!("\nAvailable commands:");
    for (cmd, desc) in slash_commands() {
        println!("  \x1b[32m{:<15}\x1b[0m {}", cmd, desc);
    }
    println!("\nCommon flows:");
    println!("  Ask directly        Explain this code path");
    println!("  Attach files        Review @src/main.rs");
    println!("  Continue work       /resume or /sessions");
    println!("  Inspect tools       /tools");
    println!("\nContext references (use @ to attach):");
    println!("  \x1b[36m@file.rs\x1b[0m       Attach file content");
    println!("  \x1b[36m@src/\x1b[0m          List directory contents");
    println!("  \x1b[36m@https://...\x1b[0m   Fetch URL content");
    println!("  \x1b[36m@diff\x1b[0m          Attach git diff");
    println!("  \x1b[36m@status\x1b[0m        Attach git status");
    println!("  \x1b[36m@artifact:latest\x1b[0m Attach generated artifact");
    println!("  \x1b[36m@resource:<uri>\x1b[0m Attach MCP resource");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replacements_for(line: &str) -> Vec<String> {
        let helper = InputHelper::new();
        let (_, pairs) = helper.complete(line, line.len());
        pairs.into_iter().map(|pair| pair.replacement).collect()
    }

    fn offered(line: &str) -> Vec<String> {
        replacements_for(line)
            .into_iter()
            .map(|item| item.trim_end().to_string())
            .collect()
    }

    /// Everything the CLI offers runs in the build under test: providers,
    /// embedding providers, `--backend` values and slash commands that need
    /// a feature. Each offer that would not run is listed with the reason.
    #[test]
    fn everything_the_cli_offers_runs_in_this_build() {
        use caliche::llm::ProviderType;
        let mut not_runnable = Vec::new();
        for name in offered("/provider ") {
            if let Err(err) = ProviderType::parse_in_build(&name) {
                not_runnable.push(format!("/provider {name}: {err}"));
            }
        }
        for name in offered("/embedding ") {
            if name == "clear" {
                continue;
            }
            match ProviderType::parse_in_build(&name) {
                Ok(provider)
                    if caliche::agent::service::default_embedding_model(provider).is_some() => {}
                Ok(_) => not_runnable.push(format!("/embedding {name}: no embeddings")),
                Err(err) => not_runnable.push(format!("/embedding {name}: {err}")),
            }
        }
        for name in offered("/local-model download --backend ") {
            #[cfg(feature = "local-inference-base")]
            if let Err(err) = caliche::llm::LocalInferenceProvider::parse_backend_in_build(&name) {
                not_runnable.push(format!("--backend {name}: {err}"));
            }
            #[cfg(not(feature = "local-inference-base"))]
            not_runnable.push(format!(
                "--backend {name}: no local inference in this build"
            ));
        }
        for name in offered("/") {
            let left_out = crate::slash_catalog::FEATURE_GATED_COMMANDS
                .iter()
                .any(|(gated, built)| *gated == name && !built);
            if left_out {
                not_runnable.push(format!("{name}: needs a feature this build left out"));
            }
        }

        assert_eq!(not_runnable, Vec::<String>::new());
    }

    /// Without local inference, `/local-model` is not offered: not in
    /// command completion, `/help` groups or argument completion. Typed
    /// anyway, it says how to build it in.
    #[cfg(not(feature = "local-inference-base"))]
    #[tokio::test]
    async fn local_model_is_not_offered_without_local_inference() {
        let in_help: Vec<&str> = crate::slash_catalog::slash_command_groups()
            .iter()
            .flat_map(|(_, commands)| commands.iter().copied())
            .filter(|command| command.starts_with("/local"))
            .collect();
        let typed = crate::core::slash_commands::SlashCommandExecutor::handle_local_model(&[
            "list".to_string(),
        ])
        .await
        .expect("/local-model");

        assert_eq!(offered("/local"), Vec::<String>::new());
        assert_eq!(
            offered("/local-model download --backend "),
            Vec::<String>::new()
        );
        assert_eq!(in_help, Vec::<&str>::new());
        assert_eq!(static_argument_hint("/local-model"), None);
        assert_eq!(
            format!("{typed:?}"),
            "Error(\"Local inference is not in this build. Rebuild with --features local-inference.\")"
        );
    }

    /// U+3000 is a three-byte whitespace; the argument token starts after it.
    #[test]
    fn argument_completion_starts_after_multibyte_whitespace() {
        let line = "/provider \u{3000}op";
        let (start, pairs) = InputHelper::new().complete(line, line.len());
        assert_eq!(start, 13);
        assert!(!pairs.is_empty());
    }

    /// `@` at byte 3 is the third character; the character before it is the space.
    #[test]
    fn at_reference_completes_after_multibyte_text() {
        let line = "é @di";
        let (start, pairs) = InputHelper::new().complete(line, line.len());
        let replacements: Vec<String> = pairs.into_iter().map(|pair| pair.replacement).collect();
        assert_eq!((start, replacements), (3, vec!["@diff ".to_string()]));
    }

    #[test]
    fn test_parse_context_references() {
        let (clean, refs) = parse_context_references("Look at @src/main.rs and fix it");
        assert_eq!(clean, "Look at  and fix it");
        assert_eq!(refs.len(), 1);

        let (clean, refs) = parse_context_references("@diff shows the changes");
        assert_eq!(clean, "shows the changes");
        assert_eq!(refs.len(), 1);

        let (clean, refs) = parse_context_references("Check @file1.rs and @file2.rs");
        assert_eq!(clean, "Check  and");
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_context_reference_parse() {
        assert!(matches!(
            ContextReference::parse("diff"),
            Some(ContextReference::GitDiff)
        ));

        assert!(matches!(
            ContextReference::parse("https://example.com"),
            Some(ContextReference::Url(_))
        ));

        assert!(matches!(
            ContextReference::parse("src/main.rs"),
            Some(ContextReference::File(_))
        ));

        assert!(matches!(
            ContextReference::parse("src/"),
            Some(ContextReference::Directory(_))
        ));
    }

    #[test]
    fn test_resolve_slash_command() {
        // Prefix matches
        assert_eq!(resolve_slash_command("/hel"), Some("/help"));
        assert_eq!(resolve_slash_command("/statu"), Some("/status"));
        // Exact match returns None (caller handles exact)
        assert_eq!(resolve_slash_command("/help"), None);
        // Unknown returns None
        assert_eq!(resolve_slash_command("/xyzz"), None);
    }

    #[test]
    fn provider_completion_uses_canonical_provider_ids() {
        let replacements = replacements_for("/provider azure");
        #[cfg(feature = "azure-openai")]
        assert_eq!(replacements, vec!["azure-openai ".to_string()]);
        #[cfg(not(feature = "azure-openai"))]
        assert_eq!(replacements, Vec::<String>::new());
    }

    #[test]
    fn provider_completion_offers_only_providers_this_build_has() {
        let replacements = replacements_for("/provider local");
        #[cfg(not(feature = "local-inference-base"))]
        assert_eq!(replacements, Vec::<String>::new());
        #[cfg(feature = "local-inference-base")]
        assert_eq!(replacements, vec!["local-inference ".to_string()]);
    }

    #[test]
    fn embedding_completion_includes_openrouter_and_cloudflare() {
        let replacements = replacements_for("/embedding ");
        assert!(replacements.iter().any(|item| item == "openrouter "));
        assert!(
            replacements
                .iter()
                .any(|item| item == "cloudflare-ai-gateway ")
        );
    }

    #[test]
    fn embedding_slash_form_completes_provider_models() {
        let replacements = replacements_for("/embedding openrouter/text");
        assert!(
            replacements
                .iter()
                .any(|item| item == "openrouter/openai/text-embedding-3-small ")
        );
    }

    #[test]
    fn image_completion_includes_routed_providers_and_models() {
        let providers = replacements_for("/image-generation ");
        assert!(providers.iter().any(|item| item == "openrouter "));
        assert!(
            providers
                .iter()
                .any(|item| item == "cloudflare-ai-gateway ")
        );

        let models = replacements_for("/image-generation cloudflare-ai-gateway/gemini");
        assert!(
            models.iter().any(
                |item| item == "cloudflare-ai-gateway/google-ai-studio/gemini-2.5-flash-image "
            )
        );
    }

    #[test]
    fn image_generators_provider_aliases_complete() {
        let replacements = replacements_for("/images cloud");
        assert!(
            replacements
                .iter()
                .any(|item| item == "cloudflare-ai-gateway ")
        );
    }
}
