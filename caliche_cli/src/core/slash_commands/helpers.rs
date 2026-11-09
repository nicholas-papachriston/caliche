use crate::core::state::InteractiveState;
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::modes::InteractionMode;
use caliche::harness::{SessionDiffSummary, SnapshotChangeKind};
use caliche::llm::ProviderType;
use chrono::{DateTime, Local, Utc};
use std::io::IsTerminal;
#[cfg(feature = "local-inference-base")]
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResumeRequest {
    PromptSelection,
    Last,
    ById(String),
}

pub(crate) async fn cli_model_catalog_options() -> caliche::llm::ModelCatalogOptions {
    let manager = crate::storage::SecureCredentialManager::new().await.ok();
    caliche::llm::ModelCatalogOptions::from_environment().with_credential_provider(
        crate::storage::session_credential_provider(manager.as_ref()).await,
    )
}

pub(crate) fn parse_resume_request(id: Option<&str>) -> ResumeRequest {
    match id.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("last") => ResumeRequest::Last,
        Some(value) if !value.is_empty() => ResumeRequest::ById(value.to_string()),
        _ => ResumeRequest::PromptSelection,
    }
}

pub(crate) fn image_model_description(
    provider: caliche::llm::ProviderType,
    model: &str,
) -> &'static str {
    match provider {
        caliche::llm::ProviderType::OpenAI if model.starts_with("gpt-image") => {
            "OpenAI image generation model"
        }
        caliche::llm::ProviderType::OpenAI if model.starts_with("dall-e") => {
            "Legacy OpenAI image generation model"
        }
        caliche::llm::ProviderType::Google if model.contains("3-pro-image") => {
            "Google Gemini preview image generation model for higher fidelity"
        }
        caliche::llm::ProviderType::Google if model.contains("3.1-flash-image") => {
            "Google Gemini preview image generation model optimized for speed"
        }
        caliche::llm::ProviderType::Google => "Google Gemini image generation model",
        caliche::llm::ProviderType::OpenRouter => {
            "OpenRouter image generation model via chat completions"
        }
        caliche::llm::ProviderType::CloudflareAIGateway => {
            "Cloudflare AI Gateway routed image generation model"
        }
        _ => "Image generation model",
    }
}

pub(crate) fn format_mode_contract(mode: InteractionMode) -> String {
    let mode_summary = match mode {
        InteractionMode::Agent => "Agent mode: read/write tools enabled; full execution.",
        InteractionMode::Plan => "Plan mode: read-only tools plus `write_plan`; no file edits.",
        InteractionMode::Ask => "Ask mode: read-only analysis; no file edits.",
    };

    format!(
        "Current mode: {}\n{}\n\nMode transitions:\n  /plan  -> read-only planning\n  /ask   -> read-only Q&A\n  /agent -> full execution",
        mode.name(),
        mode_summary
    )
}

pub(crate) fn mode_switch_message(mode: InteractionMode) -> &'static str {
    match mode {
        InteractionMode::Agent => {
            "Switched to Agent mode. Full capabilities enabled. Use /plan or /ask for read-only workflows."
        }
        InteractionMode::Plan => {
            "Switched to Plan mode. Write tools disabled except write_plan. Use /showplan to review and /agent to execute."
        }
        InteractionMode::Ask => {
            "Switched to Ask mode. Read-only analysis enabled. Use /agent to resume edits and execution."
        }
    }
}

pub(crate) fn supports_interactive_selection() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

pub(crate) fn prompt_select_index(
    prompt: &str,
    items: &[String],
    default: usize,
) -> Result<Option<usize>> {
    use dialoguer::{FuzzySelect, theme::ColorfulTheme};

    if items.is_empty() {
        return Ok(None);
    }

    let mut choices = items.to_vec();
    choices.push("Done - keep current value".to_string());

    let selection = match FuzzySelect::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .items(&choices)
        .default(default.min(choices.len().saturating_sub(1)))
        .interact()
        .map_err(anyhow::Error::from)
    {
        Ok(selection) => selection,
        // Ctrl+C keeps the current value, like "Done".
        Err(err) if crate::credentials::is_prompt_cancelled(&err) => return Ok(None),
        Err(err) => return Err(err),
    };

    if selection >= items.len() {
        Ok(None)
    } else {
        Ok(Some(selection))
    }
}

pub(crate) fn short_session_id(id: &str) -> String {
    id.chars().take(8).collect()
}

pub(crate) fn format_updated(updated_at: DateTime<Utc>) -> String {
    updated_at
        .with_timezone(&Local)
        .format("%Y-%m-%d %H:%M")
        .to_string()
}

pub(crate) fn cwd_tail(path: &std::path::Path) -> String {
    let parts: Vec<String> = path
        .components()
        .rev()
        .take(2)
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.into_iter().rev().collect::<Vec<_>>().join("/")
    }
}

pub(crate) async fn replace_service_memory_with_history(
    service: &mut AgentService,
    history: Vec<caliche::messages::Message>,
) -> Result<()> {
    use caliche::agent::memory::{
        ApproximateTokenCounter, BufferMemory, BufferMemoryConfig, MemoryStrategy,
    };
    use std::sync::Arc;

    let token_counter: Arc<dyn caliche::agent::memory::TokenCounter> =
        Arc::new(ApproximateTokenCounter::default());
    let memory_config = BufferMemoryConfig::default().with_max_tokens(16000);
    let mut memory = BufferMemory::new(memory_config, token_counter);
    if !history.is_empty() {
        memory.add_messages(history).await?;
    }
    service.set_memory_strategy(Some(Box::new(memory)));
    Ok(())
}

pub(crate) async fn apply_runtime_settings(
    state: &mut InteractiveState,
    service: &mut AgentService,
    provider: Option<ProviderType>,
    model: Option<String>,
    mode: InteractionMode,
) -> Result<()> {
    let mut settings = service.snapshot_settings();
    if let Some(provider) = provider {
        settings.provider = provider;
    }
    if let Some(model) = model {
        settings.model = model;
    }
    settings.system_prompt = Some(state.original_system_prompt.clone());
    settings.mode = Some(mode);
    let section_before = service.project_context_section().map(str::to_owned);
    service.apply_settings(settings).await?;
    follow_project_section(state, section_before, service);
    Ok(())
}

/// A model or window change can size the project context again. Put the
/// new section in the REPL's base prompt too, so later turns keep it.
pub(crate) fn follow_project_section(
    state: &mut InteractiveState,
    section_before: Option<String>,
    service: &AgentService,
) {
    if let (Some(stale), Some(current)) = (section_before, service.project_context_section())
        && stale != current
    {
        state.original_system_prompt = state.original_system_prompt.replacen(&stale, current, 1);
    }
}

pub(crate) fn sync_runtime_metadata(state: &mut InteractiveState, service: &AgentService) {
    let used = state.token_budget.used_tokens;
    let provider = service.provider().to_string();
    let model = service.model().to_string();
    state.conversation_export.provider = provider.clone();
    state.conversation_export.model = model.clone();
    let mut token_budget = crate::cost::TokenBudget::new(&provider, &model);
    if let Some(context_window) = service.context_window().and_then(|w| u64::try_from(w).ok()) {
        token_budget = token_budget.with_max_tokens(context_window);
    }
    state.token_budget = token_budget;
    state.token_budget.set_used(used);
}

#[derive(Debug, Clone)]
#[cfg(feature = "local-inference-base")]
pub(crate) struct LocalModelDownloadRequest {
    pub(crate) repo: String,
    pub(crate) revision: Option<String>,
    pub(crate) backend: Option<caliche::llm::InferenceBackendType>,
    pub(crate) model_dir: Option<PathBuf>,
    pub(crate) hf_cache_dir: Option<PathBuf>,
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn local_model_usage() -> String {
    let backends = caliche::llm::LocalInferenceProvider::backend_names().join("|");
    format!(
        "Usage: /local-model download <huggingface_repo> [--backend <{backends}>] [--dir <path>] [--revision <rev>] [--hf-cache-dir <path>]\n       /local-model list [--tag <tag>]\n       /local-model fit [catalog-id]\n       /local-model backends"
    )
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.1} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.1} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.1} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn parse_local_model_list_tags(args: &[String]) -> Vec<String> {
    let mut tags = Vec::new();
    let mut index = 1usize;
    while index < args.len() {
        let arg = &args[index];
        if let Some(value) = arg.strip_prefix("--tag=") {
            if !value.is_empty() {
                tags.push(value.to_string());
            }
            index += 1;
        } else if arg == "--tag" {
            if let Some(value) = args.get(index + 1) {
                tags.push(value.clone());
                index += 2;
            } else {
                break;
            }
        } else {
            // Bare tokens after `list` are treated as tags.
            tags.push(arg.clone());
            index += 1;
        }
    }
    tags
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn format_model_fit_report(fits: &[caliche::llm::ModelFit]) -> String {
    use caliche::llm::FitStatus;
    let mut lines = Vec::with_capacity(fits.len().saturating_add(2));
    lines.push("Local model fit (memory budget):".to_string());
    for fit in fits {
        let status = match fit.status {
            FitStatus::FitsComfortably => "OK",
            FitStatus::FitsTight => "TIGHT",
            FitStatus::ExceedsMemory => "EXCEEDS",
            FitStatus::Unknown => "UNKNOWN",
        };
        lines.push(format!(
            "  [{status}] {} ({})  weights~{}  runtime~{}  n_ctx={}  max_safe_n_ctx={}",
            fit.model_id,
            fit.display_name,
            format_bytes(fit.estimated_weight_bytes),
            format_bytes(fit.estimated_runtime_bytes),
            fit.recommended_n_ctx,
            fit.max_safe_n_ctx
        ));
        for reason in &fit.reasons {
            lines.push(format!("           {reason}"));
        }
    }
    lines.join("\n")
}

#[derive(Default)]
#[cfg(feature = "local-inference-base")]
struct LocalModelDownloadParse {
    repo: Option<String>,
    revision: Option<String>,
    backend: Option<caliche::llm::InferenceBackendType>,
    model_dir: Option<PathBuf>,
    hf_cache_dir: Option<PathBuf>,
}

#[cfg(feature = "local-inference-base")]
impl LocalModelDownloadParse {
    fn apply_inline_flag(&mut self, arg: &str) -> Result<bool> {
        if let Some(value) = arg.strip_prefix("--dir=") {
            self.model_dir = Some(PathBuf::from(required_inline_value("--dir", value)?));
            return Ok(true);
        }
        if let Some(value) = arg.strip_prefix("--model-dir=") {
            self.model_dir = Some(PathBuf::from(required_inline_value("--model-dir", value)?));
            return Ok(true);
        }
        if let Some(value) = arg.strip_prefix("--revision=") {
            self.revision = Some(required_inline_value("--revision", value)?);
            return Ok(true);
        }
        if let Some(value) = arg.strip_prefix("--backend=") {
            self.backend = Some(parse_local_inference_backend(&required_inline_value(
                "--backend",
                value,
            )?)?);
            return Ok(true);
        }
        if let Some(value) = arg.strip_prefix("--hf-cache-dir=") {
            self.hf_cache_dir = Some(PathBuf::from(required_inline_value(
                "--hf-cache-dir",
                value,
            )?));
            return Ok(true);
        }
        Ok(false)
    }

    fn apply_switch_or_repo(&mut self, args: &[String], index: &mut usize) -> Result<()> {
        let arg = &args[*index];
        match arg.as_str() {
            "--dir" | "--model-dir" => {
                *index += 1;
                self.model_dir = Some(PathBuf::from(required_arg_value(args, *index, arg)?));
            }
            "--revision" => {
                *index += 1;
                self.revision = Some(required_arg_value(args, *index, arg)?);
            }
            "--backend" => {
                *index += 1;
                self.backend = Some(parse_local_inference_backend(&required_arg_value(
                    args, *index, arg,
                )?)?);
            }
            "--hf-cache-dir" => {
                *index += 1;
                self.hf_cache_dir = Some(PathBuf::from(required_arg_value(args, *index, arg)?));
            }
            "--help" | "-h" => {
                return Err(anyhow::anyhow!(local_model_usage()));
            }
            value if value.starts_with('-') => {
                return Err(anyhow::anyhow!(
                    "unknown /local-model download option '{value}'. {}",
                    local_model_usage()
                ));
            }
            value => {
                if self.repo.replace(value.to_string()).is_some() {
                    return Err(anyhow::anyhow!(
                        "/local-model download accepts one HuggingFace repository. {}",
                        local_model_usage()
                    ));
                }
            }
        }
        Ok(())
    }

    fn into_request(self) -> Result<LocalModelDownloadRequest> {
        let repo = self.repo.ok_or_else(|| {
            anyhow::anyhow!(
                "missing HuggingFace repository for /local-model download. {}",
                local_model_usage()
            )
        })?;
        Ok(LocalModelDownloadRequest {
            repo,
            revision: self.revision,
            backend: self.backend,
            model_dir: self.model_dir,
            hf_cache_dir: self.hf_cache_dir,
        })
    }
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn parse_local_model_download_args(
    args: &[String],
) -> Result<LocalModelDownloadRequest> {
    let mut parsed = LocalModelDownloadParse::default();
    let mut index = 1usize;

    while index < args.len() {
        if !parsed.apply_inline_flag(&args[index])? {
            parsed.apply_switch_or_repo(args, &mut index)?;
        }
        index += 1;
    }

    parsed.into_request()
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn parse_local_inference_backend(
    backend: &str,
) -> Result<caliche::llm::InferenceBackendType> {
    caliche::llm::LocalInferenceProvider::parse_backend_in_build(backend)
        .map_err(|err| anyhow::anyhow!(err))
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn required_arg_value(args: &[String], index: usize, flag: &str) -> Result<String> {
    let value = args
        .get(index)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing value for {flag}. {}", local_model_usage()))?;

    Ok(value.to_string())
}

#[cfg(feature = "local-inference-base")]
pub(crate) fn required_inline_value(flag: &str, value: &str) -> Result<String> {
    if value.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "missing value for {flag}. {}",
            local_model_usage()
        ));
    }

    Ok(value.to_string())
}

pub(crate) fn format_session_diff_for_repl(s: &SessionDiffSummary) -> String {
    use std::fmt::Write;

    let mut out = String::new();

    if !s.can_undo && !s.can_redo && s.messages.is_empty() && s.reverted_messages.is_empty() {
        let _ = writeln!(
            out,
            "No recorded file change sets yet.\n\
             Per-message /undo tracks write_file and edit_file during agent turns.\n\
             Use /diff <path> for a unified diff of a modified file."
        );
        return out;
    }

    let _ = writeln!(
        out,
        "active change sets: {} | files touched: {} | undo: {} | redo: {} ({} in redo queue)",
        s.active_change_sets,
        s.files_changed.len(),
        if s.can_undo { "yes" } else { "no" },
        if s.can_redo { "yes" } else { "no" },
        s.reverted_change_sets
    );
    let _ = writeln!(out);

    if !s.messages.is_empty() {
        let _ = writeln!(out, "Active (oldest first):");
        for msg in &s.messages {
            let id_str = msg.message_id.to_string();
            let head = id_str
                .split_once('-')
                .map(|(a, _)| a)
                .unwrap_or(id_str.as_str());
            let _ = writeln!(out, "  message {}... ({} files)", head, msg.changed_files);
            for snap in &msg.snapshots {
                let kind = match snap.change {
                    SnapshotChangeKind::Created => "created",
                    SnapshotChangeKind::Modified => "modified",
                    SnapshotChangeKind::Deleted => "deleted",
                };
                let _ = writeln!(out, "    - {} ({})", snap.path.display(), kind);
            }
        }
        let _ = writeln!(out);
    }

    if !s.reverted_messages.is_empty() {
        let _ = writeln!(out, "Redo queue (most recent last):");
        for msg in &s.reverted_messages {
            let id_str = msg.message_id.to_string();
            let head = id_str
                .split_once('-')
                .map(|(a, _)| a)
                .unwrap_or(id_str.as_str());
            let _ = writeln!(
                out,
                "  message {}... ({} files) — /redo to restore",
                head, msg.changed_files
            );
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(
        out,
        "Use /diff <path> for a unified diff. /undo reverts the latest change set."
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    #[test]
    fn parse_resume_request_handles_all_variants() {
        assert_eq!(parse_resume_request(None), ResumeRequest::PromptSelection);
        assert_eq!(
            parse_resume_request(Some("")),
            ResumeRequest::PromptSelection
        );
        assert_eq!(
            parse_resume_request(Some("   ")),
            ResumeRequest::PromptSelection
        );
        assert_eq!(parse_resume_request(Some("last")), ResumeRequest::Last);
        assert_eq!(parse_resume_request(Some("LAST")), ResumeRequest::Last);
        assert_eq!(
            parse_resume_request(Some("abc123")),
            ResumeRequest::ById("abc123".to_string())
        );
    }

    #[test]
    fn formatting_helpers_are_stable() {
        assert_eq!(short_session_id("123456789"), "12345678");
        assert_eq!(short_session_id("123"), "123");
        assert_eq!(
            cwd_tail(std::path::Path::new("/tmp/project")),
            "tmp/project"
        );

        let Ok(ts) = "2026-02-24T12:30:00Z".parse::<DateTime<Utc>>() else {
            panic!("test fixture: RFC3339 datetime must parse");
        };
        let output = format_updated(ts);
        assert!(output.contains("2026-02-24"));
    }
}
