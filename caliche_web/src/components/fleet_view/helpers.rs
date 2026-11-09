use crate::api::{ModelInfo, ProviderCatalogEntry};
use crate::catalog_select::is_selectable_model_id;
use crate::state::agents::AgentStatus;

pub const INPUT_STYLE: &str = "background: var(--input-bg, var(--background)); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3em 0.75ch; font-family: inherit; font-size: var(--font-size-sm); box-sizing: border-box; width: 100%;";
pub const BTN_MUTED: &str = "padding: 0.2em 1.25ch; font-size: var(--font-size-xs); background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); cursor: pointer;";
pub const BTN_DANGER: &str = "padding: 0.2em 1.25ch; font-size: var(--font-size-xs); background: transparent; border: 1px solid var(--error); color: var(--error); cursor: pointer;";
pub const BTN_ACCENT: &str = "padding: 0.2em 1.25ch; font-size: var(--font-size-xs); background: transparent; border: 1px solid var(--accent); color: var(--accent); cursor: pointer;";

pub fn status_color(status: &str) -> &'static str {
    match status {
        "running" => "var(--accent)",
        "starting" => "var(--warning, #ffcc00)",
        "failed" => "var(--error)",
        _ => "var(--muted-color)",
    }
}

pub fn status_icon(status: &str) -> &'static str {
    match status {
        "running" => "●",
        "starting" => "○",
        "failed" => "✗",
        _ => "○",
    }
}

pub fn agent_status_color(s: &AgentStatus) -> &'static str {
    match s {
        AgentStatus::Pending => "var(--warning, #ffcc00)",
        AgentStatus::Running => "var(--accent)",
        AgentStatus::Completed => "var(--success, #00ff41)",
        AgentStatus::Failed => "var(--error)",
        AgentStatus::Cancelled => "var(--muted-color)",
    }
}

pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

pub fn snippet(s: &str) -> String {
    truncate(s.trim(), 120)
}

pub fn models_for_provider(models: &[ModelInfo], provider: &str) -> Vec<ModelInfo> {
    if provider.trim().is_empty() {
        return models
            .iter()
            .filter(|model| is_selectable_model_id(&model.id))
            .cloned()
            .collect();
    }

    models
        .iter()
        .filter(|model| model.provider_id == provider && is_selectable_model_id(&model.id))
        .cloned()
        .collect()
}

pub fn default_provider_and_model(
    providers: &[ProviderCatalogEntry],
    models: &[ModelInfo],
    current_provider: Option<&str>,
    current_model: Option<&str>,
) -> (String, String) {
    let provider = current_provider
        .filter(|provider| !provider.trim().is_empty())
        .map(str::to_string)
        .or_else(|| providers.first().map(|provider| provider.id.clone()))
        .unwrap_or_default();

    let model = current_model
        .filter(|model| !model.trim().is_empty())
        .map(str::to_string)
        .or_else(|| crate::catalog_select::first_selectable_model_id(models, &provider))
        .unwrap_or_default();

    (provider, model)
}
