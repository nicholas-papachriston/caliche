use axum::extract::Path as AxumPath;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use super::shared::internal_error;

pub(crate) async fn completions_handler(
    AxumPath(shell): AxumPath<String>,
) -> Result<Response, (StatusCode, String)> {
    use crate::commands::Args;
    use clap::CommandFactory;
    use clap_complete::{Shell, generate};

    let shell_type = match shell.trim().to_ascii_lowercase().as_str() {
        "bash" => Shell::Bash,
        "zsh" => Shell::Zsh,
        "fish" => Shell::Fish,
        "powershell" | "power-shell" | "pwsh" => Shell::PowerShell,
        "elvish" => Shell::Elvish,
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unsupported completion shell: {other}"),
            ));
        }
    };

    let mut cmd = Args::command();
    let mut buf = Vec::new();
    generate(shell_type, &mut cmd, "caliche", &mut buf);
    let body = String::from_utf8(buf).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("completion output was not utf-8: {e}"),
        )
    })?;
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response())
}

pub(crate) async fn man_page_handler() -> Result<Response, (StatusCode, String)> {
    use crate::commands::Args;
    use clap::CommandFactory;
    use clap_mangen::Man;

    let cmd = Args::command();
    let mut buf = Vec::new();
    Man::new(cmd).render(&mut buf).map_err(internal_error)?;
    let body = String::from_utf8(buf).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("man page output was not utf-8: {e}"),
        )
    })?;
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response())
}

#[derive(Debug, Serialize)]
pub(crate) struct ManPageDto {
    name: String,
    filename: String,
    content: String,
}

pub(crate) async fn man_pages_handler() -> Result<axum::Json<Vec<ManPageDto>>, (StatusCode, String)>
{
    use crate::commands::Args;
    use clap::CommandFactory;

    let cmd = Args::command();
    let mut pages = Vec::new();
    collect_man_pages(&cmd, "", &mut pages)?;
    Ok(axum::Json(pages))
}

pub(crate) fn collect_man_pages(
    cmd: &clap::Command,
    prefix: &str,
    pages: &mut Vec<ManPageDto>,
) -> Result<(), (StatusCode, String)> {
    use clap_mangen::Man;

    let name = if prefix.is_empty() {
        cmd.get_name().to_string()
    } else {
        format!("{}-{}", prefix, cmd.get_name())
    };

    let mut buf = Vec::new();
    let renamed = cmd.clone().bin_name(name.clone());
    Man::new(renamed)
        .title(name.to_uppercase())
        .render(&mut buf)
        .map_err(internal_error)?;
    let content = String::from_utf8(buf).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("man page output was not utf-8: {e}"),
        )
    })?;
    pages.push(ManPageDto {
        filename: format!("{name}.1"),
        name: name.clone(),
        content,
    });

    for sub in cmd.get_subcommands() {
        if sub.get_name() != "help" {
            collect_man_pages(sub, &name, pages)?;
        }
    }

    Ok(())
}

#[derive(Debug, Serialize)]
pub(crate) struct LspStatusResponse {
    enabled: bool,
    workspace: String,
    status_text: String,
    languages: Vec<LspLanguageDto>,
}

#[derive(Debug, Serialize)]
pub(crate) struct LspLanguageDto {
    language: String,
    available: bool,
    status: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LspInstallRequest {
    language: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct LspInstallResponse {
    language: String,
    installed: bool,
}

pub(crate) fn lsp_manager_for_cwd() -> crate::lsp::LspManager {
    crate::lsp::LspManager::new(crate::lsp::LspConfig::default())
}

pub(crate) fn lsp_language_dtos(manager: &crate::lsp::LspManager) -> Vec<LspLanguageDto> {
    let mut languages: Vec<LspLanguageDto> = manager
        .supported_languages()
        .into_iter()
        .map(|language| LspLanguageDto {
            language: language.to_string(),
            available: manager.is_server_available(&language),
            status: "stopped".to_string(),
        })
        .collect();
    languages.sort_by(|a, b| a.language.cmp(&b.language));
    languages
}

pub(crate) async fn lsp_status_handler()
-> Result<axum::Json<LspStatusResponse>, (StatusCode, String)> {
    let manager = lsp_manager_for_cwd();
    let workspace = std::env::current_dir()
        .map_err(internal_error)?
        .display()
        .to_string();
    Ok(axum::Json(LspStatusResponse {
        enabled: manager.is_enabled(),
        workspace,
        status_text: crate::lsp::format_lsp_status(&manager),
        languages: lsp_language_dtos(&manager),
    }))
}

pub(crate) async fn lsp_languages_handler()
-> Result<axum::Json<Vec<LspLanguageDto>>, (StatusCode, String)> {
    let manager = lsp_manager_for_cwd();
    Ok(axum::Json(lsp_language_dtos(&manager)))
}

pub(crate) async fn lsp_install_handler(
    axum::Json(request): axum::Json<LspInstallRequest>,
) -> Result<axum::Json<LspInstallResponse>, (StatusCode, String)> {
    crate::lsp::install_language_server(&request.language)
        .await
        .map_err(internal_error)?;
    Ok(axum::Json(LspInstallResponse {
        language: request.language,
        installed: true,
    }))
}

#[derive(Debug, Serialize)]
pub(crate) struct AgentProfileSummaryDto {
    name: String,
    description: Option<String>,
    mode: String,
    model: Option<String>,
    tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct AgentProfileDto {
    name: String,
    description: Option<String>,
    mode: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    steps: Option<u32>,
    color: Option<String>,
    tools: Option<Vec<String>>,
    system_prompt: Option<String>,
    extra: serde_norway::Value,
}

pub(crate) async fn agent_profile_search_path()
-> Result<Vec<caliche::agent::profile::ProfileDir>, (StatusCode, String)> {
    use crate::config::ConfigManager;
    use crate::utils::find_git_root;
    use caliche::agent::profile::default_search_path;

    let cwd = std::env::current_dir().map_err(internal_error)?;
    let git_root = find_git_root(&cwd);
    let mut paths = default_search_path(git_root.as_ref());
    let mut manager = ConfigManager::with_standard_sources();
    if let Ok(config) = manager.load().await
        && let Some(dirs) = &config.agents_dirs
    {
        paths.extend(
            dirs.iter()
                .cloned()
                .map(caliche::agent::profile::ProfileDir::user),
        );
    }
    Ok(paths)
}

pub(crate) async fn load_agent_profiles() -> Result<
    std::collections::HashMap<String, caliche::agent::profile::AgentProfile>,
    (StatusCode, String),
> {
    use caliche::agent::profile::load_profiles_from_search_path;

    let paths = agent_profile_search_path().await?;
    load_profiles_from_search_path(&paths).map_err(internal_error)
}

pub(crate) async fn agent_profiles_handler()
-> Result<axum::Json<Vec<AgentProfileSummaryDto>>, (StatusCode, String)> {
    let registry = load_agent_profiles().await?;
    let mut profiles: Vec<AgentProfileSummaryDto> = registry
        .into_iter()
        .map(|(name, profile)| {
            let tools = profile.enabled_tools();
            AgentProfileSummaryDto {
                name,
                description: profile.description,
                mode: profile.mode.unwrap_or_else(|| "primary".to_string()),
                model: profile.model,
                tools,
            }
        })
        .collect();
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(axum::Json(profiles))
}

pub(crate) async fn agent_profile_handler(
    AxumPath(name): AxumPath<String>,
) -> Result<axum::Json<AgentProfileDto>, (StatusCode, String)> {
    let registry = load_agent_profiles().await?;
    let profile = registry
        .get(&name)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("profile not found: {name}")))?;
    Ok(axum::Json(AgentProfileDto {
        name,
        description: profile.description.clone(),
        mode: profile.mode.clone(),
        model: profile.model.clone(),
        temperature: profile.temperature,
        steps: profile.steps,
        color: profile.color.clone(),
        tools: profile.enabled_tools(),
        system_prompt: profile.system_prompt().map(ToOwned::to_owned),
        extra: serde_norway::to_value(&profile.extra).unwrap_or(serde_norway::Value::Null),
    }))
}
pub(crate) async fn doctor_handler() -> Result<Response, (StatusCode, String)> {
    let report = crate::handlers::doctor::doctor_report_to_string()
        .await
        .map_err(internal_error)?;
    Ok((
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        report,
    )
        .into_response())
}
