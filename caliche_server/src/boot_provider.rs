//! Provider selection shared by `caliche --server`, `caliche doctor` and the
//! standalone `caliche_server` binary.
//!
//! Doctor reports this choice. `--server` and `caliche_server` apply it. They
//! must name the same provider.
//!
//! Precedence:
//! 1. `--provider` (`flag`)
//! 2. Durable `settings.json` beside `mcp.json` when it has a provider and
//!    model (`last`). The REPL writes that file when the provider or model
//!    changes. In the CLI, SQLite `last_provider` / `last_model` is the
//!    fallback when that file is absent. This wins over `AGENT_PROVIDER` and
//!    `DEFAULT_PROVIDER`, unless `--model` is set.
//! 3. `AGENT_PROVIDER`, else `DEFAULT_PROVIDER` (`env`)
//! 4. built-in `openrouter` (`default`)
//!
//! Model and temperature use the same stack, per field: CLI flag, then the
//! durable record, then the provider env default. `--provider` does not drop
//! a saved model when the record's provider matches. `--model` replaces it.
//! `--temperature` replaces the saved temperature.
//!
//! Project `caliche.toml` is not this choice. Interactive REPL and
//! one-shot can still fill a missing provider from merged config after this
//! step. That path is not what `--server` boots.

use crate::durable_settings::DurableSettings;
use caliche::llm::ProviderType;

/// Selected provider and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerBootProvider {
    pub provider: String,
    pub source: &'static str,
}

/// Inputs for [`select_server_boot_provider`]. Empty strings are ignored.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServerBootProviderInput<'a> {
    pub explicit_provider: Option<&'a str>,
    pub explicit_model: bool,
    pub agent_provider: Option<&'a str>,
    pub default_provider_env: Option<&'a str>,
    pub last_provider: Option<&'a str>,
    pub last_model: Option<&'a str>,
}

/// Resolve the provider `--server` will use.
#[must_use]
pub fn select_server_boot_provider(input: ServerBootProviderInput<'_>) -> ServerBootProvider {
    if let Some(provider) = nonempty(input.explicit_provider) {
        return ServerBootProvider {
            provider: canonical_provider(provider),
            source: "flag",
        };
    }
    if !input.explicit_model
        && let Some(provider) = nonempty(input.last_provider)
        && nonempty(input.last_model).is_some()
        && provider.parse::<ProviderType>().is_ok()
    {
        return ServerBootProvider {
            provider: canonical_provider(provider),
            source: "last",
        };
    }
    if let Some(provider) =
        nonempty(input.agent_provider).or_else(|| nonempty(input.default_provider_env))
    {
        return ServerBootProvider {
            provider: canonical_provider(provider),
            source: "env",
        };
    }
    ServerBootProvider {
        provider: ProviderType::OpenRouter.to_string(),
        source: "default",
    }
}

/// Saved provider, model, and sampling loaded for boot.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PersistedBoot {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Prefer `settings.json`. Fill a missing pair from SQLite `last_*`.
#[must_use]
pub fn merge_persisted_boot(
    file: Option<&DurableSettings>,
    sqlite_provider: Option<&str>,
    sqlite_model: Option<&str>,
) -> PersistedBoot {
    let file_provider = file.and_then(|settings| nonempty(Some(settings.provider.as_str())));
    let file_model = file.and_then(|settings| nonempty(Some(settings.model.as_str())));
    let provider = file_provider
        .or_else(|| nonempty(sqlite_provider))
        .map(str::to_owned);
    let model = if file_provider.is_some() && file_model.is_some() {
        file_model.map(str::to_owned)
    } else {
        file_model
            .or_else(|| nonempty(sqlite_model))
            .map(str::to_owned)
    };
    PersistedBoot {
        provider,
        model,
        temperature: file.and_then(|settings| settings.temperature),
        max_tokens: file.and_then(|settings| settings.max_tokens),
    }
}

/// Read `settings.json` and merge it with an already loaded SQLite pair.
pub async fn load_persisted_boot(
    sqlite_provider: Option<String>,
    sqlite_model: Option<String>,
) -> PersistedBoot {
    let file = crate::durable_settings::load(&crate::durable_settings::default_path()).await;
    merge_persisted_boot(
        file.as_ref(),
        sqlite_provider.as_deref(),
        sqlite_model.as_deref(),
    )
}

/// Apply a boot choice onto the provider/model pair `prepare` passes to service construction.
///
/// Returns true when a saved model was applied. `flag` keeps an explicit
/// `--provider` and still applies the saved model when that record names the
/// same provider and `--model` did not already set one. `resolve_provider`
/// still turns an empty provider into `openrouter`.
pub fn apply_server_boot_provider(
    choice: &ServerBootProvider,
    saved_provider: Option<&str>,
    saved_model: Option<&str>,
    provider: &mut Option<ProviderType>,
    model: &mut Option<String>,
) -> anyhow::Result<bool> {
    match choice.source {
        "last" => {
            *provider = Some(parse_provider(&choice.provider)?);
            if let Some(saved_model) = nonempty(saved_model) {
                *model = Some(saved_model.to_owned());
            }
            Ok(true)
        }
        "env" => {
            *provider = Some(parse_provider(&choice.provider)?);
            Ok(false)
        }
        "flag" => {
            if model_is_empty(model)
                && let Some(saved_model) = nonempty(saved_model)
                && providers_match(&choice.provider, saved_provider)
            {
                *model = Some(saved_model.to_owned());
                return Ok(true);
            }
            Ok(false)
        }
        "default" => Ok(false),
        other => anyhow::bail!("unknown server boot provider source '{other}'"),
    }
}

fn model_is_empty(model: &Option<String>) -> bool {
    model.as_ref().is_none_or(|value| value.trim().is_empty())
}

fn providers_match(flag: &str, saved: Option<&str>) -> bool {
    let Some(saved) = nonempty(saved) else {
        return false;
    };
    match (flag.parse::<ProviderType>(), saved.parse::<ProviderType>()) {
        (Ok(flag_provider), Ok(saved_provider)) => flag_provider == saved_provider,
        _ => flag.eq_ignore_ascii_case(saved),
    }
}

fn parse_provider(provider: &str) -> anyhow::Result<ProviderType> {
    provider
        .parse()
        .map_err(|err| anyhow::anyhow!("Invalid provider in boot selection: {err}"))
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

/// Display form of a provider name when it parses. Invalid text stays as written
/// so the caller can reject it. Doctor and boot then print the same token.
fn canonical_provider(raw: &str) -> String {
    raw.parse::<ProviderType>()
        .map_or_else(|_| raw.to_owned(), |provider| provider.to_string())
}
