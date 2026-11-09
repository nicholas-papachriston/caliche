//! Slash-command argument completion tables.

use super::CompletionData;

pub(super) const STATIC_ARGS: &[(&str, &[&str])] = &[
    ("/help", &["quick", "all"]),
    ("/?", &["quick", "all"]),
    (
        "/creds",
        &["status", "setup", "set", "import", "list", "link"],
    ),
    (
        "/memory",
        &[
            "buffer", "summary", "entity", "vector", "enhanced", "show", "config",
        ],
    ),
    (
        "/learning",
        &[
            "status",
            "on",
            "off",
            "interval",
            "min-tool-calls",
            "max-skills",
            "memory-file",
            "auto-create",
            "auto-patch",
            "prefer-weak",
            "holdout",
            "min-holdout",
        ],
    ),
    ("/learning-loop", &["status", "on", "off"]),
    ("/autocompact", &["on", "off"]),
    ("/embeddings", &[]), // no args — just trigger completion early
    ("/artifact-dir", &["clear"]),
];

/// `/embedding` completion: the embedding providers this build has, then
/// `clear`.
pub(super) fn embedding_provider_choices() -> Vec<&'static str> {
    crate::credentials::offered_embedding_providers()
        .iter()
        .map(|(name, _)| *name)
        .chain(["clear"])
        .collect()
}

pub(super) const IMAGE_PROVIDER_CHOICES: &[&str] = &[
    "openai",
    "google",
    "openrouter",
    "cloudflare-ai-gateway",
    "clear",
];

const LOCAL_MODEL_ACTIONS: &[&str] = &["download", "list", "fit", "backends"];
const LOCAL_MODEL_DOWNLOAD_OPTIONS: &[&str] = &[
    "--backend",
    "--dir",
    "--model-dir",
    "--revision",
    "--hf-cache-dir",
];

const GATEWAY_TOP_LEVEL_ARGS: &[&str] = &[
    "setup", "start", "stop", "restart", "status", "list", "forward", "route", "llm", "twilio",
    "logs", "doctor",
];
const GATEWAY_FORWARD_ARGS: &[&str] = &["start", "stop", "restart", "status", "list"];
const GATEWAY_ROUTE_ARGS: &[&str] = &["list", "pin", "unpin"];
const GATEWAY_LLM_ARGS: &[&str] = &["status", "set", "clear"];
const GATEWAY_TWILIO_ARGS: &[&str] = &["webhook", "sync", "status"];

fn make_pair(name: &str, desc: Option<&str>, partial: &str) -> Option<Pair> {
    if !name.starts_with(partial) {
        return None;
    }
    Some(Pair {
        display: match desc {
            Some(d) => format!("{} - {}", name, d),
            None => name.to_string(),
        },
        replacement: format!("{} ", name),
    })
}

fn choice_pairs(partial: &str, choices: &[&str]) -> Vec<Pair> {
    choices
        .iter()
        .filter_map(|choice| make_pair(choice, None, partial))
        .collect()
}

pub(super) fn provider_choice_pairs(partial: &str, choices: &[&str]) -> Vec<Pair> {
    let partial_lower = partial.to_lowercase();
    choices
        .iter()
        .filter(|choice| partial_lower.is_empty() || choice.contains(partial_lower.as_str()))
        .map(|choice| Pair {
            display: (*choice).to_string(),
            replacement: format!("{} ", choice),
        })
        .collect()
}

fn provider_ids() -> Vec<&'static str> {
    crate::credentials::offered_providers()
        .iter()
        .map(|(name, _, _)| *name)
        .collect()
}

pub(super) fn embedding_models_for_provider(provider: &str) -> Vec<String> {
    match provider.parse::<caliche::llm::ProviderType>() {
        Ok(caliche::llm::ProviderType::OpenAI) => vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ],
        Ok(caliche::llm::ProviderType::OpenRouter) => vec![
            "openai/text-embedding-3-small".to_string(),
            "openai/text-embedding-3-large".to_string(),
        ],
        Ok(caliche::llm::ProviderType::CloudflareAIGateway) => vec![
            "workers-ai/@cf/baai/bge-base-en-v1.5".to_string(),
            "workers-ai/@cf/baai/bge-m3".to_string(),
            "workers-ai/@cf/baai/bge-large-en-v1.5".to_string(),
            "openai/text-embedding-3-small".to_string(),
            "openai/text-embedding-3-large".to_string(),
            "google-ai-studio/gemini-embedding-001".to_string(),
        ],
        Ok(caliche::llm::ProviderType::Bedrock) => {
            vec!["amazon.titan-embed-text-v2:0".to_string()]
        }
        Ok(caliche::llm::ProviderType::LMStudio) => vec!["default".to_string()],
        _ => Vec::new(),
    }
}

pub(super) fn image_models_for_provider(provider: &str) -> Vec<String> {
    provider
        .parse::<caliche::llm::ProviderType>()
        .map(caliche::llm::fallback_image_generation_models)
        .unwrap_or_default()
}

fn model_choice_pairs(
    token_partial: &str,
    models: impl IntoIterator<Item = String>,
    replacement_prefix: &str,
) -> Vec<Pair> {
    let partial_lower = token_partial.to_lowercase();
    models
        .into_iter()
        .filter(|model| partial_lower.is_empty() || model.to_lowercase().contains(&partial_lower))
        .map(|model| Pair {
            display: model.clone(),
            replacement: format!("{}{} ", replacement_prefix, model),
        })
        .collect()
}

/// Generic completion candidate used by the REPL input helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    pub display: String,
    pub replacement: String,
}

fn last_token_is(prior: &[&str], token: &str) -> bool {
    prior.last().is_some_and(|item| *item == token)
}

fn exact_choice_pairs(
    prior: &[&str],
    token_partial: &str,
    table: &[(&[&str], &[&str])],
) -> Vec<Pair> {
    for (keys, choices) in table {
        if *keys == prior {
            return choice_pairs(token_partial, choices);
        }
    }
    Vec::new()
}

fn prefix_choice_pairs(
    prior: &[&str],
    token_partial: &str,
    table: &[(&[&str], &[&str])],
) -> Vec<Pair> {
    for (prefix, choices) in table {
        if prior.starts_with(prefix) {
            return choice_pairs(token_partial, choices);
        }
    }
    Vec::new()
}

pub(super) fn complete_provider_slash_model(
    prior_tokens: &[&str],
    token_partial: &str,
    token_start: usize,
    providers: &[&str],
    models_for: fn(&str) -> Vec<String>,
) -> (usize, Vec<Pair>) {
    if prior_tokens.is_empty() {
        if let Some((provider, model_partial)) = token_partial.split_once('/') {
            let pairs =
                model_choice_pairs(model_partial, models_for(provider), &format!("{provider}/"));
            return (token_start, pairs);
        }
        return (token_start, provider_choice_pairs(token_partial, providers));
    }
    if prior_tokens.len() == 1 {
        let provider = prior_tokens[0];
        let pairs = model_choice_pairs(token_partial, models_for(provider), "");
        return (token_start, pairs);
    }
    (token_start, Vec::new())
}

pub(super) fn complete_provider_cmd_arg(
    data: Option<&CompletionData>,
    token_partial: &str,
) -> Vec<Pair> {
    let partial_lower = token_partial.to_lowercase();
    let mut pairs: Vec<Pair> = Vec::new();
    if let Some(d) = data {
        for p in &d.configured_providers {
            if partial_lower.is_empty() || p.contains(&partial_lower) {
                pairs.push(Pair {
                    display: format!("{} [configured]", p),
                    replacement: format!("{} ", p),
                });
            }
        }
    }
    let configured_set: std::collections::HashSet<String> = data
        .map(|d| d.configured_providers.iter().cloned().collect())
        .unwrap_or_default();
    for (name, _, desc) in crate::credentials::offered_providers() {
        if (partial_lower.is_empty() || name.contains(partial_lower.as_str()))
            && !configured_set.contains(*name)
        {
            pairs.push(Pair {
                display: format!("{} - {}", name, desc),
                replacement: format!("{} ", name),
            });
        }
    }
    pairs
}

pub(super) fn complete_model_cmd_arg(
    data: Option<&CompletionData>,
    token_partial: &str,
) -> Vec<Pair> {
    let Some(data) = data else {
        return Vec::new();
    };
    let partial_lower = token_partial.to_lowercase();
    data.current_models
        .iter()
        .filter(|m| partial_lower.is_empty() || m.to_lowercase().contains(&partial_lower))
        .map(|m| Pair {
            display: m.clone(),
            replacement: format!("{} ", m),
        })
        .collect()
}

/// `--backend` completion: the backends this build can run, from core.
#[cfg(feature = "local-inference-base")]
fn local_inference_backends() -> &'static [&'static str] {
    caliche::llm::LocalInferenceProvider::backend_names()
}

#[cfg(not(feature = "local-inference-base"))]
const fn local_inference_backends() -> &'static [&'static str] {
    &[]
}

pub(super) fn complete_local_model_arg(prior_tokens: &[&str], token_partial: &str) -> Vec<Pair> {
    match prior_tokens {
        [] => choice_pairs(token_partial, LOCAL_MODEL_ACTIONS),
        ["download"] => choice_pairs(token_partial, LOCAL_MODEL_DOWNLOAD_OPTIONS),
        ["download", "--backend"] => choice_pairs(token_partial, local_inference_backends()),
        _ if last_token_is(prior_tokens, "--backend") => {
            choice_pairs(token_partial, local_inference_backends())
        }
        _ => Vec::new(),
    }
}

const GATEWAY_EXACT_CHOICES: &[(&[&str], &[&str])] = &[
    (&[], GATEWAY_TOP_LEVEL_ARGS),
    (&["setup"], &["--forward", "--provider", "--port", "--dir"]),
    (&["start"], &["--port", "--dir"]),
    (&["restart"], &["--port", "--dir"]),
    (&["status"], &["--dir"]),
    (&["stop"], &["--dir"]),
    (&["list"], &["--dir"]),
    (&["forward"], GATEWAY_FORWARD_ARGS),
    (&["forward", "start"], &["--provider", "--port", "--dir"]),
    (&["forward", "restart"], &["--provider", "--port", "--dir"]),
    (&["forward", "status"], &["--dir"]),
    (&["forward", "stop"], &["--dir"]),
    (&["forward", "list"], &["--dir"]),
    (&["route"], GATEWAY_ROUTE_ARGS),
    (&["route", "pin"], &["--agent", "--dir"]),
    (&["route", "unpin"], &["--dir"]),
    (&["route", "list"], &["--dir"]),
    (&["llm"], GATEWAY_LLM_ARGS),
    (&["llm", "set"], &["--provider", "--model", "--dir"]),
    (&["llm", "status"], &["--dir"]),
    (&["llm", "clear"], &["--dir"]),
    (&["twilio"], GATEWAY_TWILIO_ARGS),
    (&["twilio", "webhook"], &["sync", "status"]),
    (&["twilio", "sync"], &["--url", "--dir"]),
    (&["twilio", "status"], &["--url", "--dir"]),
    (&["twilio", "webhook", "sync"], &["--url", "--dir"]),
    (&["twilio", "webhook", "status"], &["--url", "--dir"]),
    (
        &["logs"],
        &["--lines", "--forward", "--no-forward", "--dir"],
    ),
    (&["doctor"], &["--all", "--dir"]),
];

const GATEWAY_PREFIX_CHOICES: &[(&[&str], &[&str])] = &[
    (&["forward", "start"], &["--provider", "--port", "--dir"]),
    (&["forward", "restart"], &["--provider", "--port", "--dir"]),
    (&["forward", "status"], &["--dir"]),
    (&["forward", "stop"], &["--dir"]),
    (&["forward", "list"], &["--dir"]),
    (&["route", "pin"], &["--agent", "--dir"]),
    (&["route", "unpin"], &["--dir"]),
    (&["route", "list"], &["--dir"]),
    (&["llm", "set"], &["--provider", "--model", "--dir"]),
    (&["llm", "status"], &["--dir"]),
    (&["llm", "clear"], &["--dir"]),
    (&["twilio", "sync"], &["--url", "--dir"]),
    (&["twilio", "status"], &["--url", "--dir"]),
    (&["twilio", "webhook", "sync"], &["--url", "--dir"]),
    (&["twilio", "webhook", "status"], &["--url", "--dir"]),
    (
        &["logs"],
        &["--lines", "--forward", "--no-forward", "--dir"],
    ),
    (&["doctor"], &["--all", "--dir"]),
];

pub(super) fn complete_gateway_pairs(
    prior_tokens: &[&str],
    token_partial: &str,
    data: Option<&CompletionData>,
) -> Vec<Pair> {
    if last_token_is(prior_tokens, "--provider") {
        return provider_choice_pairs(token_partial, &provider_ids());
    }
    if prior_tokens.starts_with(&["llm", "set"]) && last_token_is(prior_tokens, "--model") {
        return data
            .map(|d| model_choice_pairs(token_partial, d.current_models.clone(), ""))
            .unwrap_or_default();
    }
    let mut pairs = exact_choice_pairs(prior_tokens, token_partial, GATEWAY_EXACT_CHOICES);
    if pairs.is_empty() {
        pairs = prefix_choice_pairs(prior_tokens, token_partial, GATEWAY_PREFIX_CHOICES);
    }
    pairs
}

const CREDS_EXACT_CHOICES: &[(&[&str], &[&str])] = &[
    (
        &[],
        &[
            "status",
            "setup",
            "set",
            "import",
            "list",
            "link",
            "gateway",
            "providers",
        ],
    ),
    (&["setup"], &["providers", "gateway"]),
    (&["providers"], &["setup"]),
    (&["gateway"], GATEWAY_TOP_LEVEL_ARGS),
    (&["gateway", "forward"], GATEWAY_FORWARD_ARGS),
    (&["gateway", "route"], GATEWAY_ROUTE_ARGS),
    (&["gateway", "llm"], GATEWAY_LLM_ARGS),
    (&["gateway", "twilio"], GATEWAY_TWILIO_ARGS),
    (
        &["gateway", "setup"],
        &["--forward", "--provider", "--port", "--dir"],
    ),
    (&["gateway", "start"], &["--port", "--dir"]),
    (&["gateway", "restart"], &["--port", "--dir"]),
    (&["gateway", "status"], &["--dir"]),
    (&["gateway", "stop"], &["--dir"]),
    (&["gateway", "list"], &["--dir"]),
    (
        &["gateway", "forward", "start"],
        &["--provider", "--port", "--dir"],
    ),
    (
        &["gateway", "forward", "restart"],
        &["--provider", "--port", "--dir"],
    ),
    (&["gateway", "forward", "status"], &["--dir"]),
    (&["gateway", "forward", "stop"], &["--dir"]),
    (&["gateway", "forward", "list"], &["--dir"]),
    (&["gateway", "route", "pin"], &["--agent", "--dir"]),
    (&["gateway", "route", "unpin"], &["--dir"]),
    (&["gateway", "route", "list"], &["--dir"]),
    (
        &["gateway", "llm", "set"],
        &["--provider", "--model", "--dir"],
    ),
    (&["gateway", "llm", "status"], &["--dir"]),
    (&["gateway", "llm", "clear"], &["--dir"]),
    (&["gateway", "twilio", "webhook"], &["sync", "status"]),
    (&["gateway", "twilio", "sync"], &["--url", "--dir"]),
    (&["gateway", "twilio", "status"], &["--url", "--dir"]),
    (
        &["gateway", "twilio", "webhook", "sync"],
        &["--url", "--dir"],
    ),
    (
        &["gateway", "twilio", "webhook", "status"],
        &["--url", "--dir"],
    ),
];

const CREDS_PREFIX_CHOICES: &[(&[&str], &[&str])] = &[
    (
        &["gateway", "forward", "start"],
        &["--provider", "--port", "--dir"],
    ),
    (
        &["gateway", "forward", "restart"],
        &["--provider", "--port", "--dir"],
    ),
    (&["gateway", "forward", "status"], &["--dir"]),
    (&["gateway", "forward", "stop"], &["--dir"]),
    (&["gateway", "forward", "list"], &["--dir"]),
    (&["gateway", "route", "pin"], &["--agent", "--dir"]),
    (&["gateway", "route", "unpin"], &["--dir"]),
    (&["gateway", "route", "list"], &["--dir"]),
    (
        &["gateway", "llm", "set"],
        &["--provider", "--model", "--dir"],
    ),
    (&["gateway", "llm", "status"], &["--dir"]),
    (&["gateway", "llm", "clear"], &["--dir"]),
    (&["gateway", "twilio", "sync"], &["--url", "--dir"]),
    (&["gateway", "twilio", "status"], &["--url", "--dir"]),
    (
        &["gateway", "twilio", "webhook", "sync"],
        &["--url", "--dir"],
    ),
    (
        &["gateway", "twilio", "webhook", "status"],
        &["--url", "--dir"],
    ),
];

fn complete_creds_set_pairs(token_partial: &str) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for (name, _, _) in crate::credentials::known_providers() {
        if let Some(pair) = make_pair(name, None, token_partial) {
            pairs.push(pair);
        }
    }
    pairs
}

pub(super) fn complete_creds_pairs(
    prior_tokens: &[&str],
    token_partial: &str,
    data: Option<&CompletionData>,
) -> Vec<Pair> {
    if last_token_is(prior_tokens, "--provider") {
        return provider_choice_pairs(token_partial, &provider_ids());
    }
    if prior_tokens.starts_with(&["gateway", "llm", "set"])
        && last_token_is(prior_tokens, "--model")
    {
        return data
            .map(|d| model_choice_pairs(token_partial, d.current_models.clone(), ""))
            .unwrap_or_default();
    }
    if prior_tokens == ["set"] {
        return complete_creds_set_pairs(token_partial);
    }
    let mut pairs = exact_choice_pairs(prior_tokens, token_partial, CREDS_EXACT_CHOICES);
    if pairs.is_empty() {
        pairs = prefix_choice_pairs(prior_tokens, token_partial, CREDS_PREFIX_CHOICES);
    }
    pairs
}

pub(super) fn complete_static_arg(
    cmd: &str,
    prior_tokens: &[&str],
    token_partial: &str,
) -> Option<Vec<Pair>> {
    if !prior_tokens.is_empty() {
        return None;
    }
    for (static_cmd, choices) in STATIC_ARGS {
        if *static_cmd == cmd {
            return Some(
                choices
                    .iter()
                    .filter_map(|c| make_pair(c, None, token_partial))
                    .collect(),
            );
        }
    }
    None
}
