//! Canonical model pricing table and lookup helpers.

/// Pricing per 1M tokens (input, output).
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Cost per 1M input tokens in USD.
    pub input_per_million: f64,
    /// Cost per 1M output tokens in USD.
    pub output_per_million: f64,
}

impl ModelPricing {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
        }
    }

    /// Calculate cost for given token counts.
    pub fn calculate(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1_000_000.0) * self.input_per_million;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_per_million;
        input_cost + output_cost
    }
}

/// Get pricing for a provider/model pair (returns `None` if unknown).
pub fn get_pricing(provider: &str, model: &str) -> Option<ModelPricing> {
    let provider = provider.to_lowercase();
    let model = model.to_lowercase();

    match provider.as_str() {
        "openai" => get_openai_pricing(&model),
        "anthropic" => get_anthropic_pricing(&model),
        "google" => get_google_pricing(&model),
        "bedrock" => get_bedrock_pricing(&model),
        "openrouter" => get_openrouter_pricing(&model),
        "perplexity" => get_perplexity_pricing(&model),
        "cloudflare-ai-gateway" => get_cloudflare_aig_pricing(&model),
        _ => None,
    }
}

/// Cloudflare AI Gateway proxies upstream providers using the
/// `{upstream_provider}/{model}` model identifier. Strip the prefix and reuse
/// the upstream provider's pricing table when known.
fn get_cloudflare_aig_pricing(model: &str) -> Option<ModelPricing> {
    let (upstream, upstream_model) = model.split_once('/')?;
    match upstream {
        "openai" => get_openai_pricing(upstream_model),
        "anthropic" => get_anthropic_pricing(upstream_model),
        "google" => get_google_pricing(upstream_model),
        "openrouter" => get_openrouter_pricing(upstream_model),
        "perplexity" => get_perplexity_pricing(upstream_model),
        // Cloudflare-hosted Workers AI, Grok, dynamic, and others either
        // bill via Cloudflare credits or have rapidly changing pricing;
        // skip until we have authoritative rates to publish.
        _ => None,
    }
}

fn get_openai_pricing(model: &str) -> Option<ModelPricing> {
    if model.contains("gpt-4o-mini") {
        Some(ModelPricing::new(0.15, 0.60))
    } else if model.contains("gpt-4o") {
        Some(ModelPricing::new(2.50, 10.00))
    } else if model.contains("gpt-4-turbo") {
        Some(ModelPricing::new(10.00, 30.00))
    } else if model.contains("gpt-4") {
        Some(ModelPricing::new(30.00, 60.00))
    } else if model.contains("gpt-3.5-turbo") {
        Some(ModelPricing::new(0.50, 1.50))
    } else if model.contains("o1-preview") {
        Some(ModelPricing::new(15.00, 60.00))
    } else if model.contains("o1-mini") {
        Some(ModelPricing::new(3.00, 12.00))
    } else if model.contains("o3-mini") {
        Some(ModelPricing::new(1.10, 4.40))
    } else {
        None
    }
}

fn get_anthropic_pricing(model: &str) -> Option<ModelPricing> {
    if model.contains("claude-3-5-sonnet") || model.contains("claude-3.5-sonnet") {
        Some(ModelPricing::new(3.00, 15.00))
    } else if model.contains("claude-3-5-haiku") || model.contains("claude-3.5-haiku") {
        Some(ModelPricing::new(0.80, 4.00))
    } else if model.contains("claude-3-opus") {
        Some(ModelPricing::new(15.00, 75.00))
    } else if model.contains("claude-3-sonnet") {
        Some(ModelPricing::new(3.00, 15.00))
    } else if model.contains("claude-3-haiku") {
        Some(ModelPricing::new(0.25, 1.25))
    } else {
        None
    }
}

fn get_google_pricing(model: &str) -> Option<ModelPricing> {
    if model.contains("gemini-2.0-flash") {
        Some(ModelPricing::new(0.10, 0.40))
    } else if model.contains("gemini-1.5-flash") {
        Some(ModelPricing::new(0.075, 0.30))
    } else if model.contains("gemini-1.5-pro") {
        Some(ModelPricing::new(1.25, 5.00))
    } else if model.contains("gemini-pro") {
        Some(ModelPricing::new(0.50, 1.50))
    } else {
        None
    }
}

fn get_bedrock_pricing(model: &str) -> Option<ModelPricing> {
    if model.contains("claude-3-5-sonnet") {
        Some(ModelPricing::new(3.00, 15.00))
    } else if model.contains("claude-3-haiku") {
        Some(ModelPricing::new(0.25, 1.25))
    } else if model.contains("claude-3-opus") {
        Some(ModelPricing::new(15.00, 75.00))
    } else if model.contains("titan") {
        Some(ModelPricing::new(0.80, 1.60))
    } else if model.contains("llama3") {
        Some(ModelPricing::new(0.35, 0.40))
    } else {
        None
    }
}

fn get_openrouter_pricing(model: &str) -> Option<ModelPricing> {
    if model.contains("claude-3-5-sonnet") {
        Some(ModelPricing::new(3.00, 15.00))
    } else if model.contains("gpt-4o") {
        Some(ModelPricing::new(2.50, 10.00))
    } else if model.contains("llama-3") && model.contains("70b") {
        Some(ModelPricing::new(0.59, 0.79))
    } else if model.contains("llama-3") && model.contains("8b") {
        Some(ModelPricing::new(0.055, 0.055))
    } else if model.contains("mixtral") {
        Some(ModelPricing::new(0.24, 0.24))
    } else {
        None
    }
}

fn get_perplexity_pricing(model: &str) -> Option<ModelPricing> {
    if model.contains("sonar-pro") {
        Some(ModelPricing::new(3.00, 15.00))
    } else if model.contains("sonar") {
        Some(ModelPricing::new(1.00, 1.00))
    } else {
        None
    }
}
