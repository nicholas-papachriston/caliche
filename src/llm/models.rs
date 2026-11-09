use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    Bedrock,
    OpenRouter,
    Perplexity,
    OpenAI,
    LMStudio,
    Anthropic,
    Google,
    Cohere,
    AzureOpenAI,
    Ollama,
    CloudflareAIGateway,
    #[cfg(feature = "local-inference-base")]
    LocalInference,
    External,
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bedrock => write!(f, "bedrock"),
            Self::OpenRouter => write!(f, "openrouter"),
            Self::Perplexity => write!(f, "perplexity"),
            Self::OpenAI => write!(f, "openai"),
            Self::LMStudio => write!(f, "lmstudio"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Google => write!(f, "google"),
            Self::Cohere => write!(f, "cohere"),
            Self::AzureOpenAI => write!(f, "azure-openai"),
            Self::Ollama => write!(f, "ollama"),
            Self::CloudflareAIGateway => write!(f, "cloudflare-ai-gateway"),
            #[cfg(feature = "local-inference-base")]
            Self::LocalInference => write!(f, "local-inference"),
            Self::External => write!(f, "external"),
        }
    }
}

impl ProviderType {
    /// The feature that adds `name` back, when caliche knows the provider
    /// but this build left it out. `None` for a provider this build has and
    /// for a name caliche does not know.
    #[must_use]
    pub fn feature_left_out_of_build(name: &str) -> Option<&'static str> {
        let local_inference = matches!(
            name.to_lowercase().as_str(),
            "local-inference" | "localinference"
        );
        if local_inference {
            return cfg!(not(feature = "local-inference-base")).then_some("local-inference");
        }
        Self::known(name)
            .filter(|provider| !crate::capabilities::is_provider_available(*provider))
            .map(Self::build_feature)
    }

    /// Parse a provider the user selects to run: like [`str::parse`], but a
    /// provider this build left out is an error that names the feature.
    ///
    /// # Errors
    ///
    /// Returns the message for an unknown name or a provider not in this
    /// build.
    pub fn parse_in_build(name: &str) -> Result<Self, String> {
        match Self::feature_left_out_of_build(name) {
            Some(feature) => Err(not_in_build_message(name, feature)),
            None => name.parse(),
        }
    }

    /// The Cargo feature that builds this provider in.
    #[must_use]
    pub const fn build_feature(self) -> &'static str {
        match self {
            Self::Bedrock => "bedrock",
            Self::OpenRouter => "openrouter",
            Self::Perplexity => "perplexity",
            Self::OpenAI => "openai",
            Self::LMStudio => "lmstudio",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::Cohere => "cohere",
            Self::AzureOpenAI => "azure-openai",
            Self::Ollama => "ollama",
            Self::CloudflareAIGateway => "cloudflare-ai-gateway",
            #[cfg(feature = "local-inference-base")]
            Self::LocalInference => "local-inference",
            Self::External => "external",
        }
    }

    fn known(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "bedrock" => Some(Self::Bedrock),
            "openrouter" => Some(Self::OpenRouter),
            "perplexity" => Some(Self::Perplexity),
            "openai" => Some(Self::OpenAI),
            "lmstudio" => Some(Self::LMStudio),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            "cohere" => Some(Self::Cohere),
            "azure-openai" | "azure_openai" | "azureopenai" => Some(Self::AzureOpenAI),
            "ollama" => Some(Self::Ollama),
            "cloudflare-ai-gateway"
            | "cloudflare_ai_gateway"
            | "cloudflareaigateway"
            | "cf-aig"
            | "cloudflare" => Some(Self::CloudflareAIGateway),
            #[cfg(feature = "local-inference-base")]
            "local-inference" | "localinference" => Some(Self::LocalInference),
            // Alias used by credentials / JS bindings for Google Gemini.
            "gemini" => Some(Self::Google),
            "external" => Some(Self::External),
            _ => None,
        }
    }
}

/// The one-line message for a provider this build left out.
#[must_use]
pub fn not_in_build_message(name: &str, feature: &str) -> String {
    format!("{name} is not in this build. Rebuild with --features {feature}.")
}

impl std::str::FromStr for ProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(provider) = Self::known(s) {
            return Ok(provider);
        }
        if let Some(feature) = Self::feature_left_out_of_build(s) {
            return Err(not_in_build_message(s, feature));
        }
        let supported = [
            "bedrock",
            "openrouter",
            "perplexity",
            "openai",
            "lmstudio",
            "anthropic",
            "google",
            "gemini",
            "cohere",
            "azure-openai",
            "ollama",
            "cloudflare-ai-gateway",
            "external",
            #[cfg(feature = "local-inference-base")]
            "local-inference",
        ];
        Err(format!(
            "Unknown provider: {}. Supported providers: {}",
            s,
            supported.join(", ")
        ))
    }
}

/// Fallback model IDs used when live catalog fetch is unavailable.
pub mod defaults {
    /// Current Anthropic coding default (native API id).
    pub const ANTHROPIC: &str = "claude-sonnet-4-6";
    /// Current OpenAI coding default.
    pub const OPENAI: &str = "gpt-5.4";
    /// Current Google Gemini text-chat default (unprefixed).
    pub const GOOGLE: &str = "gemini-2.5-flash";
    /// Google Generative Language API model resource name.
    pub const GOOGLE_API: &str = "models/gemini-2.5-flash";
    /// Current OpenRouter coding default.
    pub const OPENROUTER: &str = "anthropic/claude-sonnet-4.6";
    /// Current Perplexity default.
    pub const PERPLEXITY: &str = "sonar-pro";
}

/// Available Bedrock models
/// Note: These are fallback models - prefer fetching from API
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BedrockModel {
    #[serde(rename = "anthropic.claude-sonnet-4-20250514-v1:0")]
    ClaudeSonnet4,
    #[serde(rename = "anthropic.claude-opus-4-20250514-v1:0")]
    ClaudeOpus4,
    #[serde(rename = "anthropic.claude-3-5-sonnet-20241022-v2:0")]
    Claude35SonnetV2,
    #[serde(rename = "anthropic.claude-3-5-haiku-20241022-v1:0")]
    Claude35Haiku,
    #[serde(rename = "meta.llama3-3-70b-instruct-v1:0")]
    Llama3370b,
    #[serde(rename = "amazon.nova-pro-v1:0")]
    NovaPro,
    #[serde(rename = "amazon.nova-lite-v1:0")]
    NovaLite,
}

impl BedrockModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ClaudeSonnet4 => "anthropic.claude-sonnet-4-20250514-v1:0",
            Self::ClaudeOpus4 => "anthropic.claude-opus-4-20250514-v1:0",
            Self::Claude35SonnetV2 => "anthropic.claude-3-5-sonnet-20241022-v2:0",
            Self::Claude35Haiku => "anthropic.claude-3-5-haiku-20241022-v1:0",
            Self::Llama3370b => "meta.llama3-3-70b-instruct-v1:0",
            Self::NovaPro => "amazon.nova-pro-v1:0",
            Self::NovaLite => "amazon.nova-lite-v1:0",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::ClaudeSonnet4,
            Self::ClaudeOpus4,
            Self::Claude35SonnetV2,
            Self::Claude35Haiku,
            Self::Llama3370b,
            Self::NovaPro,
            Self::NovaLite,
        ]
    }
}

impl fmt::Display for BedrockModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for BedrockModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "anthropic.claude-sonnet-4-20250514-v1:0" => Ok(Self::ClaudeSonnet4),
            "anthropic.claude-opus-4-20250514-v1:0" => Ok(Self::ClaudeOpus4),
            "anthropic.claude-3-5-sonnet-20241022-v2:0" => Ok(Self::Claude35SonnetV2),
            "anthropic.claude-3-5-haiku-20241022-v1:0" => Ok(Self::Claude35Haiku),
            "meta.llama3-3-70b-instruct-v1:0" => Ok(Self::Llama3370b),
            "amazon.nova-pro-v1:0" => Ok(Self::NovaPro),
            "amazon.nova-lite-v1:0" => Ok(Self::NovaLite),
            _ => Err(format!(
                "Unknown Bedrock model: {s}. Any valid model ID can be used."
            )),
        }
    }
}

/// Available `OpenRouter` models
/// Note: `OpenRouter` has hundreds of models - prefer fetching from API
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterModel {
    #[serde(rename = "anthropic/claude-sonnet-5")]
    ClaudeSonnet5,
    #[serde(rename = "anthropic/claude-opus-5")]
    ClaudeOpus5,
    #[serde(rename = "anthropic/claude-sonnet-4.6")]
    ClaudeSonnet46,
    #[serde(rename = "anthropic/claude-sonnet-4")]
    ClaudeSonnet4,
    #[serde(rename = "anthropic/claude-opus-4")]
    ClaudeOpus4,
    #[serde(rename = "anthropic/claude-haiku-4.5")]
    ClaudeHaiku45,
    #[serde(rename = "openai/gpt-5.4")]
    Gpt54,
    #[serde(rename = "openai/gpt-5")]
    Gpt5,
    #[serde(rename = "openai/gpt-4.1")]
    Gpt41,
    #[serde(rename = "openai/gpt-4o")]
    Gpt4o,
    #[serde(rename = "openai/o3")]
    O3,
    #[serde(rename = "openai/o4-mini")]
    O4Mini,
    #[serde(rename = "x-ai/grok-4.6")]
    Grok46,
    #[serde(rename = "google/gemini-3.7-flash")]
    Gemini37Flash,
    #[serde(rename = "google/gemini-3.5-flash")]
    Gemini35Flash,
    #[serde(rename = "google/gemini-2.5-pro")]
    Gemini25Pro,
    #[serde(rename = "google/gemini-2.5-flash")]
    Gemini25Flash,
    #[serde(rename = "meta-llama/llama-3.3-70b-instruct")]
    Llama3370b,
    #[serde(rename = "deepseek/deepseek-r1")]
    DeepseekR1,
    #[serde(rename = "qwen/qwen-2.5-72b-instruct")]
    Qwen25_72b,
}

impl OpenRouterModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ClaudeSonnet5 => "anthropic/claude-sonnet-5",
            Self::ClaudeOpus5 => "anthropic/claude-opus-5",
            Self::ClaudeSonnet46 => "anthropic/claude-sonnet-4.6",
            Self::ClaudeSonnet4 => "anthropic/claude-sonnet-4",
            Self::ClaudeOpus4 => "anthropic/claude-opus-4",
            Self::ClaudeHaiku45 => "anthropic/claude-haiku-4.5",
            Self::Gpt54 => "openai/gpt-5.4",
            Self::Gpt5 => "openai/gpt-5",
            Self::Gpt41 => "openai/gpt-4.1",
            Self::Gpt4o => "openai/gpt-4o",
            Self::O3 => "openai/o3",
            Self::O4Mini => "openai/o4-mini",
            Self::Grok46 => "x-ai/grok-4.6",
            Self::Gemini37Flash => "google/gemini-3.7-flash",
            Self::Gemini35Flash => "google/gemini-3.5-flash",
            Self::Gemini25Pro => "google/gemini-2.5-pro",
            Self::Gemini25Flash => "google/gemini-2.5-flash",
            Self::Llama3370b => "meta-llama/llama-3.3-70b-instruct",
            Self::DeepseekR1 => "deepseek/deepseek-r1",
            Self::Qwen25_72b => "qwen/qwen-2.5-72b-instruct",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::ClaudeSonnet5,
            Self::ClaudeOpus5,
            Self::ClaudeSonnet46,
            Self::ClaudeSonnet4,
            Self::ClaudeOpus4,
            Self::ClaudeHaiku45,
            Self::Gpt54,
            Self::Gpt5,
            Self::Gpt41,
            Self::Gpt4o,
            Self::O3,
            Self::O4Mini,
            Self::Grok46,
            Self::Gemini37Flash,
            Self::Gemini35Flash,
            Self::Gemini25Pro,
            Self::Gemini25Flash,
            Self::Llama3370b,
            Self::DeepseekR1,
            Self::Qwen25_72b,
        ]
    }
}

impl fmt::Display for OpenRouterModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for OpenRouterModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(model) = Self::from_anthropic_or_openai(s) {
            return Ok(model);
        }
        Self::from_other_provider(s)
    }
}

impl OpenRouterModel {
    fn from_anthropic_or_openai(s: &str) -> Option<Self> {
        Some(match s {
            "anthropic/claude-sonnet-5" => Self::ClaudeSonnet5,
            "anthropic/claude-opus-5" => Self::ClaudeOpus5,
            "anthropic/claude-sonnet-4.6" => Self::ClaudeSonnet46,
            "anthropic/claude-sonnet-4" => Self::ClaudeSonnet4,
            "anthropic/claude-opus-4" => Self::ClaudeOpus4,
            "anthropic/claude-haiku-4.5" => Self::ClaudeHaiku45,
            "openai/gpt-5.4" => Self::Gpt54,
            "openai/gpt-5" => Self::Gpt5,
            "openai/gpt-4.1" => Self::Gpt41,
            "openai/gpt-4o" => Self::Gpt4o,
            "openai/o3" => Self::O3,
            "openai/o4-mini" => Self::O4Mini,
            _ => return None,
        })
    }

    fn from_other_provider(s: &str) -> Result<Self, String> {
        Ok(match s {
            "x-ai/grok-4.6" | "x-ai/grok-4.3" => Self::Grok46,
            "google/gemini-3.7-flash" => Self::Gemini37Flash,
            "google/gemini-3.5-flash" => Self::Gemini35Flash,
            "google/gemini-2.5-pro" => Self::Gemini25Pro,
            "google/gemini-2.5-flash" => Self::Gemini25Flash,
            "meta-llama/llama-3.3-70b-instruct" => Self::Llama3370b,
            "deepseek/deepseek-r1" => Self::DeepseekR1,
            "qwen/qwen-2.5-72b-instruct" => Self::Qwen25_72b,
            _ => {
                return Err(format!(
                    "Unknown OpenRouter model: {s}. Any valid model ID can be used."
                ));
            }
        })
    }
}

/// Curated list of well-known Cloudflare AI Gateway model strings.
///
/// Cloudflare AI Gateway accepts the `{upstream_provider}/{model}` format and
/// supports many providers (openai, anthropic, google-ai-studio, grok,
/// workers-ai, dynamic, etc.). This enum is only a fallback for discovery; any valid
/// upstream model identifier can be used at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudflareAIGatewayModel {
    #[serde(rename = "openai/gpt-5.4")]
    OpenAiGpt54,
    #[serde(rename = "openai/gpt-5")]
    OpenAiGpt5,
    #[serde(rename = "openai/gpt-4o")]
    OpenAiGpt4o,
    #[serde(rename = "openai/gpt-4o-mini")]
    OpenAiGpt4oMini,
    #[serde(rename = "anthropic/claude-sonnet-5")]
    AnthropicClaudeSonnet5,
    #[serde(rename = "anthropic/claude-haiku-4-5")]
    AnthropicClaudeHaiku45,
    #[serde(rename = "anthropic/claude-sonnet-4-5")]
    AnthropicClaudeSonnet45,
    #[serde(rename = "anthropic/claude-opus-4-5")]
    AnthropicClaudeOpus45,
    #[serde(rename = "google-ai-studio/gemini-3.5-flash")]
    GoogleGemini35Flash,
    #[serde(rename = "google-ai-studio/gemini-2.5-pro")]
    GoogleGemini25Pro,
    #[serde(rename = "google-ai-studio/gemini-2.5-flash")]
    GoogleGemini25Flash,
    #[serde(rename = "grok/grok-4.6")]
    GrokGrok46,
    #[serde(rename = "grok/grok-4")]
    GrokGrok4,
    #[serde(rename = "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast")]
    WorkersAiLlama3370b,
    #[serde(rename = "workers-ai/@cf/meta/llama-3.1-8b-instruct")]
    WorkersAiLlama318b,
}

impl CloudflareAIGatewayModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAiGpt54 => "openai/gpt-5.4",
            Self::OpenAiGpt5 => "openai/gpt-5",
            Self::OpenAiGpt4o => "openai/gpt-4o",
            Self::OpenAiGpt4oMini => "openai/gpt-4o-mini",
            Self::AnthropicClaudeSonnet5 => "anthropic/claude-sonnet-5",
            Self::AnthropicClaudeHaiku45 => "anthropic/claude-haiku-4-5",
            Self::AnthropicClaudeSonnet45 => "anthropic/claude-sonnet-4-5",
            Self::AnthropicClaudeOpus45 => "anthropic/claude-opus-4-5",
            Self::GoogleGemini35Flash => "google-ai-studio/gemini-3.5-flash",
            Self::GoogleGemini25Pro => "google-ai-studio/gemini-2.5-pro",
            Self::GoogleGemini25Flash => "google-ai-studio/gemini-2.5-flash",
            Self::GrokGrok46 => "grok/grok-4.6",
            Self::GrokGrok4 => "grok/grok-4",
            Self::WorkersAiLlama3370b => "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            Self::WorkersAiLlama318b => "workers-ai/@cf/meta/llama-3.1-8b-instruct",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::OpenAiGpt54,
            Self::OpenAiGpt5,
            Self::OpenAiGpt4oMini,
            Self::OpenAiGpt4o,
            Self::AnthropicClaudeSonnet5,
            Self::AnthropicClaudeHaiku45,
            Self::AnthropicClaudeSonnet45,
            Self::AnthropicClaudeOpus45,
            Self::GoogleGemini35Flash,
            Self::GoogleGemini25Pro,
            Self::GoogleGemini25Flash,
            Self::GrokGrok46,
            Self::GrokGrok4,
            Self::WorkersAiLlama3370b,
            Self::WorkersAiLlama318b,
        ]
    }
}

impl fmt::Display for CloudflareAIGatewayModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for CloudflareAIGatewayModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "openai/gpt-5.4" => Ok(Self::OpenAiGpt54),
            "openai/gpt-5" => Ok(Self::OpenAiGpt5),
            "openai/gpt-4o" => Ok(Self::OpenAiGpt4o),
            "openai/gpt-4o-mini" => Ok(Self::OpenAiGpt4oMini),
            "anthropic/claude-sonnet-5" => Ok(Self::AnthropicClaudeSonnet5),
            "anthropic/claude-haiku-4-5" => Ok(Self::AnthropicClaudeHaiku45),
            "anthropic/claude-sonnet-4-5" => Ok(Self::AnthropicClaudeSonnet45),
            "anthropic/claude-opus-4-5" => Ok(Self::AnthropicClaudeOpus45),
            "google-ai-studio/gemini-3.5-flash" => Ok(Self::GoogleGemini35Flash),
            "google-ai-studio/gemini-2.5-pro" => Ok(Self::GoogleGemini25Pro),
            "google-ai-studio/gemini-2.5-flash" => Ok(Self::GoogleGemini25Flash),
            "grok/grok-4.6" | "x-ai/grok-4.6" => Ok(Self::GrokGrok46),
            "grok/grok-4" => Ok(Self::GrokGrok4),
            "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast" => Ok(Self::WorkersAiLlama3370b),
            "workers-ai/@cf/meta/llama-3.1-8b-instruct" => Ok(Self::WorkersAiLlama318b),
            _ => Err(format!(
                "Unknown Cloudflare AI Gateway model: {s}. Any valid {{provider}}/{{model}} ID can be used."
            )),
        }
    }
}

/// Available Perplexity models
/// Note: These are fallback models - prefer fetching from API
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerplexityModel {
    #[serde(rename = "sonar-pro")]
    SonarPro,
    #[serde(rename = "sonar")]
    Sonar,
    #[serde(rename = "sonar-reasoning-pro")]
    SonarReasoningPro,
    #[serde(rename = "sonar-reasoning")]
    SonarReasoning,
    #[serde(rename = "sonar-deep-research")]
    SonarDeepResearch,
}

impl PerplexityModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SonarPro => "sonar-pro",
            Self::Sonar => "sonar",
            Self::SonarReasoningPro => "sonar-reasoning-pro",
            Self::SonarReasoning => "sonar-reasoning",
            Self::SonarDeepResearch => "sonar-deep-research",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::SonarPro,
            Self::Sonar,
            Self::SonarReasoningPro,
            Self::SonarReasoning,
            Self::SonarDeepResearch,
        ]
    }
}

impl fmt::Display for PerplexityModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for PerplexityModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sonar-pro" => Ok(Self::SonarPro),
            "sonar" => Ok(Self::Sonar),
            "sonar-reasoning-pro" => Ok(Self::SonarReasoningPro),
            "sonar-reasoning" => Ok(Self::SonarReasoning),
            "sonar-deep-research" => Ok(Self::SonarDeepResearch),
            _ => Err(format!(
                "Unknown Perplexity model: {s}. Any valid model ID can be used."
            )),
        }
    }
}

/// Available `OpenAI` models
/// Note: These are fallback models - prefer fetching from API
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIModel {
    #[serde(rename = "gpt-5.4")]
    Gpt54,
    #[serde(rename = "gpt-5.4-mini")]
    Gpt54Mini,
    #[serde(rename = "gpt-5")]
    Gpt5,
    #[serde(rename = "gpt-4.1")]
    Gpt41,
    #[serde(rename = "gpt-4.1-mini")]
    Gpt41Mini,
    #[serde(rename = "gpt-4.1-nano")]
    Gpt41Nano,
    #[serde(rename = "gpt-4o")]
    Gpt4o,
    #[serde(rename = "gpt-4o-mini")]
    Gpt4oMini,
    #[serde(rename = "o3")]
    O3,
    #[serde(rename = "o3-mini")]
    O3Mini,
    #[serde(rename = "o4-mini")]
    O4Mini,
    #[serde(rename = "o1")]
    O1,
    #[serde(rename = "o1-mini")]
    O1Mini,
    #[serde(rename = "o1-pro")]
    O1Pro,
    #[serde(rename = "gpt-4-turbo")]
    Gpt4Turbo,
}

impl OpenAIModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gpt54 => "gpt-5.4",
            Self::Gpt54Mini => "gpt-5.4-mini",
            Self::Gpt5 => "gpt-5",
            Self::Gpt41 => "gpt-4.1",
            Self::Gpt41Mini => "gpt-4.1-mini",
            Self::Gpt41Nano => "gpt-4.1-nano",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::O4Mini => "o4-mini",
            Self::O1 => "o1",
            Self::O1Mini => "o1-mini",
            Self::O1Pro => "o1-pro",
            Self::Gpt4Turbo => "gpt-4-turbo",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Gpt54,
            Self::Gpt54Mini,
            Self::Gpt5,
            Self::Gpt41,
            Self::Gpt41Mini,
            Self::Gpt41Nano,
            Self::Gpt4o,
            Self::Gpt4oMini,
            Self::O3,
            Self::O3Mini,
            Self::O4Mini,
            Self::O1,
            Self::O1Mini,
            Self::O1Pro,
            Self::Gpt4Turbo,
        ]
    }
}

impl fmt::Display for OpenAIModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for OpenAIModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gpt-5.4" => Ok(Self::Gpt54),
            "gpt-5.4-mini" => Ok(Self::Gpt54Mini),
            "gpt-5" | "gpt-5.6-sol" => Ok(Self::Gpt5),
            "gpt-4.1" => Ok(Self::Gpt41),
            "gpt-4.1-mini" => Ok(Self::Gpt41Mini),
            "gpt-4.1-nano" => Ok(Self::Gpt41Nano),
            "gpt-4o" => Ok(Self::Gpt4o),
            "gpt-4o-mini" => Ok(Self::Gpt4oMini),
            "o3" => Ok(Self::O3),
            "o3-mini" => Ok(Self::O3Mini),
            "o4-mini" => Ok(Self::O4Mini),
            "o1" => Ok(Self::O1),
            "o1-mini" => Ok(Self::O1Mini),
            "o1-pro" => Ok(Self::O1Pro),
            "gpt-4-turbo" => Ok(Self::Gpt4Turbo),
            _ => Err(format!(
                "Unknown OpenAI model: {s}. Any valid model ID can be used."
            )),
        }
    }
}

/// Available LM Studio models
/// Note: LM Studio can use any model loaded locally, so this enum is just for `provider_utils` models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LMStudioModel {
    #[serde(rename = "local-model")]
    LocalModel,
}

impl LMStudioModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LocalModel => "local-model",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::LocalModel]
    }
}

impl fmt::Display for LMStudioModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for LMStudioModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "local-model" => Ok(Self::LocalModel),
            _ => Err(format!(
                "Unknown LM Studio model: {s}. Use the model ID directly - LM Studio accepts any model name."
            )),
        }
    }
}

/// Available Ollama models
/// Note: Ollama can use any model installed locally, so these are common fallback models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OllamaModel {
    #[serde(rename = "llama3.3:70b")]
    Llama3_3_70b,
    #[serde(rename = "llama3.2")]
    Llama3_2,
    #[serde(rename = "qwen2.5")]
    Qwen2_5,
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "codellama")]
    CodeLlama,
    #[serde(rename = "deepseek-coder")]
    DeepseekCoder,
}

impl OllamaModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Llama3_3_70b => "llama3.3:70b",
            Self::Llama3_2 => "llama3.2",
            Self::Qwen2_5 => "qwen2.5",
            Self::Mistral => "mistral",
            Self::CodeLlama => "codellama",
            Self::DeepseekCoder => "deepseek-coder",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Llama3_3_70b,
            Self::Llama3_2,
            Self::Qwen2_5,
            Self::Mistral,
            Self::CodeLlama,
            Self::DeepseekCoder,
        ]
    }
}

impl fmt::Display for OllamaModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for OllamaModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "llama3.3:70b" | "llama3.3" => Ok(Self::Llama3_3_70b),
            "llama3.2" => Ok(Self::Llama3_2),
            "qwen2.5" => Ok(Self::Qwen2_5),
            "mistral" => Ok(Self::Mistral),
            "codellama" => Ok(Self::CodeLlama),
            "deepseek-coder" => Ok(Self::DeepseekCoder),
            _ => Err(format!(
                "Unknown Ollama model: {s}. Use the model ID directly - Ollama accepts any installed model name."
            )),
        }
    }
}

/// Available Anthropic models
/// Note: These are fallback models - prefer fetching from API
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicModel {
    #[serde(rename = "claude-opus-5")]
    ClaudeOpus5,
    #[serde(rename = "claude-sonnet-5")]
    ClaudeSonnet5,
    #[serde(rename = "claude-sonnet-4-6")]
    ClaudeSonnet46,
    #[serde(rename = "claude-haiku-4-5-20251001")]
    ClaudeHaiku45,
    #[serde(rename = "claude-sonnet-4-5-20250929")]
    ClaudeSonnet45,
    #[serde(rename = "claude-opus-4-20250514")]
    ClaudeOpus4,
    #[serde(rename = "claude-sonnet-4-20250514")]
    ClaudeSonnet4,
    #[serde(rename = "claude-haiku-4-20250514")]
    ClaudeHaiku4,
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Claude35Sonnet,
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude35Haiku,
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
}

impl AnthropicModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ClaudeOpus5 => "claude-opus-5",
            Self::ClaudeSonnet5 => "claude-sonnet-5",
            Self::ClaudeSonnet46 => "claude-sonnet-4-6",
            Self::ClaudeHaiku45 => "claude-haiku-4-5-20251001",
            Self::ClaudeSonnet45 => "claude-sonnet-4-5-20250929",
            Self::ClaudeOpus4 => "claude-opus-4-20250514",
            Self::ClaudeSonnet4 => "claude-sonnet-4-20250514",
            Self::ClaudeHaiku4 => "claude-haiku-4-20250514",
            Self::Claude35Sonnet => "claude-3-5-sonnet-20241022",
            Self::Claude35Haiku => "claude-3-5-haiku-20241022",
            Self::Claude3Opus => "claude-3-opus-20240229",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::ClaudeOpus5,
            Self::ClaudeSonnet5,
            Self::ClaudeSonnet46,
            Self::ClaudeHaiku45,
            Self::ClaudeSonnet45,
            Self::ClaudeOpus4,
            Self::ClaudeSonnet4,
            Self::ClaudeHaiku4,
            Self::Claude35Sonnet,
            Self::Claude35Haiku,
            Self::Claude3Opus,
        ]
    }
}

impl fmt::Display for AnthropicModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for AnthropicModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "claude-opus-5" => Ok(Self::ClaudeOpus5),
            "claude-sonnet-5" => Ok(Self::ClaudeSonnet5),
            "claude-sonnet-4-6" | "claude-sonnet-4" => Ok(Self::ClaudeSonnet46),
            "claude-haiku-4-5-20251001" | "claude-haiku-4-5" => Ok(Self::ClaudeHaiku45),
            "claude-sonnet-4-5-20250929" | "claude-sonnet-4-5" => Ok(Self::ClaudeSonnet45),
            "claude-opus-4-20250514" => Ok(Self::ClaudeOpus4),
            "claude-sonnet-4-20250514" => Ok(Self::ClaudeSonnet4),
            "claude-haiku-4-20250514" => Ok(Self::ClaudeHaiku4),
            "claude-3-5-sonnet-20241022" => Ok(Self::Claude35Sonnet),
            "claude-3-5-haiku-20241022" => Ok(Self::Claude35Haiku),
            "claude-3-opus-20240229" => Ok(Self::Claude3Opus),
            _ => Err(format!(
                "Unknown Anthropic model: {s}. Any valid model ID can be used."
            )),
        }
    }
}

/// Available Google models (Gemini API)
///
/// Note: These are fallback models - prefer fetching from API via `fetch_available_models()`
/// Use stable model names (not preview versions) as defaults since previews expire
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GoogleModel {
    #[serde(rename = "gemini-3.7-flash")]
    Gemini37Flash,
    #[serde(rename = "gemini-3.5-flash")]
    Gemini35Flash,
    #[serde(rename = "gemini-2.5-pro")]
    Gemini25Pro,
    #[serde(rename = "gemini-2.5-flash")]
    Gemini25Flash,
    #[serde(rename = "gemini-2.0-flash")]
    Gemini20Flash,
    #[serde(rename = "gemini-1.5-flash")]
    Gemini15Flash,
    #[serde(rename = "gemini-1.5-pro")]
    Gemini15Pro,
}

impl GoogleModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gemini37Flash => "gemini-3.7-flash",
            Self::Gemini35Flash => "gemini-3.5-flash",
            Self::Gemini25Pro => "gemini-2.5-pro",
            Self::Gemini25Flash => "gemini-2.5-flash",
            Self::Gemini20Flash => "gemini-2.0-flash",
            Self::Gemini15Flash => "gemini-1.5-flash",
            Self::Gemini15Pro => "gemini-1.5-pro",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Gemini37Flash,
            Self::Gemini35Flash,
            Self::Gemini25Pro,
            Self::Gemini25Flash,
            Self::Gemini20Flash,
            Self::Gemini15Flash,
            Self::Gemini15Pro,
        ]
    }
}

impl fmt::Display for GoogleModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for GoogleModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let model_name = s.strip_prefix("models/").unwrap_or(s);

        match model_name {
            "gemini-3.7-flash" => Ok(Self::Gemini37Flash),
            "gemini-3.5-flash" => Ok(Self::Gemini35Flash),
            "gemini-2.5-pro" | "gemini-2.5-pro-latest" => Ok(Self::Gemini25Pro),
            "gemini-2.5-flash" | "gemini-2.5-flash-latest" => Ok(Self::Gemini25Flash),
            "gemini-2.0-flash" | "gemini-2.0-flash-exp" => Ok(Self::Gemini20Flash),
            "gemini-1.5-flash" | "gemini-1.5-flash-latest" => Ok(Self::Gemini15Flash),
            "gemini-1.5-pro" | "gemini-1.5-pro-latest" => Ok(Self::Gemini15Pro),
            _ => Err(format!(
                "Unknown Google model: {s}. Any valid model ID can be used. Fetch models from API for current list."
            )),
        }
    }
}

/// Available Cohere models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CohereModel {
    #[serde(rename = "command")]
    Command,
    #[serde(rename = "command-r")]
    CommandR,
    #[serde(rename = "command-r-plus")]
    CommandRPlus,
    #[serde(rename = "embed-english-v3.0")]
    EmbedEnglishV3,
    #[serde(rename = "embed-multilingual-v3.0")]
    EmbedMultilingualV3,
    #[serde(rename = "rerank-english-v3.0")]
    RerankEnglishV3,
    #[serde(rename = "rerank-multilingual-v3.0")]
    RerankMultilingualV3,
}

impl CohereModel {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::CommandR => "command-r",
            Self::CommandRPlus => "command-r-plus",
            Self::EmbedEnglishV3 => "embed-english-v3.0",
            Self::EmbedMultilingualV3 => "embed-multilingual-v3.0",
            Self::RerankEnglishV3 => "rerank-english-v3.0",
            Self::RerankMultilingualV3 => "rerank-multilingual-v3.0",
        }
    }

    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Command,
            Self::CommandR,
            Self::CommandRPlus,
            Self::EmbedEnglishV3,
            Self::EmbedMultilingualV3,
            Self::RerankEnglishV3,
            Self::RerankMultilingualV3,
        ]
    }
}

impl fmt::Display for CohereModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for CohereModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "command" => Ok(Self::Command),
            "command-r" => Ok(Self::CommandR),
            "command-r-plus" => Ok(Self::CommandRPlus),
            "embed-english-v3.0" => Ok(Self::EmbedEnglishV3),
            "embed-multilingual-v3.0" => Ok(Self::EmbedMultilingualV3),
            "rerank-english-v3.0" => Ok(Self::RerankEnglishV3),
            "rerank-multilingual-v3.0" => Ok(Self::RerankMultilingualV3),
            _ => Err(format!(
                "Unknown Cohere model: {s}. Use the model ID directly if it's not in the enum."
            )),
        }
    }
}

/// Available Azure `OpenAI` models
/// Note: Azure `OpenAI` uses deployment names, so this is just a wrapper
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AzureOpenAIModel {
    /// Deployment name/ID
    pub deployment: String,
}

impl AzureOpenAIModel {
    pub fn new(deployment: impl Into<String>) -> Self {
        Self {
            deployment: deployment.into(),
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.deployment
    }

    #[must_use]
    pub const fn all() -> Vec<Self> {
        // Azure OpenAI deployments are user-defined, so we can't enumerate them
        // Return empty vec - users should specify their deployment name directly
        vec![]
    }
}

impl fmt::Display for AzureOpenAIModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.deployment)
    }
}

impl std::str::FromStr for AzureOpenAIModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

/// Trait for providers that can list their available models
pub trait ModelProvider: Send + Sync {
    /// Get a list of available model identifiers for this provider
    fn available_models(&self) -> Vec<String>;

    /// Get the provider type
    fn provider_type(&self) -> ProviderType;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn provider_type_parses_aliases_and_display_roundtrips() {
        assert_eq!(
            ProviderType::from_str("azure-openai").unwrap(),
            ProviderType::AzureOpenAI
        );
        assert_eq!(
            ProviderType::from_str("azure_openai").unwrap(),
            ProviderType::AzureOpenAI
        );
        assert_eq!(
            ProviderType::from_str("AzureOpenAI").unwrap(),
            ProviderType::AzureOpenAI
        );
        assert_eq!(
            ProviderType::from_str("cf-aig").unwrap(),
            ProviderType::CloudflareAIGateway
        );
        assert_eq!(
            ProviderType::from_str("cloudflare").unwrap(),
            ProviderType::CloudflareAIGateway
        );
        assert_eq!(
            ProviderType::from_str("OPENAI").unwrap(),
            ProviderType::OpenAI
        );
        assert_eq!(
            ProviderType::from_str("gemini").unwrap(),
            ProviderType::Google
        );
        assert_eq!(
            ProviderType::from_str("GEMINI").unwrap(),
            ProviderType::Google
        );
        // Alias maps to Google; display round-trips as "google".
        assert_eq!(ProviderType::Google.to_string(), "google");

        for provider in all_provider_types() {
            let displayed = provider.to_string();
            assert_eq!(
                ProviderType::from_str(&displayed).unwrap(),
                provider,
                "display {displayed} must parse back so a thin-client turn does not 400"
            );
            match provider {
                ProviderType::Bedrock
                | ProviderType::OpenRouter
                | ProviderType::Perplexity
                | ProviderType::OpenAI
                | ProviderType::LMStudio
                | ProviderType::Anthropic
                | ProviderType::Google
                | ProviderType::Cohere
                | ProviderType::AzureOpenAI
                | ProviderType::Ollama
                | ProviderType::CloudflareAIGateway
                | ProviderType::External => {}
                #[cfg(feature = "local-inference-base")]
                ProviderType::LocalInference => {}
            }
        }
    }

    fn all_provider_types() -> Vec<ProviderType> {
        #[cfg_attr(
            not(feature = "local-inference-base"),
            expect(unused_mut, reason = "local-inference-base pushes one more")
        )]
        let mut providers = vec![
            ProviderType::Bedrock,
            ProviderType::OpenRouter,
            ProviderType::Perplexity,
            ProviderType::OpenAI,
            ProviderType::LMStudio,
            ProviderType::Anthropic,
            ProviderType::Google,
            ProviderType::Cohere,
            ProviderType::AzureOpenAI,
            ProviderType::Ollama,
            ProviderType::CloudflareAIGateway,
            ProviderType::External,
        ];
        #[cfg(feature = "local-inference-base")]
        providers.push(ProviderType::LocalInference);
        providers
    }

    #[test]
    fn provider_type_unknown_lists_supported() {
        let err = ProviderType::from_str("not-a-provider").unwrap_err();
        assert!(err.contains("Unknown provider"));
        assert!(err.contains("openai"));
        assert!(err.contains("anthropic"));
        assert!(err.contains("gemini"));
    }

    #[cfg(not(feature = "local-inference-base"))]
    #[test]
    fn a_provider_left_out_of_the_build_names_the_feature() {
        assert_eq!(
            ProviderType::from_str("local-inference"),
            Err(
                "local-inference is not in this build. Rebuild with --features local-inference."
                    .to_string()
            )
        );
        assert_eq!(
            ProviderType::feature_left_out_of_build("LocalInference"),
            Some("local-inference")
        );
        assert_eq!(ProviderType::feature_left_out_of_build("openai"), None);
        assert_eq!(
            ProviderType::feature_left_out_of_build("not-a-provider"),
            None
        );
    }

    #[cfg(feature = "local-inference-base")]
    #[test]
    fn provider_type_parses_local_inference_aliases() {
        assert_eq!(
            ProviderType::from_str("local-inference").unwrap(),
            ProviderType::LocalInference
        );
        assert_eq!(
            ProviderType::from_str("localinference").unwrap(),
            ProviderType::LocalInference
        );
        assert_eq!(
            ProviderType::from_str("LOCAL-INFERENCE").unwrap(),
            ProviderType::LocalInference
        );
        assert_eq!(ProviderType::LocalInference.to_string(), "local-inference");
    }

    #[test]
    fn fallback_catalogs_include_current_live_ids() {
        assert!(
            AnthropicModel::all()
                .iter()
                .any(|m| m.as_str() == defaults::ANTHROPIC)
        );
        assert!(
            OpenAIModel::all()
                .iter()
                .any(|m| m.as_str() == defaults::OPENAI)
        );
        assert!(
            GoogleModel::all()
                .iter()
                .any(|m| m.as_str() == defaults::GOOGLE)
        );
        assert!(
            OpenRouterModel::all()
                .iter()
                .any(|m| m.as_str() == defaults::OPENROUTER)
        );
        assert_eq!(
            "claude-sonnet-4-6"
                .parse::<AnthropicModel>()
                .unwrap()
                .as_str(),
            defaults::ANTHROPIC
        );
        assert_eq!(
            "claude-sonnet-4"
                .parse::<AnthropicModel>()
                .unwrap()
                .as_str(),
            defaults::ANTHROPIC
        );
        assert_eq!(
            "gpt-5.4".parse::<OpenAIModel>().unwrap().as_str(),
            defaults::OPENAI
        );
        assert_eq!(
            "anthropic/claude-sonnet-4.6"
                .parse::<OpenRouterModel>()
                .unwrap()
                .as_str(),
            defaults::OPENROUTER
        );
        assert_eq!(
            "x-ai/grok-4.6".parse::<OpenRouterModel>().unwrap().as_str(),
            "x-ai/grok-4.6"
        );
        assert_eq!(
            "gemini-3.5-flash".parse::<GoogleModel>().unwrap().as_str(),
            "gemini-3.5-flash"
        );
    }
}
