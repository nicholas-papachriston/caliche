use async_trait::async_trait;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::traits::credentials::{CredentialProvider, Credentials};

/// Supported image output formats for generated artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ImageOutputFormat {
    #[default]
    Png,
    Jpeg,
    Webp,
}

impl ImageOutputFormat {
    #[must_use]
    pub const fn as_extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
        }
    }

    #[must_use]
    pub const fn as_media_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
        }
    }

    #[must_use]
    pub const fn as_api_value(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }
}

impl std::str::FromStr for ImageOutputFormat {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::Webp),
            other => Err(format!(
                "unsupported image output format '{other}'; expected png, jpeg, or webp"
            )),
        }
    }
}

/// Request for image generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageOutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u8>,
}

/// A single generated image returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub data: Vec<u8>,
    pub media_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// Response containing one or more generated images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub images: Vec<GeneratedImage>,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<crate::messages::TokenUsage>,
}

/// Trait for providers that can generate images from text prompts.
#[async_trait]
pub trait ImageGenerationProvider: Send + Sync + std::fmt::Debug {
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> anyhow::Result<ImageGenerationResponse>;

    fn name(&self) -> &str;

    fn model(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct OpenAIImageGenerationProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIImageGenerationProvider {
    #[must_use]
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com/v1".to_string(),
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[derive(Debug, Serialize)]
struct OpenAIImageRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<&'static str>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageResponse {
    #[serde(default)]
    data: Vec<OpenAIImageData>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    #[serde(default)]
    data: Vec<OpenAIModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelInfo {
    id: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageData {
    b64_json: Option<String>,
    url: Option<String>,
    revised_prompt: Option<String>,
}

#[async_trait]
impl ImageGenerationProvider for OpenAIImageGenerationProvider {
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> anyhow::Result<ImageGenerationResponse> {
        let model = request.model.as_deref().unwrap_or(&self.model);
        let output_format = request.output_format.unwrap_or_default();
        let payload = OpenAIImageRequest {
            model,
            prompt: &request.prompt,
            n: request.count,
            size: request.size.as_deref(),
            quality: request.quality.as_deref(),
            response_format: if model.starts_with("gpt-image") {
                None
            } else {
                Some("b64_json")
            },
            output_format: if model.starts_with("gpt-image") {
                Some(output_format.as_api_value())
            } else {
                None
            },
        };

        let response = self
            .client
            .post(format!(
                "{}/images/generations",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to send OpenAI image request: {e}"))?;
        let response =
            crate::llm::provider_utils::check_response_status(response, "openai").await?;
        let parsed: OpenAIImageResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse OpenAI image response: {e}"))?;

        let mut images = Vec::with_capacity(parsed.data.len());
        for image in parsed.data {
            let (data, media_type) = if let Some(encoded) = image.b64_json {
                let data = base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|e| anyhow::anyhow!("failed to decode OpenAI image payload: {e}"))?;
                let media_type = detect_image_media_type(&data)
                    .unwrap_or_else(|| output_format.as_media_type().to_string());
                (data, media_type)
            } else if let Some(url) = image.url {
                let response = self
                    .client
                    .get(url)
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to fetch OpenAI image URL: {e}"))?;
                let media_type = response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .map(ToOwned::to_owned);
                let data = response
                    .bytes()
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to read OpenAI image bytes: {e}"))?
                    .to_vec();
                let media_type = media_type
                    .or_else(|| detect_image_media_type(&data))
                    .unwrap_or_else(|| output_format.as_media_type().to_string());
                (data, media_type)
            } else {
                anyhow::bail!("OpenAI image response did not include b64_json or url");
            };
            images.push(GeneratedImage {
                data,
                media_type,
                revised_prompt: image.revised_prompt,
            });
        }

        Ok(ImageGenerationResponse {
            images,
            provider: self.name().to_string(),
            model: model.to_string(),
            usage: None,
        })
    }

    fn name(&self) -> &'static str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatImageUsage, GoogleImageUsage};

    #[test]
    fn usage_wire_fields_preserve_boundary_values() {
        let google: GoogleImageUsage = serde_json::from_value(serde_json::json!({
            "promptTokenCount": u32::MAX,
            "candidatesTokenCount": u32::MAX,
            "totalTokenCount": u32::MAX
        }))
        .expect("Google usage");
        assert_eq!(google.prompt, u32::MAX);
        assert_eq!(google.candidates, u32::MAX);
        assert_eq!(google.total, u32::MAX);

        let chat: ChatImageUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": u32::MAX,
            "completion_tokens": u32::MAX,
            "total_tokens": u32::MAX
        }))
        .expect("chat usage");
        assert_eq!(chat.prompt, u32::MAX);
        assert_eq!(chat.completion, u32::MAX);
        assert_eq!(chat.total, u32::MAX);
    }
}

#[derive(Debug, Clone)]
pub struct GoogleImageGenerationProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GoogleImageGenerationProvider {
    #[must_use]
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: normalize_google_image_model(model.into()),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIChatImageGenerationProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    provider_name: &'static str,
    extra_headers: Vec<(&'static str, String)>,
}

impl OpenAIChatImageGenerationProvider {
    #[must_use]
    pub fn new(
        provider_name: &'static str,
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            provider_name,
            extra_headers: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.extra_headers.push((name, value.into()));
        self
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleImageRequest<'a> {
    contents: Vec<GoogleImageContent<'a>>,
    generation_config: GoogleImageGenerationConfig,
}

#[derive(Debug, Serialize)]
struct GoogleImageContent<'a> {
    role: &'static str,
    parts: Vec<GoogleImageTextPart<'a>>,
}

#[derive(Debug, Serialize)]
struct GoogleImageTextPart<'a> {
    text: &'a str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleImageGenerationConfig {
    response_modalities: Vec<&'static str>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleImageResponse {
    #[serde(default)]
    candidates: Vec<GoogleImageCandidate>,
    #[serde(default)]
    usage_metadata: Option<GoogleImageUsage>,
}

#[derive(Debug, Deserialize)]
struct GoogleImageCandidate {
    content: Option<GoogleImageResponseContent>,
}

#[derive(Debug, Deserialize)]
struct GoogleImageResponseContent {
    #[serde(default)]
    parts: Vec<GoogleImageResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GoogleImageResponsePart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GoogleInlineData,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
struct GoogleImageUsage {
    #[serde(default, rename = "promptTokenCount")]
    prompt: u32,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates: u32,
    #[serde(default, rename = "totalTokenCount")]
    total: u32,
}

#[derive(Debug, Deserialize)]
struct GoogleModelsResponse {
    #[serde(default)]
    models: Vec<GoogleModelInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModelInfo {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ChatImageRequest<'a> {
    model: &'a str,
    messages: Vec<ChatImageMessage<'a>>,
    modalities: Vec<&'static str>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatImageMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatImageResponse {
    #[serde(default)]
    choices: Vec<ChatImageChoice>,
    #[serde(default)]
    usage: Option<ChatImageUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatImageChoice {
    message: ChatImageAssistantMessage,
}

#[derive(Debug, Deserialize)]
struct ChatImageAssistantMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    images: Vec<ChatGeneratedImage>,
}

#[derive(Debug, Deserialize)]
struct ChatGeneratedImage {
    #[serde(rename = "image_url")]
    image_url: ChatGeneratedImageUrl,
}

#[derive(Debug, Deserialize)]
struct ChatGeneratedImageUrl {
    url: String,
}

#[derive(Debug, Deserialize)]
struct ChatImageUsage {
    #[serde(default, rename = "prompt_tokens")]
    prompt: u32,
    #[serde(default, rename = "completion_tokens")]
    completion: u32,
    #[serde(default, rename = "total_tokens")]
    total: u32,
}

#[async_trait]
impl ImageGenerationProvider for GoogleImageGenerationProvider {
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> anyhow::Result<ImageGenerationResponse> {
        let model = request
            .model
            .as_deref()
            .map_or_else(|| self.model.clone(), normalize_google_image_model);
        let payload = GoogleImageRequest {
            contents: vec![GoogleImageContent {
                role: "user",
                parts: vec![GoogleImageTextPart {
                    text: &request.prompt,
                }],
            }],
            generation_config: GoogleImageGenerationConfig {
                response_modalities: vec!["TEXT", "IMAGE"],
            },
        };

        let endpoint = format!(
            "{}/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            model
        );
        let response = self
            .client
            .post(endpoint)
            .query(&[("key", &self.api_key)])
            .json(&payload)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("failed to send Google image request: {e}"))?;
        let response =
            crate::llm::provider_utils::check_response_status(response, "google").await?;
        let parsed: GoogleImageResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse Google image response: {e}"))?;

        let mut images = Vec::new();
        let mut revised_prompt = None;
        for candidate in parsed.candidates {
            if let Some(content) = candidate.content {
                for part in content.parts {
                    match part {
                        GoogleImageResponsePart::Text { text } => {
                            if revised_prompt.is_none() && !text.trim().is_empty() {
                                revised_prompt = Some(text);
                            }
                        }
                        GoogleImageResponsePart::InlineData { inline_data } => {
                            let data = base64::engine::general_purpose::STANDARD
                                .decode(inline_data.data)
                                .map_err(|e| {
                                    anyhow::anyhow!("failed to decode Google image payload: {e}")
                                })?;
                            let media_type =
                                detect_image_media_type(&data).unwrap_or(inline_data.mime_type);
                            images.push(GeneratedImage {
                                data,
                                media_type,
                                revised_prompt: revised_prompt.clone(),
                            });
                        }
                    }
                }
            }
        }

        if images.is_empty() {
            anyhow::bail!("Google image response did not include generated image data");
        }

        Ok(ImageGenerationResponse {
            images,
            provider: self.name().to_string(),
            model,
            usage: parsed.usage_metadata.map(|usage| {
                crate::messages::TokenUsage::from_counts(
                    Some(usage.prompt),
                    Some(usage.candidates),
                    Some(usage.total),
                )
            }),
        })
    }

    fn name(&self) -> &'static str {
        "google"
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl ImageGenerationProvider for OpenAIChatImageGenerationProvider {
    async fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> anyhow::Result<ImageGenerationResponse> {
        let model = request.model.as_deref().unwrap_or(&self.model);
        let payload = ChatImageRequest {
            model,
            messages: vec![ChatImageMessage {
                role: "user",
                content: &request.prompt,
            }],
            modalities: vec!["image", "text"],
            stream: false,
        };

        let mut req = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.base_url.trim_end_matches('/')
            ))
            .header("Content-Type", "application/json");
        if !self.api_key.trim().is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        for (name, value) in &self.extra_headers {
            req = req.header(*name, value);
        }
        let response =
            req.json(&payload).send().await.map_err(|e| {
                anyhow::anyhow!("failed to send {} image request: {e}", self.name())
            })?;
        let response =
            crate::llm::provider_utils::check_response_status(response, self.name()).await?;
        let parsed: ChatImageResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("failed to parse {} image response: {e}", self.name()))?;

        let mut images = Vec::new();
        let mut revised_prompt = None;
        for choice in parsed.choices {
            if revised_prompt.is_none()
                && let Some(content) = choice.message.content
                && !content.trim().is_empty()
            {
                revised_prompt = Some(content);
            }
            for image in choice.message.images {
                let (data, media_type) =
                    decode_data_url_image(&image.image_url.url).ok_or_else(|| {
                        anyhow::anyhow!(
                            "{} image response included an unsupported image URL format",
                            self.name()
                        )
                    })?;
                images.push(GeneratedImage {
                    data,
                    media_type,
                    revised_prompt: revised_prompt.clone(),
                });
            }
        }

        if images.is_empty() {
            anyhow::bail!(
                "{} image response did not include generated image data",
                self.name()
            );
        }

        Ok(ImageGenerationResponse {
            images,
            provider: self.name().to_string(),
            model: model.to_string(),
            usage: parsed.usage.map(|usage| {
                crate::messages::TokenUsage::from_counts(
                    Some(usage.prompt),
                    Some(usage.completion),
                    Some(usage.total),
                )
            }),
        })
    }

    fn name(&self) -> &str {
        self.provider_name
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[must_use]
pub const fn default_image_generation_model(
    provider: crate::llm::ProviderType,
) -> Option<&'static str> {
    match provider {
        crate::llm::ProviderType::OpenAI => Some("gpt-image-1"),
        crate::llm::ProviderType::Google => Some("models/gemini-2.5-flash-image"),
        crate::llm::ProviderType::OpenRouter => Some("google/gemini-2.5-flash-image"),
        crate::llm::ProviderType::CloudflareAIGateway => {
            Some("google-ai-studio/gemini-2.5-flash-image")
        }
        _ => None,
    }
}

#[must_use]
pub fn resolve_image_generation_model(
    provider: crate::llm::ProviderType,
    configured: Option<String>,
) -> Option<String> {
    configured
        .or_else(|| std::env::var("CALICHE_IMAGE_MODEL").ok())
        .or_else(|| match provider {
            crate::llm::ProviderType::OpenAI => std::env::var("OPENAI_IMAGE_MODEL").ok(),
            crate::llm::ProviderType::OpenRouter => std::env::var("OPENROUTER_IMAGE_MODEL").ok(),
            crate::llm::ProviderType::CloudflareAIGateway => {
                std::env::var("CLOUDFLARE_AIG_IMAGE_MODEL").ok()
            }
            crate::llm::ProviderType::Google => std::env::var("GOOGLE_IMAGE_MODEL")
                .ok()
                .or_else(|| std::env::var("GEMINI_IMAGE_MODEL").ok()),
            _ => None,
        })
        .or_else(|| default_image_generation_model(provider).map(ToOwned::to_owned))
}

/// Fetch image-generation-capable model IDs from the provider API.
///
/// Falls back to known compatible models when credentials are unavailable or the
/// provider API cannot be queried.
/// # Errors
///
/// Returns an error if the language-model operation cannot be completed.
pub async fn discover_image_generation_models(
    provider: crate::llm::ProviderType,
    credential_provider: Arc<dyn CredentialProvider>,
) -> anyhow::Result<Vec<String>> {
    let provider_name = provider.to_string();
    let credentials = credential_provider
        .get_credentials(&provider_name)
        .await
        .ok();
    match provider {
        crate::llm::ProviderType::OpenAI => {
            let Some(api_key) = credentials.as_ref().and_then(Credentials::as_api_key) else {
                return Ok(fallback_image_generation_models(provider));
            };
            fetch_openai_image_models(api_key).await
        }
        crate::llm::ProviderType::Google => {
            let Some(api_key) = credentials.as_ref().and_then(Credentials::as_api_key) else {
                return Ok(fallback_image_generation_models(provider));
            };
            fetch_google_image_models(api_key).await
        }
        crate::llm::ProviderType::OpenRouter => {
            let Some(api_key) = credentials.as_ref().and_then(Credentials::as_api_key) else {
                return Ok(fallback_image_generation_models(provider));
            };
            fetch_openrouter_image_models(api_key).await
        }
        crate::llm::ProviderType::CloudflareAIGateway => {
            Ok(fallback_image_generation_models(provider))
        }
        _ => Ok(Vec::new()),
    }
}

#[must_use]
pub fn fallback_image_generation_models(provider: crate::llm::ProviderType) -> Vec<String> {
    match provider {
        crate::llm::ProviderType::OpenAI => {
            vec!["gpt-image-1".to_string(), "dall-e-3".to_string()]
        }
        crate::llm::ProviderType::Google => vec![
            "models/gemini-2.5-flash-image".to_string(),
            "models/gemini-3.1-flash-image-preview".to_string(),
            "models/gemini-3-pro-image-preview".to_string(),
        ],
        crate::llm::ProviderType::OpenRouter => vec![
            "google/gemini-2.5-flash-image".to_string(),
            "openai/gpt-image-1".to_string(),
            "google/gemini-3.1-flash-image-preview".to_string(),
            "google/gemini-3-pro-image-preview".to_string(),
        ],
        crate::llm::ProviderType::CloudflareAIGateway => vec![
            "google-ai-studio/gemini-2.5-flash-image".to_string(),
            "openai/gpt-5-image".to_string(),
            "openai/gpt-image-1.5".to_string(),
        ],
        _ => Vec::new(),
    }
}

async fn fetch_openai_image_models(api_key: &str) -> anyhow::Result<Vec<String>> {
    let response = reqwest::Client::new()
        .get("https://api.openai.com/v1/models")
        .bearer_auth(api_key)
        .send()
        .await;
    let Ok(response) = response else {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenAI,
        ));
    };
    if !response.status().is_success() {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenAI,
        ));
    }
    let parsed = response.json::<OpenAIModelsResponse>().await;
    let Ok(parsed) = parsed else {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenAI,
        ));
    };
    let mut models = parsed
        .data
        .into_iter()
        .filter_map(|model| {
            let id = model.id;
            if is_openai_image_model(&id) {
                Some(id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    models.sort_by(|a, b| {
        image_model_priority(b)
            .cmp(&image_model_priority(a))
            .then(a.cmp(b))
    });
    models.dedup();
    if models.is_empty() {
        Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenAI,
        ))
    } else {
        Ok(models)
    }
}

#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    #[serde(default)]
    data: Vec<OpenRouterImageModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterImageModelInfo {
    id: String,
    #[serde(default)]
    output_modalities: Vec<String>,
}

async fn fetch_openrouter_image_models(api_key: &str) -> anyhow::Result<Vec<String>> {
    let response = reqwest::Client::new()
        .get("https://openrouter.ai/api/v1/models")
        .query(&[("output_modalities", "image")])
        .bearer_auth(api_key)
        .header("HTTP-Referer", crate::llm::OPENROUTER_HTTP_REFERER)
        .header("X-Title", crate::llm::OPENROUTER_X_TITLE)
        .send()
        .await;
    let Ok(response) = response else {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenRouter,
        ));
    };
    if !response.status().is_success() {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenRouter,
        ));
    }
    let parsed = response.json::<OpenRouterModelsResponse>().await;
    let Ok(parsed) = parsed else {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenRouter,
        ));
    };
    let mut models = parsed
        .data
        .into_iter()
        .filter_map(|model| {
            if model.output_modalities.iter().any(|m| m == "image") {
                Some(model.id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    models.sort_by(|a, b| {
        image_model_priority(b)
            .cmp(&image_model_priority(a))
            .then(a.cmp(b))
    });
    models.dedup();
    if models.is_empty() {
        Ok(fallback_image_generation_models(
            crate::llm::ProviderType::OpenRouter,
        ))
    } else {
        Ok(models)
    }
}

async fn fetch_google_image_models(api_key: &str) -> anyhow::Result<Vec<String>> {
    let response = reqwest::Client::new()
        .get("https://generativelanguage.googleapis.com/v1beta/models")
        .query(&[("key", api_key)])
        .send()
        .await;
    let Ok(response) = response else {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::Google,
        ));
    };
    if !response.status().is_success() {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::Google,
        ));
    }
    let parsed = response.json::<GoogleModelsResponse>().await;
    let Ok(parsed) = parsed else {
        return Ok(fallback_image_generation_models(
            crate::llm::ProviderType::Google,
        ));
    };
    let mut models = parsed
        .models
        .into_iter()
        .filter_map(|model| {
            let supports_generate_content = model
                .supported_generation_methods
                .iter()
                .any(|method| method == "generateContent");
            if supports_generate_content && is_google_gemini_image_model(&model.name) {
                Some(model.name)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    models.sort_by(|a, b| {
        image_model_priority(b)
            .cmp(&image_model_priority(a))
            .then(a.cmp(b))
    });
    models.dedup();
    if models.is_empty() {
        Ok(fallback_image_generation_models(
            crate::llm::ProviderType::Google,
        ))
    } else {
        Ok(models)
    }
}

fn is_openai_image_model(id: &str) -> bool {
    id.starts_with("gpt-image") || id.starts_with("dall-e")
}

fn is_google_gemini_image_model(name: &str) -> bool {
    name.contains("gemini") && name.contains("image")
}

fn image_model_priority(model: &str) -> u8 {
    if model.contains("gpt-image") || model.contains("gemini-2.5-flash-image") {
        4
    } else if model.contains("3.1-flash-image") {
        3
    } else if model.contains("3-pro-image") {
        2
    } else {
        u8::from(model.contains("dall-e-3"))
    }
}

fn decode_data_url_image(url: &str) -> Option<(Vec<u8>, String)> {
    let (metadata, encoded) = url.split_once(',')?;
    if !metadata.starts_with("data:image/") || !metadata.contains(";base64") {
        return None;
    }
    let media_type = metadata
        .strip_prefix("data:")
        .and_then(|value| value.split(';').next())
        .unwrap_or("image/png")
        .to_string();
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    Some((data, media_type))
}

fn normalize_google_image_model(model: impl AsRef<str>) -> String {
    let trimmed = model.as_ref().trim();
    if trimmed.is_empty() {
        return "models/gemini-2.5-flash-image".to_string();
    }
    if trimmed.starts_with("models/") {
        trimmed.to_string()
    } else {
        format!("models/{trimmed}")
    }
}

fn detect_image_media_type(data: &[u8]) -> Option<String> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png".to_string())
    } else if data.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg".to_string())
    } else if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp".to_string())
    } else {
        None
    }
}
