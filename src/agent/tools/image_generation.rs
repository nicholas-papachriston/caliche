use crate::agent::tools::{Tool, ToolCapability, ToolMeta, ToolResult};
use crate::artifacts::{ArtifactMetadata, ArtifactStore};
use crate::llm::{ImageGenerationProvider, ImageGenerationRequest, ImageOutputFormat};
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ImageGenerationTool {
    provider: Arc<dyn ImageGenerationProvider>,
    artifact_store: Arc<ArtifactStore>,
}

impl ImageGenerationTool {
    #[must_use]
    pub fn new(
        provider: Arc<dyn ImageGenerationProvider>,
        artifact_store: Arc<ArtifactStore>,
    ) -> Self {
        Self {
            provider,
            artifact_store,
        }
    }
}

impl ToolMeta for ImageGenerationTool {
    fn name(&self) -> &str {
        crate::agent::tools::names::TOOL_GENERATE_IMAGE
    }

    fn description(&self) -> &'static str {
        "Generate images from a text prompt and save them as workspace artifacts"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed prompt describing the image to generate"
                },
                "size": {
                    "type": "string",
                    "description": "Provider-specific image size, such as 1024x1024"
                },
                "quality": {
                    "type": "string",
                    "description": "Provider-specific quality setting, such as standard, hd, low, medium, or high"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "jpg", "webp"],
                    "description": "Requested output format"
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "description": "Number of images to generate"
                }
            },
            "required": ["prompt"]
        })
    }

    fn is_readonly(&self) -> bool {
        false
    }

    fn capabilities(&self) -> Vec<ToolCapability> {
        vec![ToolCapability::RequiresCredentials, ToolCapability::Audited]
    }
}

#[async_trait]
impl Tool for ImageGenerationTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        match self.execute_inner(arguments).await {
            Ok(result) => result,
            Err(error) => ToolResult::error(error.to_string()),
        }
    }
}

impl ImageGenerationTool {
    async fn execute_inner(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        let prompt = arguments
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("prompt is required"))?
            .to_string();
        let size = optional_string(&arguments, "size");
        let quality = optional_string(&arguments, "quality");
        let output_format = optional_string(&arguments, "output_format")
            .map(|value| value.parse::<ImageOutputFormat>())
            .transpose()
            .map_err(anyhow::Error::msg)?;
        let count = optional_count(&arguments)?;

        let request = ImageGenerationRequest {
            prompt: prompt.clone(),
            model: None,
            size: size.clone(),
            quality: quality.clone(),
            output_format,
            count,
        };
        let response = self.provider.generate(request).await?;

        let mut records = Vec::with_capacity(response.images.len());
        for image in &response.images {
            let metadata = ArtifactMetadata {
                prompt: prompt.clone(),
                revised_prompt: image.revised_prompt.clone(),
                provider: response.provider.clone(),
                model: response.model.clone(),
                size: size.clone(),
                quality: quality.clone(),
                output_format: output_format.map(|format| format.as_api_value().to_string()),
                ..ArtifactMetadata::default()
            };
            records.push(self.artifact_store.save_image(image, metadata).await?);
        }

        let saved_paths = records
            .iter()
            .map(|record| record.path.display().to_string())
            .collect::<Vec<_>>();
        let payload = serde_json::json!({
            "provider": response.provider,
            "model": response.model,
            "artifacts": records,
            "artifact_root": self.artifact_store.root().display().to_string(),
        });
        let mut result = ToolResult::success_with_payload(
            format!(
                "Generated {} image(s):\n{}",
                saved_paths.len(),
                saved_paths.join("\n")
            ),
            payload,
        );
        result.usage = response.usage;
        Ok(result)
    }
}

fn optional_string(arguments: &serde_json::Value, name: &str) -> Option<String> {
    arguments
        .get(name)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn optional_count(arguments: &serde_json::Value) -> anyhow::Result<Option<u8>> {
    let Some(value) = arguments.get("count") else {
        return Ok(None);
    };
    let Some(raw) = value.as_u64() else {
        anyhow::bail!("count must be an integer between 1 and 4");
    };
    if !(1..=4).contains(&raw) {
        anyhow::bail!("count must be between 1 and 4");
    }
    let count = u8::try_from(raw).map_err(|e| anyhow::anyhow!("invalid count: {e}"))?;
    Ok(Some(count))
}
