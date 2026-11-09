//! Context compression for tool outputs and large messages.

use crate::agent::memory::token_counter::TokenCounter;
use crate::llm::Provider;
use crate::messages::{CompletionRequest, Message, MessageContent, MessageRole};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Compression strategy for reducing message size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// Summarize using LLM
    Summarize,
    /// Extract key points only
    ExtractKeyPoints,
    /// Truncate to maximum length
    Truncate { max_length: usize },
    /// Combine multiple strategies
    Hybrid,
}

/// Trait for compressing messages.
#[async_trait]
pub trait MessageCompressor: Send + Sync {
    /// Compress a message to fit within token budget.
    async fn compress(&self, message: &Message, max_tokens: usize) -> Result<CompressedMessage>;
}

/// Result of message compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMessage {
    /// Compressed message content
    pub message: Message,
    /// Original token count
    pub original_tokens: usize,
    /// Compressed token count
    pub compressed_tokens: usize,
    /// Compression ratio (compressed / original)
    pub compression_ratio: f32,
    /// Whether compression was lossy
    pub lossy: bool,
}

impl CompressedMessage {
    /// Create a new compressed message result.
    #[must_use]
    pub fn new(message: Message, original_tokens: usize, compressed_tokens: usize) -> Self {
        let compression_ratio = if original_tokens > 0 {
            compressed_tokens as f32 / original_tokens as f32
        } else {
            1.0
        };
        let lossy = compressed_tokens < original_tokens;

        Self {
            message,
            original_tokens,
            compressed_tokens,
            compression_ratio,
            lossy,
        }
    }
}

/// LLM-based message compressor.
pub struct LlmMessageCompressor {
    provider: Arc<dyn Provider>,
    token_counter: Arc<dyn TokenCounter>,
    strategy: CompressionStrategy,
    max_summary_tokens: usize,
}

impl LlmMessageCompressor {
    /// Create a new LLM compressor.
    pub fn new(
        provider: Arc<dyn Provider>,
        token_counter: Arc<dyn TokenCounter>,
        strategy: CompressionStrategy,
    ) -> Self {
        Self {
            provider,
            token_counter,
            strategy,
            max_summary_tokens: 500,
        }
    }

    /// Set maximum tokens for summary.
    #[must_use]
    pub const fn with_max_summary_tokens(mut self, max_tokens: usize) -> Self {
        self.max_summary_tokens = max_tokens;
        self
    }
}

#[async_trait]
impl MessageCompressor for LlmMessageCompressor {
    async fn compress(&self, message: &Message, max_tokens: usize) -> Result<CompressedMessage> {
        let original_tokens = self.token_counter.estimate_message_tokens(message);

        // If already within budget, return as-is
        if original_tokens <= max_tokens {
            return Ok(CompressedMessage::new(
                message.clone(),
                original_tokens,
                original_tokens,
            ));
        }

        let compressed = match self.strategy {
            CompressionStrategy::Summarize => self.compress_summarize(message, max_tokens).await?,
            CompressionStrategy::ExtractKeyPoints => {
                self.compress_extract_key_points(message, max_tokens)
                    .await?
            }
            CompressionStrategy::Truncate { max_length } => {
                self.compress_truncate(message, max_length)?
            }
            CompressionStrategy::Hybrid => {
                // Try summarization first, fall back to truncation if needed
                match self.compress_summarize(message, max_tokens).await {
                    Ok(compressed) if compressed.compressed_tokens <= max_tokens => compressed,
                    _ => self.compress_truncate(message, max_tokens * 4)?, // Rough char estimate
                }
            }
        };

        Ok(compressed)
    }
}

impl LlmMessageCompressor {
    /// Compress by summarizing with LLM.
    async fn compress_summarize(
        &self,
        message: &Message,
        max_tokens: usize,
    ) -> Result<CompressedMessage> {
        let original_tokens = self.token_counter.estimate_message_tokens(message);
        let text = message.content.extract_text();

        let summary_prompt = format!(
            r"Summarize the following content concisely, preserving all critical information, facts, and key points. Target length: approximately {max_tokens} tokens.

Content to summarize:
{text}

Summary:"
        );

        let request = CompletionRequest {
            messages: Arc::new(vec![Message {
                role: MessageRole::User,
                content: MessageContent::text(summary_prompt),
                metadata: None,
            }]),
            max_tokens: Some(self.max_summary_tokens.min(max_tokens) as u32),
            temperature: Some(0.2),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = self.provider.complete(request).await?;
        let summary_text = response.content.extract_text();

        let mut metadata = message
            .metadata
            .clone()
            .unwrap_or_else(|| serde_json::json!({}));
        let metadata_obj = metadata
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("Metadata is not an object"))?;
        metadata_obj.insert(
            "compressed".to_string(),
            serde_json::json!({
                "original_tokens": original_tokens,
                "method": "summarize"
            }),
        );
        let compressed_message = Message {
            role: message.role.clone(),
            content: MessageContent::text(summary_text),
            metadata: Some(metadata),
        };

        let compressed_tokens = self
            .token_counter
            .estimate_message_tokens(&compressed_message);

        Ok(CompressedMessage::new(
            compressed_message,
            original_tokens,
            compressed_tokens,
        ))
    }

    /// Compress by extracting key points only.
    async fn compress_extract_key_points(
        &self,
        message: &Message,
        max_tokens: usize,
    ) -> Result<CompressedMessage> {
        let original_tokens = self.token_counter.estimate_message_tokens(message);
        let text = message.content.extract_text();

        let extraction_prompt = format!(
            r"Extract only the key points, facts, and essential information from the following content. Format as a bulleted list. Target: {max_tokens} tokens maximum.

Content:
{text}

Key points:"
        );

        let request = CompletionRequest {
            messages: Arc::new(vec![Message {
                role: MessageRole::User,
                content: MessageContent::text(extraction_prompt),
                metadata: None,
            }]),
            max_tokens: Some(max_tokens.min(1000) as u32),
            temperature: Some(0.1),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = self.provider.complete(request).await?;
        let extracted_text = response.content.extract_text();

        let mut metadata = message
            .metadata
            .clone()
            .unwrap_or_else(|| serde_json::json!({}));
        let metadata_obj = metadata
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("Metadata is not an object"))?;
        metadata_obj.insert(
            "compressed".to_string(),
            serde_json::json!({
                "original_tokens": original_tokens,
                "method": "extract_key_points"
            }),
        );
        let compressed_message = Message {
            role: message.role.clone(),
            content: MessageContent::text(extracted_text),
            metadata: Some(metadata),
        };

        let compressed_tokens = self
            .token_counter
            .estimate_message_tokens(&compressed_message);

        Ok(CompressedMessage::new(
            compressed_message,
            original_tokens,
            compressed_tokens,
        ))
    }

    /// Compress by truncating to maximum length.
    fn compress_truncate(&self, message: &Message, max_length: usize) -> Result<CompressedMessage> {
        let original_tokens = self.token_counter.estimate_message_tokens(message);
        let text = message.content.extract_text();

        let truncated_text = if text.len() > max_length {
            format!(
                "{}... [truncated]",
                caliche_contracts::text::truncate_utf8(&text, max_length)
            )
        } else {
            text
        };

        let mut metadata = message
            .metadata
            .clone()
            .unwrap_or_else(|| serde_json::json!({}));
        let metadata_obj = metadata
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("Metadata is not an object"))?;
        metadata_obj.insert(
            "compressed".to_string(),
            serde_json::json!({
                "original_tokens": original_tokens,
                "method": "truncate"
            }),
        );
        let compressed_message = Message {
            role: message.role.clone(),
            content: MessageContent::text(truncated_text),
            metadata: Some(metadata),
        };

        let compressed_tokens = self
            .token_counter
            .estimate_message_tokens(&compressed_message);

        Ok(CompressedMessage::new(
            compressed_message,
            original_tokens,
            compressed_tokens,
        ))
    }
}

/// Tool output compressor that automatically compresses large tool outputs.
pub struct ToolOutputCompressor {
    compressor: Arc<dyn MessageCompressor>,
    token_counter: Arc<dyn TokenCounter>,
    threshold_tokens: usize,
}

impl ToolOutputCompressor {
    /// Create a new tool output compressor.
    pub fn new(
        compressor: Arc<dyn MessageCompressor>,
        token_counter: Arc<dyn TokenCounter>,
        threshold_tokens: usize,
    ) -> Self {
        Self {
            compressor,
            token_counter,
            threshold_tokens,
        }
    }

    /// Compress tool output if it exceeds threshold.
    pub async fn compress_if_needed(
        &self,
        message: &Message,
        max_tokens: usize,
    ) -> Result<Message> {
        let token_count = self.token_counter.estimate_message_tokens(message);

        if token_count > self.threshold_tokens {
            let compressed = self.compressor.compress(message, max_tokens).await?;
            Ok(compressed.message)
        } else {
            Ok(message.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_message() {
        let msg = Message::new(MessageRole::User, "test".to_string());
        let compressed = CompressedMessage::new(msg, 100, 50);
        assert_eq!(compressed.compression_ratio, 0.5);
        assert!(compressed.lossy);
    }
}
