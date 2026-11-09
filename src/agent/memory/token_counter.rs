use std::sync::Arc;

use crate::messages::{Message, MessageContent};

/// Estimated token usage for a collection of messages.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenCountEstimate {
    /// Estimated token count for the prompt portion.
    pub prompt_tokens: usize,
    /// Estimated token count for the completion portion.
    pub completion_tokens: usize,
}

impl TokenCountEstimate {
    /// Total estimated tokens.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Combine two estimates.
    #[must_use]
    pub const fn combine(self, other: Self) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_add(other.completion_tokens),
        }
    }
}

/// Trait for estimating token counts for messages and text fragments.
///
/// When using the `token-tiktoken` feature, the counter is model-specific: counts are accurate only
/// for the model encoding it was built with. Using the same counter for a different model may yield
/// inaccurate counts.
pub trait TokenCounter: Send + Sync {
    /// Returns true if this counter uses an approximate (character-ratio) method.
    /// When true, counts may be inaccurate; enable the `token-tiktoken` feature for precise counts.
    fn is_approximate(&self) -> bool {
        false
    }

    /// Estimate token count for a single message.
    fn estimate_message_tokens(&self, message: &Message) -> usize {
        const MESSAGE_OVERHEAD: usize = 4;
        // Use as_text() for zero-copy when possible
        let text = message.content.as_text();
        let token_count = if let Some(text_ref) = text {
            self.estimate_text_tokens(text_ref)
        } else {
            // Only extract and allocate if not simple text
            let extracted = extract_message_text(message);
            self.estimate_text_tokens(&extracted)
        };
        MESSAGE_OVERHEAD.saturating_add(token_count)
    }

    /// Estimate token count for multiple messages.
    fn estimate_messages_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|msg| self.estimate_message_tokens(msg))
            .sum()
    }

    /// Estimate token count for arbitrary text.
    fn estimate_text_tokens(&self, text: &str) -> usize;
}

/// Helper function to estimate tokens from an iterator without collecting into Vec.
/// This avoids intermediate allocations when counting tokens from `VecDeque` or other collections.
pub fn estimate_messages_tokens_iter<'a>(
    counter: &dyn TokenCounter,
    messages: impl Iterator<Item = &'a Message>,
) -> usize {
    messages
        .map(|msg| counter.estimate_message_tokens(msg))
        .sum()
}

/// Approximate token counter using a configurable characters-per-token ratio.
///
/// Best-effort only: the fixed ratio is inaccurate for code, JSON, and non-English text.
/// Enable the `token-tiktoken` feature for precise, model-specific counts.
#[derive(Debug, Clone)]
pub struct ApproximateTokenCounter {
    /// Average number of characters per token.
    pub characters_per_token: f32,
    /// Minimum number of tokens to attribute to any message.
    pub minimum_tokens: usize,
}

impl Default for ApproximateTokenCounter {
    fn default() -> Self {
        Self {
            characters_per_token: 3.8,
            minimum_tokens: 1,
        }
    }
}

impl TokenCounter for ApproximateTokenCounter {
    fn is_approximate(&self) -> bool {
        true
    }

    fn estimate_message_tokens(&self, message: &Message) -> usize {
        // Use as_text() for zero-copy when possible
        let token_count = if let Some(text) = message.content.as_text() {
            self.estimate_text_tokens(text)
        } else {
            let text = extract_message_text(message);
            self.estimate_text_tokens(&text)
        };
        let total = 4 + token_count;
        total.max(self.minimum_tokens)
    }

    fn estimate_text_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let raw = (text.len() as f32 / self.characters_per_token).ceil() as usize;
        raw.max(self.minimum_tokens)
    }
}

/// Factory for building token counters with optional precise implementations.
#[derive(Debug, Clone)]
pub struct TokenCounterFactory {
    approximate: ApproximateTokenCounter,
    #[cfg(feature = "token-tiktoken")]
    exact: Option<Arc<TiktokenCounter>>,
}

impl TokenCounterFactory {
    /// Create a new factory with default approximation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            approximate: ApproximateTokenCounter::default(),
            #[cfg(feature = "token-tiktoken")]
            exact: None,
        }
    }

    /// Create a factory that attempts to use a specific model encoding (when `token-tiktoken` is enabled).
    /// The returned counter is model-specific: use it only with the same model name for accurate counts.
    pub fn with_model(model: Option<String>) -> Self {
        #[allow(unused_mut)]
        let mut factory = Self::new();

        #[cfg(feature = "token-tiktoken")]
        {
            if let Some(model_name) = model {
                factory.exact = TiktokenCounter::for_model(&model_name).ok().map(Arc::new);
            }
        }

        #[cfg(not(feature = "token-tiktoken"))]
        let _ = model;

        factory
    }

    /// Return the best available token counter.
    /// When the approximate counter is used (tiktoken unavailable or disabled), a tracing warning is emitted.
    pub fn build(&self) -> Arc<dyn TokenCounter> {
        #[cfg(feature = "token-tiktoken")]
        if let Some(counter) = &self.exact {
            return counter.clone();
        }

        tracing::warn!(
            "Using approximate token counter; tiktoken unavailable. Counts may be inaccurate. \
             Enable the `token-tiktoken` feature for precise counts."
        );
        Arc::new(self.approximate.clone())
    }

    /// Access the underlying approximate counter for fine-tuning.
    pub const fn approximate_mut(&mut self) -> &mut ApproximateTokenCounter {
        &mut self.approximate
    }
}

impl Default for TokenCounterFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "token-tiktoken")]
use anyhow::Result;

#[cfg(feature = "token-tiktoken")]
use std::sync::RwLock;

#[cfg(feature = "token-tiktoken")]
use tiktoken_rs::{
    CoreBPE, cl100k_base, o200k_base, tokenizer::Tokenizer, tokenizer::get_tokenizer,
};

/// Precise token counter backed by the `tiktoken-rs` crate.
/// Model-specific: counts are accurate only for the encoding used at construction (e.g. the model name passed to `for_model`).
#[cfg(feature = "token-tiktoken")]
pub struct TiktokenCounter {
    bpe: Arc<RwLock<CoreBPE>>,
}

#[cfg(feature = "token-tiktoken")]
impl std::fmt::Debug for TiktokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiktokenCounter")
            .field("bpe", &"<CoreBPE>")
            .finish()
    }
}

#[cfg(feature = "token-tiktoken")]
impl TiktokenCounter {
    /// Construct a counter using the encoding for the provided model.
    /// Models that use `o200k` get `o200k_base`; every other model, including the
    /// older `p50k`/`r50k` models and unknown names, gets `cl100k_base`. Only these
    /// two tables are compiled in, so counts for `p50k`/`r50k` models are approximate.
    /// Use this counter only with the same model (or compatible encoding) for accurate counts.
    pub fn for_model(model: &str) -> Result<Self> {
        let bpe = match get_tokenizer(model) {
            Some(Tokenizer::O200kBase | Tokenizer::O200kHarmony) => o200k_base()?,
            Some(_) => cl100k_base()?,
            None => {
                tracing::debug!("No tiktoken encoding for model {model}, using cl100k_base");
                cl100k_base()?
            }
        };
        Ok(Self {
            bpe: Arc::new(RwLock::new(bpe)),
        })
    }

    /// Encode raw text using the underlying tokenizer.
    fn encode(&self, text: &str) -> usize {
        self.bpe
            .read()
            .map_err(|e| anyhow::anyhow!("RwLock poisoned: {e}"))
            .map(|guard| guard.encode_with_special_tokens(text).len())
            .unwrap_or(0)
    }
}

#[cfg(feature = "token-tiktoken")]
impl TokenCounter for TiktokenCounter {
    fn estimate_message_tokens(&self, message: &Message) -> usize {
        const MESSAGE_OVERHEAD: usize = 4;
        // Use as_text() for zero-copy when possible
        let token_count = if let Some(text) = message.content.as_text() {
            self.estimate_text_tokens(text)
        } else {
            let text = extract_message_text(message);
            self.estimate_text_tokens(&text)
        };
        MESSAGE_OVERHEAD.saturating_add(token_count)
    }

    fn estimate_text_tokens(&self, text: &str) -> usize {
        self.encode(text)
    }
}

fn extract_message_text(message: &Message) -> String {
    match &message.content {
        MessageContent::Text(text) => (**text).clone(),
        MessageContent::Multimodal(parts) => parts
            .iter()
            .filter_map(|part| match part {
                crate::messages::ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        MessageContent::Structured {
            text,
            tool_calls,
            tool_call_id,
        } => {
            let mut buffer = String::new();

            if let Some(text) = text {
                buffer.push_str(text.as_str());
            }

            if let Some(tool_call_id) = tool_call_id {
                if !buffer.is_empty() {
                    buffer.push('\n');
                }
                buffer.push_str("[tool_call_id: ");
                buffer.push_str(tool_call_id);
                buffer.push(']');
            }

            if let Some(tool_calls) = tool_calls {
                for call in tool_calls {
                    if !buffer.is_empty() {
                        buffer.push('\n');
                    }
                    if let Ok(json) = serde_json::to_string(call) {
                        buffer.push_str(&json);
                    }
                }
            }

            buffer
        }
    }
}
