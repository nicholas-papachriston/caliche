use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use async_trait::async_trait;
use futures_util::Stream;

/// How a provider reports token usage across the lifetime of a single
/// streaming response. Used by the agent streaming loop to decide whether to
///
/// replace the running `final_usage` with each `Metadata` event (cumulative)
/// or to sum deltas from multiple events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageMergePolicy {
    /// Each `Metadata.usage` event is a *delta* covering only the tokens
    /// emitted since the previous event. The agent should accumulate them
    /// with `prev + new`.
    Delta,
    /// Each `Metadata.usage` event is a *cumulative* snapshot for the entire
    /// stream so far. The agent should replace the running total with the
    /// latest value (taking max/last) rather than summing.
    Cumulative,
}

#[macro_export]
macro_rules! delegate_provider_capabilities {
    ($inner:ident) => {
        fn supports_native_tools(&self) -> bool {
            self.$inner.supports_native_tools()
        }

        fn supports_text_based_tools(&self) -> bool {
            self.$inner.supports_text_based_tools()
        }

        fn supports_streaming(&self) -> bool {
            self.$inner.supports_streaming()
        }

        fn supports_vision(&self) -> bool {
            self.$inner.supports_vision()
        }

        fn supports_audio_input(&self) -> bool {
            self.$inner.supports_audio_input()
        }

        fn supports_document_input(&self) -> bool {
            self.$inner.supports_document_input()
        }

        fn supported_media_types(&self) -> Vec<&str> {
            self.$inner.supported_media_types()
        }

        fn context_window(&self) -> Option<usize> {
            self.$inner.context_window()
        }

        fn model_load_probe(
            &self,
        ) -> Option<$crate::llm::provider_utils::first_token::ModelLoadProbe> {
            self.$inner.model_load_probe()
        }

        fn streaming_usage_policy(&self) -> $crate::llm::provider::UsageMergePolicy {
            self.$inner.streaming_usage_policy()
        }
    };
}

#[async_trait]
pub trait Provider: Send + Sync + std::fmt::Debug {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse>;

    /// Stream completion tokens from the LLM
    /// Returns a stream of structured events
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>>;

    fn name(&self) -> &str;

    /// Check if this provider supports native function/tool calling
    fn supports_native_tools(&self) -> bool {
        false // Default to false for backward compatibility
    }

    /// Check if this provider supports text-based tool calling (JSON emitted
    /// inside `content` rather than native `tool_calls`). Default is `false`;
    /// override for providers that rely on in-band tool specs (e.g. Perplexity).
    fn supports_text_based_tools(&self) -> bool {
        false
    }

    /// Check if this provider supports streaming
    fn supports_streaming(&self) -> bool {
        true // Default to true, providers can override if needed
    }

    /// Check if this provider supports vision/multimodal inputs (images, videos)
    fn supports_vision(&self) -> bool {
        false // Default to false, providers can override if they support vision
    }

    /// Check if this provider supports native audio input.
    fn supports_audio_input(&self) -> bool {
        false
    }

    /// Check if this provider supports native document input.
    fn supports_document_input(&self) -> bool {
        false
    }

    /// Return media types supported by the provider for multimodal input.
    fn supported_media_types(&self) -> Vec<&str> {
        vec![]
    }

    /// Return the context window (max input tokens) for the active model, if known.
    /// Providers that can fetch this from their API should override this method.
    /// Returns None when the value is not available; callers fall back to hardcoded tables.
    fn context_window(&self) -> Option<usize> {
        None
    }

    /// How to ask the provider's server about the active model (whether it
    /// is loaded, its context window), for local servers that report it.
    fn model_load_probe(&self) -> Option<crate::llm::provider_utils::first_token::ModelLoadProbe> {
        None
    }

    /// Describe how the provider reports streaming token usage in
    /// `StreamEvent::Metadata` events. Default is
    /// [`UsageMergePolicy::Cumulative`] to match the OpenAI-compatible
    /// convention (one final usage snapshot at end of stream). Override to
    /// [`UsageMergePolicy::Delta`] for providers that emit per-chunk deltas.
    fn streaming_usage_policy(&self) -> UsageMergePolicy {
        UsageMergePolicy::Cumulative
    }
}
