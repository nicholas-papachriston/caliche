//! Stand-in provider used when a process may listen before completions can run.
//!
//! Construction records the selected provider. Completions fail with that
//! provider's credential or egress error. This does not select a different
//! provider. A missing key is not authentication or an isolation boundary.
//! An egress-blocked provider URL is not authentication or a reason to
//! refuse listen.

use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use async_trait::async_trait;
use futures_util::stream;
use std::pin::Pin;

/// Provider that keeps its configured name and refuses completions.
#[derive(Debug)]
pub struct UnconfiguredProvider {
    provider_name: String,
    detail: String,
}

impl UnconfiguredProvider {
    /// `provider_name` is the selected provider's display name (`openrouter`).
    /// `detail` is the credential lookup or builder error.
    #[must_use]
    pub fn new(provider_name: impl Into<String>, detail: impl Into<String>) -> Self {
        let provider_name = provider_name.into();
        let detail = detail.into();
        Self {
            detail: completion_detail(&provider_name, &detail),
            provider_name,
        }
    }

    fn failure(&self) -> anyhow::Error {
        anyhow::anyhow!("{}", self.detail)
    }
}

/// True when an error means the selected provider has no usable model key.
///
/// This is a credential absence check. It is not an authentication result.
#[must_use]
pub fn is_missing_provider_credentials(err: &anyhow::Error) -> bool {
    is_missing_provider_credentials_message(&err.to_string())
}

fn is_missing_provider_credentials_message(message: &str) -> bool {
    if message.contains("No credentials found")
        || message.contains("No credential provider in chain")
        || message.contains("No keychain credentials")
    {
        return true;
    }
    let lower = message.to_ascii_lowercase();
    lower.contains("api_key is required")
        || lower.contains("api key is required")
        || (lower.contains("api_key") && lower.contains("must be set"))
        || (lower.contains("api key") && lower.contains("must be set"))
}

fn completion_detail(provider_name: &str, detail: &str) -> String {
    if detail.contains("No credentials found") {
        return detail.to_owned();
    }
    if is_blocked_provider_url_message(detail) {
        if detail.contains(&format!("provider '{provider_name}'")) {
            return detail.to_owned();
        }
        return format!("provider '{provider_name}': {detail}");
    }
    format!("No credentials found for provider '{provider_name}'. {detail}")
}

/// True when provider construction rejected the selected base URL.
///
/// This is an egress check. It is not an authentication result and not an
/// isolation boundary.
#[must_use]
pub fn is_blocked_provider_url(err: &anyhow::Error) -> bool {
    is_blocked_provider_url_message(&format!("{err:#}"))
}

fn is_blocked_provider_url_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("egress provider url rejected")
        || lower.contains("plugin http url targets a blocked host")
        || lower.contains("egress denied for provider")
}

/// True when `--server` should keep the selected provider and listen anyway.
///
/// Covers a missing model key and an egress-blocked provider URL. Neither is
/// authentication, and neither selects a different provider.
#[must_use]
pub fn should_defer_provider_setup(err: &anyhow::Error) -> bool {
    is_missing_provider_credentials(err) || is_blocked_provider_url(err)
}

#[async_trait]
impl Provider for UnconfiguredProvider {
    async fn complete(&self, _request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        Err(self.failure())
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Pin<Box<dyn futures_util::Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let err = self.failure();
        Box::pin(stream::once(async move { Err(err) }))
    }

    fn name(&self) -> &str {
        &self.provider_name
    }
}

#[cfg(test)]
mod tests {
    use super::{UnconfiguredProvider, is_missing_provider_credentials_message};
    use crate::llm::Provider;
    use crate::messages::{CompletionRequest, Message};
    use futures_util::StreamExt;
    use std::sync::Arc;

    fn request() -> CompletionRequest {
        CompletionRequest::new(Arc::new(vec![Message::user("hello")]))
    }

    #[tokio::test]
    async fn completion_keeps_the_selected_provider() {
        let provider = UnconfiguredProvider::new(
            "openai",
            "No credentials found for provider 'openai'. Set OPENAI_API_KEY or API_KEY environment variable.",
        );
        let err = provider.complete(request()).await.expect_err("missing key");
        let message = err.to_string();
        assert!(message.contains("provider 'openai'"), "{message}");
        assert!(message.contains("No credentials found"), "{message}");
        assert!(
            !message.to_ascii_lowercase().contains("openrouter"),
            "{message}"
        );
        assert!(
            !message.to_ascii_lowercase().contains("fallback"),
            "{message}"
        );
        assert_eq!(provider.name(), "openai");
    }

    #[tokio::test]
    async fn stream_uses_the_same_credential_error() {
        let provider = UnconfiguredProvider::new("anthropic", "api_key is required");
        let mut stream = provider.stream(request());
        let item = stream.next().await.expect("one event");
        let message = item.expect_err("missing key").to_string();
        assert!(
            message.contains("No credentials found for provider 'anthropic'"),
            "{message}"
        );
        assert!(message.contains("api_key is required"), "{message}");
    }

    #[test]
    fn missing_credential_messages_are_recognized() {
        assert!(is_missing_provider_credentials_message(
            "No credentials found for provider 'openrouter'. Set OPENROUTER_API_KEY or API_KEY environment variable."
        ));
        assert!(is_missing_provider_credentials_message(
            "api_key is required or OPENROUTER_API_KEY must be set"
        ));
        assert!(is_missing_provider_credentials_message(
            "No keychain credentials for provider 'openai' (scope: global)"
        ));
        assert!(!is_missing_provider_credentials_message(
            "connection refused"
        ));
        assert!(!is_missing_provider_credentials_message(
            "invalid or missing authentication token"
        ));
    }

    #[tokio::test]
    async fn blocked_url_keeps_the_egress_error() {
        let provider = UnconfiguredProvider::new(
            "ollama",
            "egress provider url rejected: http://localhost:11434/v1\n\nCaused by:\n    plugin http url targets a blocked host: localhost",
        );
        let err = provider.complete(request()).await.expect_err("blocked url");
        let message = err.to_string();
        assert!(message.contains("provider 'ollama'"), "{message}");
        assert!(
            message.contains("egress provider url rejected"),
            "{message}"
        );
        assert!(message.contains("blocked host: localhost"), "{message}");
        assert!(!message.contains("No credentials found"), "{message}");
        assert!(
            !message.to_ascii_lowercase().contains("openrouter"),
            "{message}"
        );
        assert_eq!(provider.name(), "ollama");
    }
}
