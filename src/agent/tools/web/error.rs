//! Errors for web search and fetch backends.

use thiserror::Error;

/// Recoverable provider or transport failure.
#[derive(Debug, Error)]
pub enum WebError {
    /// Request URL failed SSRF or scheme checks.
    #[error("blocked url: {0}")]
    BlockedUrl(String),
    /// Remote HTTP status was not success.
    #[error("{provider} http {status}: {body}")]
    Http {
        provider: String,
        status: u16,
        body: String,
    },
    /// Response JSON could not be used.
    #[error("{provider} returned invalid payload: {detail}")]
    InvalidPayload { provider: String, detail: String },
    /// Every selected backend failed.
    #[error("no web provider succeeded: {0}")]
    AllFailed(String),
    /// Named provider is not configured.
    #[error("web provider '{0}' is not configured")]
    ProviderUnavailable(String),
    /// Transport or policy failure.
    #[error("{0}")]
    Message(String),
}

impl WebError {
    pub(crate) fn message(detail: impl Into<String>) -> Self {
        Self::Message(detail.into())
    }

    pub(crate) fn http(provider: &str, status: u16, body: impl Into<String>) -> Self {
        Self::Http {
            provider: provider.to_owned(),
            status,
            body: truncate_body(body.into()),
        }
    }

    pub(crate) fn invalid(provider: &str, detail: impl Into<String>) -> Self {
        Self::InvalidPayload {
            provider: provider.to_owned(),
            detail: detail.into(),
        }
    }
}

fn truncate_body(body: String) -> String {
    const MAX: usize = 500;
    if body.len() <= MAX {
        return body;
    }
    let end = body.floor_char_boundary(MAX);
    format!("{}…", &body[..end])
}

#[cfg(test)]
mod tests {
    use super::truncate_body;

    #[test]
    fn truncate_body_keeps_char_boundaries() {
        let mut body = "a".repeat(499);
        body.push('é');
        body.push_str(&"b".repeat(20));
        let truncated = truncate_body(body);
        assert!(truncated.ends_with('…'), "{truncated}");
        assert!(truncated.is_char_boundary(truncated.len()));
        assert!(!truncated.contains('é'));
    }
}
