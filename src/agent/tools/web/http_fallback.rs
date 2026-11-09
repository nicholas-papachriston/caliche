//! Direct HTTP fetch fallback when Firecrawl is unavailable.

use super::error::WebError;
use super::transport::WebHttpTransport;
use super::types::{PROVIDER_HTTP, WebFetchOutcome};
use std::sync::Arc;

const DEFAULT_MAX_CHARS: usize = 50_000;

pub struct HttpFetch {
    transport: Arc<dyn WebHttpTransport>,
}

impl HttpFetch {
    #[must_use]
    pub fn new(transport: Arc<dyn WebHttpTransport>) -> Self {
        Self { transport }
    }

    pub async fn fetch(&self, url: &str) -> Result<WebFetchOutcome, WebError> {
        crate::agent::tools::common::http::validate_http_url(url)
            .map_err(|err| WebError::BlockedUrl(err.to_string()))?;
        let html = self.transport.get_text(url).await?;
        let cleaned = crate::agent::tools::common::text::clean_html_text(&html);
        let markdown = truncate_page_text(cleaned, DEFAULT_MAX_CHARS);
        if markdown.trim().is_empty() {
            return Err(WebError::invalid(PROVIDER_HTTP, "empty page text"));
        }
        Ok(WebFetchOutcome {
            provider: PROVIDER_HTTP.to_owned(),
            url: url.to_owned(),
            title: None,
            markdown,
        })
    }
}

fn truncate_page_text(cleaned: String, max_chars: usize) -> String {
    if cleaned.len() <= max_chars {
        return cleaned;
    }
    let end = cleaned.floor_char_boundary(max_chars);
    cleaned[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::truncate_page_text;

    #[test]
    fn truncate_page_text_keeps_utf8_char_boundary() {
        let text = "é".repeat(8);
        let truncated = truncate_page_text(text, 5);
        assert!(truncated.is_char_boundary(truncated.len()));
        assert!(truncated.chars().all(|ch| ch == 'é'));
        assert!(truncated.len() <= 5);
        assert!(!truncated.is_empty());
    }
}
