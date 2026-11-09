//! Exa semantic / neural web search.

use super::error::WebError;
use super::transport::{WebHttpResponse, WebHttpTransport};
use super::types::{PROVIDER_EXA, WebProviderConfig, WebSearchHit, WebSearchOutcome};
use serde_json::{Value, json};
use std::sync::Arc;

const EXA_SEARCH_URL: &str = "https://api.exa.ai/search";

pub struct ExaSearch {
    transport: Arc<dyn WebHttpTransport>,
    config: Arc<WebProviderConfig>,
}

impl ExaSearch {
    #[must_use]
    pub fn new(transport: Arc<dyn WebHttpTransport>, config: Arc<WebProviderConfig>) -> Self {
        Self { transport, config }
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<WebSearchOutcome, WebError> {
        let api_key = self
            .config
            .exa_api_key
            .as_deref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_EXA.to_owned()))?;
        let body = json!({
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "contents": { "text": { "maxCharacters": 1200 } },
        });
        let response = self
            .transport
            .post_json(EXA_SEARCH_URL, &[("x-api-key", api_key)], &body)
            .await?;
        ensure_success(&response)?;
        Ok(WebSearchOutcome {
            provider: PROVIDER_EXA.to_owned(),
            query: query.to_owned(),
            answer: None,
            results: parse_hits(&response.body),
        })
    }
}

fn ensure_success(response: &WebHttpResponse) -> Result<(), WebError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let detail = response
        .body
        .get("error")
        .and_then(Value::as_str)
        .map_or_else(|| response.body.to_string(), ToOwned::to_owned);
    Err(WebError::http(PROVIDER_EXA, response.status, detail))
}

fn parse_hits(body: &Value) -> Vec<WebSearchHit> {
    let Some(entries) = body.get("results").and_then(Value::as_array) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let url = entry.get("url").and_then(Value::as_str)?;
            if url.is_empty() {
                return None;
            }
            let title = entry
                .get("title")
                .and_then(Value::as_str)
                .filter(|title| !title.is_empty())
                .unwrap_or(url);
            Some(WebSearchHit {
                title: title.to_owned(),
                url: url.to_owned(),
                snippet: entry
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or("")
                    .to_owned(),
                score: entry.get("score").and_then(Value::as_f64),
            })
        })
        .collect()
}
