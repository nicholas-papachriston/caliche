//! Tavily source-first web search.

use super::error::WebError;
use super::transport::{WebHttpResponse, WebHttpTransport};
use super::types::{PROVIDER_TAVILY, WebProviderConfig, WebSearchHit, WebSearchOutcome};
use serde_json::{Value, json};
use std::sync::Arc;

const TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";

pub struct TavilySearch {
    transport: Arc<dyn WebHttpTransport>,
    config: Arc<WebProviderConfig>,
}

impl TavilySearch {
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
            .tavily_api_key
            .as_deref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_TAVILY.to_owned()))?;
        let bearer = format!("Bearer {api_key}");
        let body = json!({
            "query": query,
            "max_results": max_results,
            "include_answer": true,
            "search_depth": "basic",
        });
        let response = self
            .transport
            .post_json(
                TAVILY_SEARCH_URL,
                &[("Authorization", bearer.as_str())],
                &body,
            )
            .await?;
        ensure_success(&response)?;
        Ok(WebSearchOutcome {
            provider: PROVIDER_TAVILY.to_owned(),
            query: query.to_owned(),
            answer: response
                .body
                .get("answer")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            results: parse_hits(&response.body),
        })
    }
}

fn ensure_success(response: &WebHttpResponse) -> Result<(), WebError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }
    let detail = match response.body.get("detail") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Object(map)) => map
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("request failed")
            .to_owned(),
        _ => response.body.to_string(),
    };
    Err(WebError::http(PROVIDER_TAVILY, response.status, detail))
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
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .unwrap_or("")
                    .to_owned(),
                score: entry.get("score").and_then(Value::as_f64),
            })
        })
        .collect()
}
