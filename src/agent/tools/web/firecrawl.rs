//! Firecrawl v2 search and scrape.
//!
//! Search can run on the keyless tier. `web_fetch` auto mode constructs this
//! client only when `FIRECRAWL_API_KEY` is set.

use super::error::WebError;
use super::transport::{WebHttpResponse, WebHttpTransport};
use super::types::{
    PROVIDER_FIRECRAWL, WebFetchOutcome, WebProviderConfig, WebSearchHit, WebSearchOutcome,
    firecrawl_endpoint,
};
use serde_json::{Value, json};
use std::sync::Arc;

pub struct FirecrawlClient {
    transport: Arc<dyn WebHttpTransport>,
    config: Arc<WebProviderConfig>,
}

impl FirecrawlClient {
    #[must_use]
    pub fn new(transport: Arc<dyn WebHttpTransport>, config: Arc<WebProviderConfig>) -> Self {
        Self { transport, config }
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<WebSearchOutcome, WebError> {
        let url = firecrawl_endpoint(&self.config.firecrawl_api_url, "search");
        let body = json!({
            "query": query,
            "limit": max_results,
        });
        let response = self.post(&url, &body).await?;
        ensure_success(PROVIDER_FIRECRAWL, &response)?;
        let results = parse_search_hits(&response.body);
        Ok(WebSearchOutcome {
            provider: PROVIDER_FIRECRAWL.to_owned(),
            query: query.to_owned(),
            answer: None,
            results,
        })
    }

    pub async fn scrape(&self, page_url: &str) -> Result<WebFetchOutcome, WebError> {
        crate::agent::tools::common::http::validate_http_url(page_url)
            .map_err(|err| WebError::BlockedUrl(err.to_string()))?;
        let url = firecrawl_endpoint(&self.config.firecrawl_api_url, "scrape");
        let body = json!({
            "url": page_url,
            "formats": ["markdown"],
            "onlyMainContent": true,
        });
        let response = self.post(&url, &body).await?;
        ensure_success(PROVIDER_FIRECRAWL, &response)?;
        parse_scrape(&response.body, page_url)
    }

    async fn post(&self, url: &str, body: &Value) -> Result<WebHttpResponse, WebError> {
        if let Some(key) = self.config.firecrawl_api_key.as_deref() {
            let bearer = format!("Bearer {key}");
            return self
                .transport
                .post_json(url, &[("Authorization", bearer.as_str())], body)
                .await;
        }
        self.transport.post_json(url, &[], body).await
    }
}

fn ensure_success(provider: &str, response: &WebHttpResponse) -> Result<(), WebError> {
    if (200..300).contains(&response.status) {
        if response.body.get("success").and_then(Value::as_bool) == Some(false) {
            let detail = response
                .body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("request failed");
            return Err(WebError::invalid(provider, detail));
        }
        return Ok(());
    }
    let detail = response
        .body
        .get("error")
        .and_then(Value::as_str)
        .map_or_else(|| response.body.to_string(), ToOwned::to_owned);
    Err(WebError::http(provider, response.status, detail))
}

fn parse_search_hits(body: &Value) -> Vec<WebSearchHit> {
    let data = body.get("data").unwrap_or(body);
    let entries = data
        .get("web")
        .and_then(Value::as_array)
        .or_else(|| data.as_array())
        .or_else(|| body.get("results").and_then(Value::as_array));
    let Some(entries) = entries else {
        return Vec::new();
    };
    entries.iter().filter_map(hit_from_value).collect()
}

fn hit_from_value(entry: &Value) -> Option<WebSearchHit> {
    let url = entry.get("url").and_then(Value::as_str)?;
    if url.is_empty() {
        return None;
    }
    let title = entry
        .get("title")
        .and_then(Value::as_str)
        .filter(|title| !title.is_empty())
        .unwrap_or(url);
    let snippet = first_string(entry, &["description", "snippet", "markdown", "content"])
        .unwrap_or("")
        .trim();
    Some(WebSearchHit {
        title: title.to_owned(),
        url: url.to_owned(),
        snippet: snippet.to_owned(),
        score: entry.get("score").and_then(Value::as_f64),
    })
}

fn parse_scrape(body: &Value, fallback_url: &str) -> Result<WebFetchOutcome, WebError> {
    let data = body.get("data").unwrap_or(body);
    let markdown = first_string(data, &["markdown", "content"])
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| WebError::invalid(PROVIDER_FIRECRAWL, "empty markdown"))?;
    let metadata = data.get("metadata");
    let title = metadata
        .and_then(|value| value.get("title"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned);
    let url = metadata
        .and_then(|value| first_string(value, &["sourceURL", "url"]))
        .unwrap_or(fallback_url)
        .to_owned();
    Ok(WebFetchOutcome {
        provider: PROVIDER_FIRECRAWL.to_owned(),
        url,
        title,
        markdown: markdown.to_owned(),
    })
}

fn first_string<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
}
