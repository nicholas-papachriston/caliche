//! TinyFish search and fetch client.

use super::error::WebError;
use super::transport::{WebHttpResponse, WebHttpTransport};
use super::types::{
    PROVIDER_TINYFISH, WebFetchOutcome, WebProviderConfig, WebSearchHit, WebSearchOutcome,
};
use serde_json::{Value, json};
use std::sync::Arc;

pub struct TinyFishClient {
    transport: Arc<dyn WebHttpTransport>,
    config: Arc<WebProviderConfig>,
}

impl TinyFishClient {
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
            .tinyfish_api_key
            .as_deref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_TINYFISH.to_owned()))?;

        let base_url = self.config.tinyfish_search_api_url.trim_end_matches('/');
        let mut parsed_url = url::Url::parse(base_url)
            .map_err(|err| WebError::invalid(PROVIDER_TINYFISH, err.to_string()))?;
        parsed_url.query_pairs_mut().append_pair("query", query);
        let url = parsed_url.to_string();

        let response = self
            .transport
            .get_json(&url, &[("X-API-Key", api_key)])
            .await?;
        ensure_success(&response)?;

        let mut results = parse_search_hits(&response.body);
        if results.len() > max_results {
            results.truncate(max_results);
        }

        Ok(WebSearchOutcome {
            provider: PROVIDER_TINYFISH.to_owned(),
            query: query.to_owned(),
            answer: None,
            results,
        })
    }

    pub async fn fetch(&self, page_url: &str) -> Result<WebFetchOutcome, WebError> {
        crate::agent::tools::common::http::validate_http_url(page_url)
            .map_err(|err| WebError::BlockedUrl(err.to_string()))?;

        let api_key = self
            .config
            .tinyfish_api_key
            .as_deref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_TINYFISH.to_owned()))?;

        let base_url = self.config.tinyfish_fetch_api_url.trim_end_matches('/');
        let body = json!({
            "urls": [page_url],
            "format": "markdown",
        });

        let response = self
            .transport
            .post_json(base_url, &[("X-API-Key", api_key)], &body)
            .await?;
        ensure_success(&response)?;

        parse_fetch(&response.body, page_url)
    }
}

fn ensure_success(response: &WebHttpResponse) -> Result<(), WebError> {
    if (200..300).contains(&response.status) {
        return Ok(());
    }

    let detail = if let Some(error_obj) = response.body.get("error") {
        if let Some(msg) = error_obj.get("message").and_then(Value::as_str) {
            msg.to_owned()
        } else if let Some(err_str) = error_obj.as_str() {
            err_str.to_owned()
        } else {
            error_obj.to_string()
        }
    } else if let Some(detail_val) = response.body.get("detail").and_then(Value::as_str) {
        detail_val.to_owned()
    } else {
        response.body.to_string()
    };

    Err(WebError::http(PROVIDER_TINYFISH, response.status, detail))
}

fn parse_search_hits(body: &Value) -> Vec<WebSearchHit> {
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
            let snippet = entry
                .get("snippet")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("");

            Some(WebSearchHit {
                title: title.to_owned(),
                url: url.to_owned(),
                snippet: snippet.to_owned(),
                score: None,
            })
        })
        .collect()
}

fn parse_fetch(body: &Value, original_url: &str) -> Result<WebFetchOutcome, WebError> {
    if let Some(results) = body.get("results").and_then(Value::as_array) {
        if let Some(entry) = results.first() {
            let markdown = entry
                .get("text")
                .and_then(Value::as_str)
                .map(str::trim)
                .unwrap_or("");

            if !markdown.is_empty() {
                let title = entry
                    .get("title")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(ToOwned::to_owned);

                let url = entry
                    .get("final_url")
                    .and_then(Value::as_str)
                    .or_else(|| entry.get("url").and_then(Value::as_str))
                    .unwrap_or(original_url)
                    .to_owned();

                return Ok(WebFetchOutcome {
                    provider: PROVIDER_TINYFISH.to_owned(),
                    url,
                    title,
                    markdown: markdown.to_owned(),
                });
            }
        }
    }

    if let Some(errors) = body.get("errors").and_then(Value::as_array) {
        if let Some(err_entry) = errors.first() {
            let error_code = err_entry
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("fetch failed");
            let detail = if let Some(status) = err_entry.get("status").and_then(Value::as_u64) {
                format!("{error_code} (status {status})")
            } else {
                error_code.to_owned()
            };
            return Err(WebError::invalid(PROVIDER_TINYFISH, detail));
        }
    }

    Err(WebError::invalid(
        PROVIDER_TINYFISH,
        "empty or missing markdown content",
    ))
}
