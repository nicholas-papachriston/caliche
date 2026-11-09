//! First-party `web_search` tool.

use super::error::WebError;
use super::provider::WebSearchRouter;
use super::transport::{ReqwestWebTransport, WebHttpTransport};
use super::types::{
    PROVIDER_AUTO, PROVIDER_EXA, PROVIDER_FIRECRAWL, PROVIDER_TAVILY, PROVIDER_TINYFISH,
    WebProviderConfig, clamp_max_results,
};
use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolCapability, ToolMeta, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

const NAME: &str = crate::agent::tools::names::TOOL_WEB_SEARCH;
const DESCRIPTION: &str = "Search the live web and return ranked URLs with snippets. Use for source-first discovery. Prefer this over guessing URLs. Use web_fetch to read a chosen page. Use perplexity_search when you need a cited synthesized answer instead of raw hits. Optional provider: auto (default), tavily, exa, tinyfish, firecrawl. Firecrawl works without an API key.";

pub struct WebSearchTool {
    router: WebSearchRouter,
}

impl std::fmt::Debug for WebSearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSearchTool").finish_non_exhaustive()
    }
}

impl WebSearchTool {
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_config(
            Arc::new(WebProviderConfig::from_env()),
            Arc::new(ReqwestWebTransport::new()),
        )
    }

    #[must_use]
    pub fn from_config(
        config: Arc<WebProviderConfig>,
        transport: Arc<dyn WebHttpTransport>,
    ) -> Self {
        Self {
            router: WebSearchRouter::from_config(config, transport),
        }
    }
}

impl ToolMeta for WebSearchTool {
    fn name(&self) -> &str {
        NAME
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn is_readonly(&self) -> bool {
        true
    }

    fn capabilities(&self) -> Vec<ToolCapability> {
        vec![ToolCapability::ReadOnly]
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Be specific."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of hits to return (1-20, default 5)."
                },
                "provider": {
                    "type": "string",
                    "enum": [
                        PROVIDER_AUTO,
                        PROVIDER_TAVILY,
                        PROVIDER_EXA,
                        PROVIDER_TINYFISH,
                        PROVIDER_FIRECRAWL,
                    ],
                    "description": "Backend. auto uses Tavily, then Exa, then TinyFish, then Firecrawl keyless."
                }
            },
            "required": ["query"]
        })
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let query = match params::extract_string(&arguments, "query", None) {
            Ok(value) => value,
            Err(err) => return ToolResult::error(err),
        };
        if query.trim().is_empty() {
            return ToolResult::error("query must not be empty");
        }
        let raw = params::extract_number_or(
            &arguments,
            "max_results",
            super::types::DEFAULT_MAX_RESULTS as u64,
        );
        let max_results =
            clamp_max_results(usize::try_from(raw).unwrap_or(super::types::DEFAULT_MAX_RESULTS));
        let provider = params::extract_string_or(&arguments, "provider", PROVIDER_AUTO);
        match self
            .router
            .search(query.trim(), max_results, provider.trim())
            .await
        {
            Ok(outcome) => json_result(&outcome),
            Err(err) => ToolResult::error(err),
        }
    }
}

pub(crate) fn json_result<T: serde::Serialize>(value: &T) -> ToolResult {
    match serde_json::to_value(value) {
        Ok(payload) => {
            let content =
                serde_json::to_string_pretty(&payload).unwrap_or_else(|_| payload.to_string());
            ToolResult::success_with_payload(content, payload)
        }
        Err(err) => ToolResult::error(WebError::message(err.to_string())),
    }
}
