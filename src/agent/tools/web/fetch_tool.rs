//! First-party `web_fetch` tool.

use super::provider::WebFetchRouter;
use super::search_tool::json_result;
use super::transport::{ReqwestWebTransport, WebHttpTransport};
use super::types::{
    PROVIDER_AUTO, PROVIDER_FIRECRAWL, PROVIDER_HTTP, PROVIDER_TINYFISH, WebProviderConfig,
};
use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolCapability, ToolMeta, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

const NAME: &str = crate::agent::tools::names::TOOL_WEB_FETCH;
const DESCRIPTION: &str = "Fetch a URL and return clean markdown. This is not a browser and does not execute page JavaScript. Use after web_search when you have a specific page. Auto order is TinyFish when TINYFISH_API_KEY is set, then Firecrawl when FIRECRAWL_API_KEY is set, then plain HTTP. With no fetch keys, only plain HTTP runs. Do not use this for private or metadata hosts.";

pub struct WebFetchTool {
    router: WebFetchRouter,
}

impl std::fmt::Debug for WebFetchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebFetchTool").finish_non_exhaustive()
    }
}

impl WebFetchTool {
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
            router: WebFetchRouter::from_config(config, transport),
        }
    }
}

impl ToolMeta for WebFetchTool {
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
                "url": {
                    "type": "string",
                    "description": "http or https URL to extract."
                },
                "provider": {
                    "type": "string",
                    "enum": [
                        PROVIDER_AUTO,
                        PROVIDER_TINYFISH,
                        PROVIDER_FIRECRAWL,
                        PROVIDER_HTTP,
                    ],
                    "description": "Backend. auto uses TinyFish (key), then Firecrawl (key), then plain HTTP."
                }
            },
            "required": ["url"]
        })
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let url = match params::extract_string(&arguments, "url", None) {
            Ok(value) => value,
            Err(err) => return ToolResult::error(err),
        };
        if url.trim().is_empty() {
            return ToolResult::error("url must not be empty");
        }
        let provider = params::extract_string_or(&arguments, "provider", PROVIDER_AUTO);
        match self.router.fetch(url.trim(), provider.trim()).await {
            Ok(outcome) => json_result(&outcome),
            Err(err) => ToolResult::error(err),
        }
    }
}
