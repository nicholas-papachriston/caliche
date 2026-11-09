//! Shared result contracts for web search and fetch backends.

use serde::{Deserialize, Serialize};

pub const PROVIDER_AUTO: &str = "auto";
pub const PROVIDER_TAVILY: &str = "tavily";
pub const PROVIDER_EXA: &str = "exa";
pub const PROVIDER_TINYFISH: &str = "tinyfish";
pub const PROVIDER_FIRECRAWL: &str = "firecrawl";
pub const PROVIDER_HTTP: &str = "http";

pub const DEFAULT_FIRECRAWL_API_URL: &str = "https://api.firecrawl.dev";
pub const DEFAULT_TINYFISH_SEARCH_API_URL: &str = "https://api.search.tinyfish.ai";
pub const DEFAULT_TINYFISH_FETCH_API_URL: &str = "https://api.fetch.tinyfish.ai";
pub const DEFAULT_MAX_RESULTS: usize = 5;
pub const MAX_RESULTS_CAP: usize = 20;

/// One ranked discovery hit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

/// Source-first search outcome returned to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchOutcome {
    pub provider: String,
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    pub results: Vec<WebSearchHit>,
}

/// URL extraction outcome returned to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebFetchOutcome {
    pub provider: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub markdown: String,
}

/// Optional backend credentials.
///
/// A missing Firecrawl key keeps `web_fetch` on plain HTTP. `web_search` can
/// still use Firecrawl's keyless tier.
#[derive(Clone, PartialEq, Eq)]
pub struct WebProviderConfig {
    pub tavily_api_key: Option<String>,
    pub exa_api_key: Option<String>,
    pub tinyfish_api_key: Option<String>,
    pub tinyfish_search_api_url: String,
    pub tinyfish_fetch_api_url: String,
    pub firecrawl_api_key: Option<String>,
    pub firecrawl_api_url: String,
}

impl Default for WebProviderConfig {
    fn default() -> Self {
        Self {
            tavily_api_key: None,
            exa_api_key: None,
            tinyfish_api_key: None,
            tinyfish_search_api_url: DEFAULT_TINYFISH_SEARCH_API_URL.to_owned(),
            tinyfish_fetch_api_url: DEFAULT_TINYFISH_FETCH_API_URL.to_owned(),
            firecrawl_api_key: None,
            firecrawl_api_url: DEFAULT_FIRECRAWL_API_URL.to_owned(),
        }
    }
}

impl WebProviderConfig {
    /// Load keys from process environment. Empty values are treated as unset.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            tavily_api_key: env_nonempty("TAVILY_API_KEY"),
            exa_api_key: env_nonempty("EXA_API_KEY"),
            tinyfish_api_key: env_nonempty("TINYFISH_API_KEY"),
            tinyfish_search_api_url: env_nonempty("TINYFISH_SEARCH_API_URL")
                .unwrap_or_else(|| DEFAULT_TINYFISH_SEARCH_API_URL.to_owned()),
            tinyfish_fetch_api_url: env_nonempty("TINYFISH_FETCH_API_URL")
                .unwrap_or_else(|| DEFAULT_TINYFISH_FETCH_API_URL.to_owned()),
            firecrawl_api_key: env_nonempty("FIRECRAWL_API_KEY"),
            firecrawl_api_url: env_nonempty("FIRECRAWL_API_URL")
                .unwrap_or_else(|| DEFAULT_FIRECRAWL_API_URL.to_owned()),
        }
    }

    #[must_use]
    pub fn tavily_configured(&self) -> bool {
        self.tavily_api_key.is_some()
    }

    #[must_use]
    pub fn exa_configured(&self) -> bool {
        self.exa_api_key.is_some()
    }

    #[must_use]
    pub fn tinyfish_configured(&self) -> bool {
        self.tinyfish_api_key.is_some()
    }

    #[must_use]
    pub fn firecrawl_configured(&self) -> bool {
        self.firecrawl_api_key.is_some()
    }
}

impl std::fmt::Debug for WebProviderConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WebProviderConfig")
            .field(
                "tavily_api_key",
                &redacted_key(self.tavily_api_key.as_deref()),
            )
            .field("exa_api_key", &redacted_key(self.exa_api_key.as_deref()))
            .field(
                "tinyfish_api_key",
                &redacted_key(self.tinyfish_api_key.as_deref()),
            )
            .field("tinyfish_search_api_url", &self.tinyfish_search_api_url)
            .field("tinyfish_fetch_api_url", &self.tinyfish_fetch_api_url)
            .field(
                "firecrawl_api_key",
                &redacted_key(self.firecrawl_api_key.as_deref()),
            )
            .field("firecrawl_api_url", &self.firecrawl_api_url)
            .finish()
    }
}

fn redacted_key(key: Option<&str>) -> Option<&'static str> {
    key.is_some().then_some("[redacted]")
}

#[must_use]
pub fn clamp_max_results(requested: usize) -> usize {
    requested.clamp(1, MAX_RESULTS_CAP)
}

pub(crate) fn firecrawl_endpoint(api_url: &str, path: &str) -> String {
    let trimmed = api_url.trim_end_matches('/');
    if trimmed.ends_with("/v2") {
        format!("{trimmed}/{path}")
    } else {
        format!("{trimmed}/v2/{path}")
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    caliche_contracts::credential_env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_FIRECRAWL_API_URL, WebProviderConfig, clamp_max_results, firecrawl_endpoint,
    };

    #[test]
    fn clamp_max_results_bounds_input() {
        assert_eq!(clamp_max_results(0), 1);
        assert_eq!(clamp_max_results(5), 5);
        assert_eq!(clamp_max_results(99), 20);
    }

    #[test]
    fn firecrawl_endpoint_appends_v2_once() {
        assert_eq!(
            firecrawl_endpoint(DEFAULT_FIRECRAWL_API_URL, "search"),
            "https://api.firecrawl.dev/v2/search"
        );
        assert_eq!(
            firecrawl_endpoint("https://api.firecrawl.dev/v2/", "scrape"),
            "https://api.firecrawl.dev/v2/scrape"
        );
    }

    #[test]
    fn debug_redacts_api_keys() {
        let config = WebProviderConfig {
            tavily_api_key: Some("tvly-secret".to_owned()),
            exa_api_key: Some("exa-secret".to_owned()),
            tinyfish_api_key: Some("tf-secret".to_owned()),
            firecrawl_api_key: Some("fc-secret".to_owned()),
            ..WebProviderConfig::default()
        };
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("tvly-secret"), "{rendered}");
        assert!(!rendered.contains("exa-secret"), "{rendered}");
        assert!(!rendered.contains("tf-secret"), "{rendered}");
        assert!(!rendered.contains("fc-secret"), "{rendered}");
        assert!(rendered.contains("[redacted]"), "{rendered}");
    }
}
