//! Provider routers for `web_search` and `web_fetch`.

use super::error::WebError;
use super::exa::ExaSearch;
use super::firecrawl::FirecrawlClient;
use super::http_fallback::HttpFetch;
use super::tavily::TavilySearch;
use super::tinyfish::TinyFishClient;
use super::transport::WebHttpTransport;
use super::types::{
    PROVIDER_AUTO, PROVIDER_EXA, PROVIDER_FIRECRAWL, PROVIDER_HTTP, PROVIDER_TAVILY,
    PROVIDER_TINYFISH, WebFetchOutcome, WebProviderConfig, WebSearchOutcome,
};
use std::sync::Arc;

/// Ordered search backends. Auto prefers Tavily, then Exa, then TinyFish, then Firecrawl.
pub struct WebSearchRouter {
    tavily: Option<TavilySearch>,
    exa: Option<ExaSearch>,
    tinyfish: Option<TinyFishClient>,
    firecrawl: FirecrawlClient,
}

impl WebSearchRouter {
    #[must_use]
    pub fn from_config(
        config: Arc<WebProviderConfig>,
        transport: Arc<dyn WebHttpTransport>,
    ) -> Self {
        Self {
            tavily: config
                .tavily_configured()
                .then(|| TavilySearch::new(Arc::clone(&transport), Arc::clone(&config))),
            exa: config
                .exa_configured()
                .then(|| ExaSearch::new(Arc::clone(&transport), Arc::clone(&config))),
            tinyfish: config
                .tinyfish_configured()
                .then(|| TinyFishClient::new(Arc::clone(&transport), Arc::clone(&config))),
            firecrawl: FirecrawlClient::new(transport, config),
        }
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
        requested: &str,
    ) -> Result<WebSearchOutcome, WebError> {
        match requested {
            PROVIDER_TAVILY => self.search_tavily(query, max_results).await,
            PROVIDER_EXA => self.search_exa(query, max_results).await,
            PROVIDER_TINYFISH => self.search_tinyfish(query, max_results).await,
            PROVIDER_FIRECRAWL => self.firecrawl.search(query, max_results).await,
            PROVIDER_AUTO => self.search_auto(query, max_results).await,
            other => Err(WebError::ProviderUnavailable(other.to_owned())),
        }
    }

    async fn search_tavily(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<WebSearchOutcome, WebError> {
        self.tavily
            .as_ref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_TAVILY.to_owned()))?
            .search(query, max_results)
            .await
    }

    async fn search_exa(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<WebSearchOutcome, WebError> {
        self.exa
            .as_ref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_EXA.to_owned()))?
            .search(query, max_results)
            .await
    }

    async fn search_tinyfish(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<WebSearchOutcome, WebError> {
        self.tinyfish
            .as_ref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_TINYFISH.to_owned()))?
            .search(query, max_results)
            .await
    }

    async fn search_auto(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<WebSearchOutcome, WebError> {
        let mut errors = Vec::new();
        let mut last_empty: Option<WebSearchOutcome> = None;
        if let Some(tavily) = &self.tavily {
            match tavily.search(query, max_results).await {
                Ok(outcome) if outcome.results.is_empty() => last_empty = Some(outcome),
                Ok(outcome) => return Ok(outcome),
                Err(err) => errors.push(format!("{PROVIDER_TAVILY}: {err}")),
            }
        }
        if let Some(exa) = &self.exa {
            match exa.search(query, max_results).await {
                Ok(outcome) if outcome.results.is_empty() => last_empty = Some(outcome),
                Ok(outcome) => return Ok(outcome),
                Err(err) => errors.push(format!("{PROVIDER_EXA}: {err}")),
            }
        }
        if let Some(tinyfish) = &self.tinyfish {
            match tinyfish.search(query, max_results).await {
                Ok(outcome) if outcome.results.is_empty() => last_empty = Some(outcome),
                Ok(outcome) => return Ok(outcome),
                Err(err) => errors.push(format!("{PROVIDER_TINYFISH}: {err}")),
            }
        }
        match self.firecrawl.search(query, max_results).await {
            Ok(outcome) if outcome.results.is_empty() => {
                last_empty = Some(outcome);
            }
            Ok(outcome) => return Ok(outcome),
            Err(err) => errors.push(format!("{PROVIDER_FIRECRAWL}: {err}")),
        }
        if let Some(empty) = last_empty {
            return Ok(empty);
        }
        Err(WebError::AllFailed(errors.join("; ")))
    }
}

/// Auto fetch order.
///
/// TinyFish is included only when `TINYFISH_API_KEY` is set. Firecrawl is
/// included only when `FIRECRAWL_API_KEY` is set. Plain HTTP is always last, so
/// a zero-key environment fetches locally and does not send the URL to a
/// third party. This chain does not launch a browser.
#[must_use]
pub fn auto_fetch_providers(
    tinyfish_configured: bool,
    firecrawl_configured: bool,
) -> &'static [&'static str] {
    match (tinyfish_configured, firecrawl_configured) {
        (true, true) => &[PROVIDER_TINYFISH, PROVIDER_FIRECRAWL, PROVIDER_HTTP],
        (true, false) => &[PROVIDER_TINYFISH, PROVIDER_HTTP],
        (false, true) => &[PROVIDER_FIRECRAWL, PROVIDER_HTTP],
        (false, false) => &[PROVIDER_HTTP],
    }
}

/// Ordered fetch backends. Auto prefers TinyFish (key), then Firecrawl (key), then local HTTP extract.
pub struct WebFetchRouter {
    tinyfish: Option<TinyFishClient>,
    firecrawl: Option<FirecrawlClient>,
    http: HttpFetch,
}

impl WebFetchRouter {
    #[must_use]
    pub fn from_config(
        config: Arc<WebProviderConfig>,
        transport: Arc<dyn WebHttpTransport>,
    ) -> Self {
        Self {
            tinyfish: config
                .tinyfish_configured()
                .then(|| TinyFishClient::new(Arc::clone(&transport), Arc::clone(&config))),
            firecrawl: config
                .firecrawl_configured()
                .then(|| FirecrawlClient::new(Arc::clone(&transport), Arc::clone(&config))),
            http: HttpFetch::new(transport),
        }
    }

    pub async fn fetch(&self, url: &str, requested: &str) -> Result<WebFetchOutcome, WebError> {
        match requested {
            PROVIDER_TINYFISH => self.fetch_tinyfish(url).await,
            PROVIDER_FIRECRAWL => self.fetch_firecrawl(url).await,
            PROVIDER_HTTP => self.http.fetch(url).await,
            PROVIDER_AUTO => self.fetch_auto(url).await,
            other => Err(WebError::ProviderUnavailable(other.to_owned())),
        }
    }

    async fn fetch_tinyfish(&self, url: &str) -> Result<WebFetchOutcome, WebError> {
        self.tinyfish
            .as_ref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_TINYFISH.to_owned()))?
            .fetch(url)
            .await
    }

    async fn fetch_firecrawl(&self, url: &str) -> Result<WebFetchOutcome, WebError> {
        self.firecrawl
            .as_ref()
            .ok_or_else(|| WebError::ProviderUnavailable(PROVIDER_FIRECRAWL.to_owned()))?
            .scrape(url)
            .await
    }

    async fn fetch_auto(&self, url: &str) -> Result<WebFetchOutcome, WebError> {
        let mut errors = Vec::new();
        for provider in auto_fetch_providers(self.tinyfish.is_some(), self.firecrawl.is_some()) {
            let result = match *provider {
                PROVIDER_TINYFISH => self.fetch_tinyfish(url).await,
                PROVIDER_FIRECRAWL => self.fetch_firecrawl(url).await,
                PROVIDER_HTTP => self.http.fetch(url).await,
                other => Err(WebError::ProviderUnavailable(other.to_owned())),
            };
            match result {
                Ok(outcome) => return Ok(outcome),
                Err(err) => errors.push(format!("{provider}: {err}")),
            }
        }
        Err(WebError::AllFailed(errors.join("; ")))
    }
}
