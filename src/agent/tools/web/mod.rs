//! Source-first web search and URL fetch tools with a provider abstraction.

mod error;
mod exa;
mod fetch_tool;
mod firecrawl;
mod http_fallback;
mod provider;
mod search_tool;
mod tavily;
mod tinyfish;
mod transport;
mod types;

pub use fetch_tool::WebFetchTool;
pub use search_tool::WebSearchTool;
pub use types::{WebFetchOutcome, WebProviderConfig, WebSearchHit, WebSearchOutcome};

#[cfg(test)]
mod tests;
