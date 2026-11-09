//! Trait adapters: schema-defined plugins that implement Tool, Provider, Backend, Skill, Hook,
//! Embedding, Evaluator, DocumentLoader, DurableStreamBackend, OutputParser.

use crate::security::http_policy::HttpPolicy;
use anyhow::{Result, bail};

/// Build a DNS-pinned, no-redirect request for a plugin adapter URL.
pub(crate) fn pinned_request(method: &str, url: &str) -> Result<reqwest::RequestBuilder> {
    let (client, parsed) = HttpPolicy::from_env()?.egress_client(url)?;
    match method {
        "GET" => Ok(client.get(parsed)),
        "POST" => Ok(client.post(parsed)),
        "PUT" => Ok(client.put(parsed)),
        "PATCH" => Ok(client.patch(parsed)),
        "DELETE" => Ok(client.delete(parsed)),
        other => bail!("unsupported method: {other}"),
    }
}

#[cfg(feature = "vector-memory")]
pub mod backend_adapter;
#[cfg(all(feature = "plugins", feature = "document-loaders"))]
pub mod document_loader_adapter;
pub mod embedding_adapter;
pub mod evaluator_adapter;
pub mod hook_adapter;
#[cfg(all(feature = "plugins", feature = "output-parsing"))]
pub mod output_parser_adapter;
pub mod provider_adapter;
pub mod skill_adapter;
pub mod stream_backend_adapter;
pub mod tool_adapter;
