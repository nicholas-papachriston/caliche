//! Chain abstractions for composable LLM workflows.
//!
//! Chains allow you to compose multiple operations into reusable workflows.
//!
//! # Available Chains
//!
//! ## Basic Chains
//! - **LLMChain**: Simple prompt → LLM → output
//! - **SequentialChain**: Chain multiple operations in sequence
//! - **MapReduceChain**: Process arrays of data in parallel, then reduce
//!
//! ## Retrieval Chains (with `retrieval-chains` feature)
//! - **RetrievalChain**: RAG pattern with vector similarity search
//! - **HybridSearchChain**: Combines vector + keyword search with fusion
//! - **DocumentMapReduceChain**: Process large docs with semantic chunking
//!
//! # Examples
//!
//! LLM Chain:
//!
//! ```rust,no_run
//! use caliche::chains::{LLMChain, Chain};
//! use caliche::prompts::PromptTemplate;
//! use caliche::AgentService;
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//! use serde_json::json;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = Arc::new(Mutex::new(AgentService::new().await?));
//! let prompt = PromptTemplate::new("Summarize: {text}");
//!
//! let chain = LLMChain::new(agent, prompt);
//! let result = chain.run(&json!({"text": "Long document..."})).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Sequential Chain:
//!
//! ```rust,no_run
//! use caliche::chains::{LLMChain, CompositionSequentialChain, Chain};
//! use caliche::prompts::PromptTemplate;
//! use std::sync::Arc;
//! use serde_json::json;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let agent = Arc::new(tokio::sync::Mutex::new(caliche::AgentService::new().await?));
//! let analyze = Arc::new(LLMChain::new(
//!     agent.clone(),
//!     PromptTemplate::new("Analyze: {input}")
//! ));
//!
//! let summarize = Arc::new(LLMChain::new(
//!     agent.clone(),
//!     PromptTemplate::new("Summarize: {output}")
//! ));
//!
//! let chain = CompositionSequentialChain::new(analyze, summarize);
//!
//! let result = chain.run(&json!({"input": "data"})).await?;
//! # Ok(())
//! # }
//! ```

pub mod builder;
pub mod chain;
pub mod composition;
pub mod llm_chain;
pub mod map_reduce;

#[cfg(feature = "retrieval-chains")]
pub mod document_map_reduce;
#[cfg(feature = "retrieval-chains")]
pub mod hybrid_search;
#[cfg(feature = "retrieval-chains")]
pub mod retrieval;

pub use builder::ChainBuilder;
pub use chain::{Chain, ChainContext, ChainError, ChainResult};
pub use composition::{
    ComposableChain, ParallelChain, SequentialChain as CompositionSequentialChain,
};
pub use llm_chain::LLMChain;
pub use map_reduce::MapReduceChain;

#[cfg(feature = "retrieval-chains")]
pub use document_map_reduce::{ChunkingStrategy, DocumentChunk, DocumentMapReduceChain};
#[cfg(feature = "retrieval-chains")]
pub use hybrid_search::{HybridSearchChain, ScoredDocument, SearchSource};
#[cfg(feature = "retrieval-chains")]
pub use retrieval::RetrievalChain;
