//! Chain composition operators for ergonomic chain building.
//!
//! This module provides operator overloading for composing chains:
//! - `>>` for sequential composition (pipe output to next chain)
//! - `|` for parallel composition (run chains in parallel and combine results)
//!
//! # Examples
//!
//! ## Sequential Composition
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use caliche::chains::{Chain, ComposableChain, LLMChain};
//! use caliche::prompts::PromptTemplate;
//! use serde_json::json;
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .build()
//!     .await?;
//! let agent = Arc::new(Mutex::new(agent));
//!
//! let chain1 = LLMChain::new(
//!     agent.clone(),
//!     PromptTemplate::new("Summarize: {text}")
//! );
//!
//! let chain2 = LLMChain::new(
//!     agent.clone(),
//!     PromptTemplate::new("Translate to Spanish: {output}")
//! );
//!
//! // Sequential composition: chain1 >> chain2
//! let pipeline = ComposableChain::new(chain1) >> ComposableChain::new(chain2);
//!
//! let result = pipeline.run(&json!({"text": "Long article..."})).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Parallel Composition
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use caliche::chains::{Chain, ComposableChain, LLMChain};
//! use caliche::prompts::PromptTemplate;
//! use serde_json::json;
//! use std::sync::Arc;
//! use tokio::sync::Mutex;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .build()
//!     .await?;
//! let agent = Arc::new(Mutex::new(agent));
//!
//! let chain1 = LLMChain::new(
//!     agent.clone(),
//!     PromptTemplate::new("Summarize: {text}")
//! );
//!
//! let chain2 = LLMChain::new(
//!     agent.clone(),
//!     PromptTemplate::new("Extract keywords: {text}")
//! );
//!
//! // Parallel composition: chain1 | chain2
//! let parallel = ComposableChain::new(chain1) | ComposableChain::new(chain2);
//!
//! let result = parallel.run(&json!({"text": "Article content..."})).await?;
//! // result.output contains results from both chains
//! # Ok(())
//! # }
//! ```

use super::chain::{Chain, ChainError, ChainResult};
use async_trait::async_trait;
use serde_json::Value;
use std::ops::{BitOr, Shr};
use std::sync::Arc;

/// Wrapper for chains that implements composition operators.
///
/// This wrapper enables ergonomic chain composition using `>>` and `|` operators.
pub struct ComposableChain {
    inner: Arc<dyn Chain>,
}

impl std::fmt::Debug for ComposableChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComposableChain")
            .field("inner", &format!("Arc<dyn Chain: {}>", self.inner.name()))
            .finish()
    }
}

impl ComposableChain {
    /// Create a new composable chain from any Chain implementation.
    pub fn new<C: Chain + 'static>(chain: C) -> Self {
        Self {
            inner: Arc::new(chain),
        }
    }

    /// Get a reference to the inner chain.
    pub fn inner(&self) -> &Arc<dyn Chain> {
        &self.inner
    }
}

#[async_trait]
impl Chain for ComposableChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        self.inner.run(input).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn input_keys(&self) -> Vec<String> {
        self.inner.input_keys()
    }

    fn output_keys(&self) -> Vec<String> {
        self.inner.output_keys()
    }
}

// ==================== Sequential Composition (>>) ====================

/// Sequential chain that pipes the output of one chain into another.
pub struct SequentialChain {
    first: Arc<dyn Chain>,
    second: Arc<dyn Chain>,
}

impl std::fmt::Debug for SequentialChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequentialChain")
            .field("first", &format!("Arc<dyn Chain: {}>", self.first.name()))
            .field("second", &format!("Arc<dyn Chain: {}>", self.second.name()))
            .finish()
    }
}

impl SequentialChain {
    /// Create a new sequential chain.
    pub fn new(first: Arc<dyn Chain>, second: Arc<dyn Chain>) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl Chain for SequentialChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        // Run first chain
        let first_result = self.first.run(input).await?;

        // Merge first chain's output with original input for second chain
        // Avoid cloning entire input map - build new map with needed fields
        let mut combined_input = if let Value::Object(map) = input {
            map.clone()
        } else {
            serde_json::Map::new()
        };

        // Add first chain output fields
        if let Value::Object(output_map) = first_result.output {
            for (key, value) in output_map {
                combined_input.insert(key, value);
            }
        }

        // Run second chain with combined input
        let mut second_result = self.second.run(&Value::Object(combined_input)).await?;

        // Merge contexts efficiently (move instead of clone)
        second_result.context.append_steps(first_result.context);

        Ok(second_result)
    }

    fn name(&self) -> &str {
        "sequential_chain"
    }

    fn input_keys(&self) -> Vec<String> {
        self.first.input_keys()
    }

    fn output_keys(&self) -> Vec<String> {
        self.second.output_keys()
    }
}

// Implement >> operator for sequential composition
impl Shr<ComposableChain> for ComposableChain {
    type Output = ComposableChain;

    fn shr(self, rhs: ComposableChain) -> Self::Output {
        ComposableChain {
            inner: Arc::new(SequentialChain::new(self.inner, rhs.inner)),
        }
    }
}

// ==================== Parallel Composition (|) ====================

/// Parallel chain that runs multiple chains concurrently and combines results.
pub struct ParallelChain {
    chains: Vec<Arc<dyn Chain>>,
}

impl std::fmt::Debug for ParallelChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelChain")
            .field(
                "chains",
                &self.chains.iter().map(|c| c.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ParallelChain {
    /// Create a new parallel chain from two chains.
    pub fn new(first: Arc<dyn Chain>, second: Arc<dyn Chain>) -> Self {
        Self {
            chains: vec![first, second],
        }
    }

    /// Create a parallel chain from multiple chains.
    pub fn from_chains(chains: Vec<Arc<dyn Chain>>) -> Self {
        Self { chains }
    }

    /// Add another chain to run in parallel.
    pub fn and(mut self, chain: Arc<dyn Chain>) -> Self {
        self.chains.push(chain);
        self
    }
}

#[async_trait]
impl Chain for ParallelChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        // Run all chains in parallel
        let futures: Vec<_> = self.chains.iter().map(|chain| chain.run(input)).collect();

        let results = futures_util::future::try_join_all(futures)
            .await
            .map_err(|e| ChainError::ExecutionFailed(format!("Parallel chain failed: {}", e)))?;

        // Combine all outputs into a single JSON object
        let mut combined_output = serde_json::Map::new();
        let mut combined_context = super::chain::ChainContext::new();

        for result in results {
            // Merge output
            if let Value::Object(map) = result.output {
                for (key, value) in map {
                    combined_output.insert(key, value);
                }
            }

            // Merge context efficiently (move instead of clone)
            combined_context.append_steps(result.context);
        }

        Ok(ChainResult::with_context(
            Value::Object(combined_output),
            combined_context,
        ))
    }

    fn name(&self) -> &str {
        "parallel_chain"
    }

    fn input_keys(&self) -> Vec<String> {
        // Collect unique input keys from all chains
        let mut keys: Vec<String> = self.chains.iter().flat_map(|c| c.input_keys()).collect();
        keys.sort();
        keys.dedup();
        keys
    }

    fn output_keys(&self) -> Vec<String> {
        // Collect all output keys from all chains
        self.chains.iter().flat_map(|c| c.output_keys()).collect()
    }
}

// Implement | operator for parallel composition
impl BitOr<ComposableChain> for ComposableChain {
    type Output = ComposableChain;

    fn bitor(self, rhs: ComposableChain) -> Self::Output {
        ComposableChain {
            inner: Arc::new(ParallelChain::new(self.inner, rhs.inner)),
        }
    }
}

// Allow chaining multiple parallel chains
impl BitOr<ComposableChain> for ParallelChain {
    type Output = ParallelChain;

    fn bitor(mut self, rhs: ComposableChain) -> Self::Output {
        self.chains.push(rhs.inner);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Mock chain for testing
    struct MockChain {
        name: String,
        output_key: String,
        transform: Box<dyn Fn(&Value) -> Value + Send + Sync>,
    }

    impl std::fmt::Debug for MockChain {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockChain")
                .field("name", &self.name)
                .field("output_key", &self.output_key)
                .field("transform", &"Box<dyn Fn(&Value) -> Value>")
                .finish()
        }
    }

    #[async_trait]
    impl Chain for MockChain {
        async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
            let output = (self.transform)(input);
            Ok(ChainResult::new(json!({ &self.output_key: output })))
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn input_keys(&self) -> Vec<String> {
            vec!["input".to_owned()]
        }

        fn output_keys(&self) -> Vec<String> {
            vec![self.output_key.clone()]
        }
    }

    #[tokio::test]
    async fn test_sequential_composition() {
        let chain1 = MockChain {
            name: "chain1".to_string(),
            output_key: "step1".to_string(),
            transform: Box::new(|input| {
                json!(format!(
                    "processed: {}",
                    input["input"].as_str().expect("input should be a string")
                ))
            }),
        };

        let chain2 = MockChain {
            name: "chain2".to_string(),
            output_key: "step2".to_string(),
            transform: Box::new(|input| {
                json!(format!(
                    "final: {}",
                    input["step1"].as_str().expect("step1 should be a string")
                ))
            }),
        };

        let pipeline = ComposableChain::new(chain1) >> ComposableChain::new(chain2);

        let result = pipeline
            .run(&json!({"input": "test"}))
            .await
            .expect("chain execution should succeed");
        assert_eq!(result.output["step2"], "final: processed: test");
    }

    #[tokio::test]
    async fn test_parallel_composition() {
        let chain1 = MockChain {
            name: "chain1".to_string(),
            output_key: "result1".to_string(),
            transform: Box::new(|_| json!("output1")),
        };

        let chain2 = MockChain {
            name: "chain2".to_string(),
            output_key: "result2".to_string(),
            transform: Box::new(|_| json!("output2")),
        };

        let parallel = ComposableChain::new(chain1) | ComposableChain::new(chain2);

        let result = parallel
            .run(&json!({"input": "test"}))
            .await
            .expect("parallel chain execution should succeed");
        assert_eq!(result.output["result1"], "output1");
        assert_eq!(result.output["result2"], "output2");
    }
}
