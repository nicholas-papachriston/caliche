//! Map-reduce chain - process items in parallel and combine results.

use super::chain::{Chain, ChainError, ChainResult};
use async_trait::async_trait;
use futures_util::future::join_all;
use serde_json::Value;
use std::sync::Arc;

/// Chain that maps over a collection and reduces the results.
///
/// Useful for processing multiple documents in parallel and then combining the results.
/// Supports streaming batch processing to control memory usage for large collections.
pub struct MapReduceChain {
    map_chain: Arc<dyn Chain>,
    reduce_chain: Arc<dyn Chain>,
    batch_size: usize,
    /// Enable incremental reduce (reduce batches as they complete instead of waiting for all)
    incremental_reduce: bool,
}

impl std::fmt::Debug for MapReduceChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapReduceChain")
            .field("map_chain", &"Arc<dyn Chain>")
            .field("reduce_chain", &"Arc<dyn Chain>")
            .field("batch_size", &self.batch_size)
            .field("incremental_reduce", &self.incremental_reduce)
            .finish()
    }
}

impl MapReduceChain {
    /// Create a new map-reduce chain.
    ///
    /// # Arguments
    ///
    /// * `map_chain` - Chain to apply to each item
    /// * `reduce_chain` - Chain to combine the mapped results
    pub fn new(map_chain: Arc<dyn Chain>, reduce_chain: Arc<dyn Chain>) -> Self {
        Self {
            map_chain,
            reduce_chain,
            batch_size: 10,
            incremental_reduce: false,
        }
    }

    /// Set the batch size for parallel processing.
    /// Smaller batches use less memory but may be slower due to more reduce operations.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable incremental reduce mode.
    /// When enabled, reduces batches as they complete instead of collecting all results first.
    /// This reduces memory usage for large collections but may produce slightly different results
    /// if the reduce operation is not associative.
    pub fn with_incremental_reduce(mut self, enabled: bool) -> Self {
        self.incremental_reduce = enabled;
        self
    }

    /// Run the chain with a vector of strings.
    ///
    /// This is a convenience method that automatically wraps strings in JSON objects
    /// if the map chain expects a single "input" variable.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use caliche::chains::{MapReduceChain, LLMChain};
    /// # use caliche::prompts::PromptTemplate;
    /// # use std::sync::Arc;
    /// # async fn example(agent: Arc<tokio::sync::Mutex<caliche::AgentService>>) -> anyhow::Result<()> {
    /// let map_chain = Arc::new(LLMChain::new(
    ///     agent.clone(),
    ///     PromptTemplate::new("Summarize: {input}")
    /// ));
    /// let reduce_chain = Arc::new(LLMChain::new(
    ///     agent.clone(),
    ///     PromptTemplate::new("Combine: {output}")
    /// ));
    /// let chain = MapReduceChain::new(map_chain, reduce_chain);
    ///
    /// // Much simpler than wrapping in JSON objects!
    /// let items = vec!["Article 1", "Article 2", "Article 3"];
    /// let result = chain.run_with_strings(&items, "input").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run_with_strings(
        &self,
        items: &[&str],
        input_key: &str,
    ) -> Result<ChainResult, ChainError> {
        let json_items: Vec<Value> = items
            .iter()
            .map(|item| serde_json::json!({ input_key: item }))
            .collect();

        self.run(&Value::Array(json_items)).await
    }

    /// Run with incremental reduce (process and reduce batches as they complete)
    async fn run_incremental(&self, items: &[Value]) -> Result<ChainResult, ChainError> {
        let mut accumulated_result: Option<Value> = None;

        for chunk in items.chunks(self.batch_size) {
            // Process batch
            let futures: Vec<_> = chunk.iter().map(|item| self.map_chain.run(item)).collect();
            let chunk_results = join_all(futures).await;

            // Extract text from batch results
            let mut batch_texts = Vec::with_capacity(chunk_results.len());
            for result in chunk_results {
                let output = result?.output;
                // Extract text from output
                let text = if let Some(obj) = output.as_object()
                    && let Some(output_val) = obj.get("output")
                {
                    output_val.as_str().unwrap_or("").to_string()
                } else {
                    output.as_str().unwrap_or("").to_string()
                };
                batch_texts.push(text);
            }

            let batch_combined = batch_texts.join("\n\n");

            // Incrementally reduce with accumulated result
            if let Some(acc) = accumulated_result {
                // Combine accumulated with new batch
                let reduce_input = serde_json::json!({
                    "output": format!("{}\n\n{}", acc.as_str().unwrap_or(""), batch_combined)
                });
                let reduced = self.reduce_chain.run(&reduce_input).await?;
                accumulated_result = Some(reduced.output);
            } else {
                // First batch becomes accumulated result
                accumulated_result = Some(serde_json::json!(batch_combined));
            }
        }

        // Final reduce if we have accumulated results
        if let Some(acc) = accumulated_result {
            let reduce_input = serde_json::json!({
                "output": acc.as_str().unwrap_or("")
            });
            self.reduce_chain.run(&reduce_input).await
        } else {
            // Empty input case
            Ok(ChainResult::new(serde_json::json!({})))
        }
    }

    /// Run with batch processing (collect all, then reduce)
    async fn run_batch(&self, items: &[Value]) -> Result<ChainResult, ChainError> {
        // Map phase: process each item in batches
        let mut mapped_results = Vec::with_capacity(items.len());

        for chunk in items.chunks(self.batch_size) {
            let futures: Vec<_> = chunk.iter().map(|item| self.map_chain.run(item)).collect();
            let chunk_results = join_all(futures).await;

            for result in chunk_results {
                mapped_results.push(result?.output);
            }
        }

        // Reduce phase: combine the results (optimized formatting)
        let mut combined_results = String::with_capacity(mapped_results.len() * 100);
        for (i, v) in mapped_results.iter().enumerate() {
            if i > 0 {
                combined_results.push_str("\n\n");
            }
            // Extract text from each mapped result
            let text = if let Some(obj) = v.as_object()
                && let Some(output_val) = obj.get("output")
            {
                output_val.as_str().unwrap_or("")
            } else {
                v.as_str().unwrap_or("")
            };
            combined_results.push_str(text);
        }

        let reduce_input = serde_json::json!({
            "output": combined_results
        });
        let final_result = self.reduce_chain.run(&reduce_input).await?;

        Ok(final_result)
    }
}

#[async_trait]
impl Chain for MapReduceChain {
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError> {
        // Extract array of items to process
        let items = match input {
            Value::Array(arr) => arr,
            _ => {
                return Err(ChainError::InvalidInput(
                    "MapReduceChain requires an array input".to_string(),
                ));
            }
        };

        if self.incremental_reduce {
            self.run_incremental(items).await
        } else {
            self.run_batch(items).await
        }
    }

    fn name(&self) -> &str {
        "map_reduce_chain"
    }

    fn input_keys(&self) -> Vec<String> {
        vec!["items".to_owned()]
    }

    fn output_keys(&self) -> Vec<String> {
        self.reduce_chain.output_keys()
    }
}
