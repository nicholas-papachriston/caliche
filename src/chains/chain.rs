//! Core chain trait and types.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChainError {
    #[error("Chain execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Missing required input: {0}")]
    MissingInput(String),

    #[error("Agent error: {0}")]
    AgentError(#[from] anyhow::Error),

    #[error("Agent domain error: {0}")]
    AgentDomainError(#[from] crate::errors::AgentError),

    #[error("Tool error: {0}")]
    ToolError(#[from] crate::errors::ToolError),
}

/// Strategy for managing intermediate steps when limit is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepEvictionStrategy {
    /// Keep first N steps, drop oldest when limit exceeded
    KeepFirst,
    /// Keep last N steps, drop oldest when limit exceeded
    KeepLast,
    /// Keep first N/2 and last N/2 steps, drop middle
    KeepEnds,
}

/// Context shared across chain execution.
#[derive(Debug, Clone)]
pub struct ChainContext {
    /// Intermediate results from previous chains.
    pub intermediate_steps: Vec<(String, Value)>,

    /// Metadata for the chain execution.
    pub metadata: HashMap<String, String>,

    /// Maximum number of intermediate steps to keep (None = unbounded).
    max_steps: Option<usize>,

    /// Strategy for evicting steps when limit is reached.
    eviction_strategy: StepEvictionStrategy,

    /// Lightweight mode: only store step names, not full values.
    lightweight_mode: bool,
}

impl Default for ChainContext {
    fn default() -> Self {
        Self {
            intermediate_steps: Vec::new(),
            metadata: HashMap::new(),
            max_steps: Some(100),
            eviction_strategy: StepEvictionStrategy::KeepEnds,
            lightweight_mode: false,
        }
    }
}

impl ChainContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context with unbounded step storage (use with caution).
    pub fn unbounded() -> Self {
        Self {
            intermediate_steps: Vec::new(),
            metadata: HashMap::new(),
            max_steps: None,
            eviction_strategy: StepEvictionStrategy::KeepLast,
            lightweight_mode: false,
        }
    }

    /// Set the maximum number of intermediate steps to keep.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Set the eviction strategy when step limit is reached.
    pub fn with_eviction_strategy(mut self, strategy: StepEvictionStrategy) -> Self {
        self.eviction_strategy = strategy;
        self
    }

    /// Enable lightweight mode (store step names only, not full values).
    pub fn with_lightweight_mode(mut self, enabled: bool) -> Self {
        self.lightweight_mode = enabled;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn add_step(&mut self, name: impl Into<String>, result: Value) {
        let name = name.into();

        if self.lightweight_mode {
            // Store only a placeholder value in lightweight mode
            self.intermediate_steps.push((name, Value::Null));
        } else {
            self.intermediate_steps.push((name, result));
        }

        // Apply eviction if limit is reached
        if let Some(max) = self.max_steps
            && self.intermediate_steps.len() > max
        {
            self.evict_steps(max);
        }
    }

    /// Evict steps according to the configured strategy.
    fn evict_steps(&mut self, target_size: usize) {
        match self.eviction_strategy {
            StepEvictionStrategy::KeepFirst => {
                // Keep the first `target_size` steps by dropping trailing elements.
                self.intermediate_steps.truncate(target_size);
            }
            StepEvictionStrategy::KeepLast => {
                // Keep last steps, remove from beginning
                if self.intermediate_steps.len() > target_size {
                    let to_remove = self.intermediate_steps.len() - target_size;
                    self.intermediate_steps.drain(..to_remove);
                }
            }
            StepEvictionStrategy::KeepEnds => {
                // Keep first N/2 and last N/2
                if self.intermediate_steps.len() > target_size {
                    let keep_first = target_size / 2;
                    let keep_last = target_size - keep_first;
                    let total = self.intermediate_steps.len();

                    if total > keep_first + keep_last {
                        // Remove middle steps
                        let mut new_steps = Vec::with_capacity(target_size);
                        new_steps.extend_from_slice(&self.intermediate_steps[..keep_first]);
                        new_steps.extend_from_slice(&self.intermediate_steps[total - keep_last..]);
                        self.intermediate_steps = new_steps;
                    }
                }
            }
        }
    }

    pub fn get_step(&self, name: &str) -> Option<&Value> {
        self.intermediate_steps
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
    }

    /// Merge another context into this one, moving data when possible.
    /// This is more efficient than extend() as it can move ownership instead of cloning.
    pub fn merge_from(&mut self, other: ChainContext) {
        // Move intermediate steps (more efficient than extend which clones)
        self.intermediate_steps
            .reserve(other.intermediate_steps.len());
        for step in other.intermediate_steps {
            self.add_step(step.0, step.1);
        }
        // Move metadata
        self.metadata.extend(other.metadata);
    }

    /// Merge another context by reference (clones when necessary).
    /// Use merge_from() when you can take ownership for better performance.
    pub fn merge_from_ref(&mut self, other: &ChainContext) {
        // Clone intermediate steps
        for (name, value) in &other.intermediate_steps {
            self.add_step(name.clone(), value.clone());
        }
        // Clone metadata
        for (k, v) in &other.metadata {
            self.metadata.insert(k.clone(), v.clone());
        }
    }

    /// Append intermediate steps from another context, moving ownership.
    pub fn append_steps(&mut self, mut other: ChainContext) {
        self.intermediate_steps
            .append(&mut other.intermediate_steps);
        // Apply eviction if needed
        if let Some(max) = self.max_steps
            && self.intermediate_steps.len() > max
        {
            self.evict_steps(max);
        }
    }
}

/// Result of a chain execution.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub output: Value,
    pub context: ChainContext,
}

impl ChainResult {
    pub fn new(output: Value) -> Self {
        Self {
            output,
            context: ChainContext::new(),
        }
    }

    pub fn with_context(output: Value, context: ChainContext) -> Self {
        Self { output, context }
    }
}

/// Trait for composable chains.
#[async_trait]
pub trait Chain: Send + Sync + std::fmt::Debug {
    /// Run the chain with the given input.
    async fn run(&self, input: &Value) -> Result<ChainResult, ChainError>;

    /// Run with an existing context.
    async fn run_with_context(
        &self,
        input: &Value,
        _context: ChainContext,
    ) -> Result<ChainResult, ChainError> {
        // Default implementation ignores context
        self.run(input).await
    }

    /// Get the name of this chain.
    fn name(&self) -> &str {
        "chain"
    }

    /// Get the input keys this chain expects.
    fn input_keys(&self) -> Vec<String> {
        vec![]
    }

    /// Get the output keys this chain produces.
    fn output_keys(&self) -> Vec<String> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_context() {
        let mut ctx = ChainContext::new();
        ctx.add_step("step1", Value::String("result1".to_string()));
        ctx.add_step("step2", Value::Number(42.into()));

        assert_eq!(ctx.intermediate_steps.len(), 2);
        assert_eq!(
            ctx.get_step("step1"),
            Some(&Value::String("result1".to_string()))
        );
    }

    #[test]
    fn test_bounded_context() {
        let mut ctx = ChainContext::new().with_max_steps(3);
        ctx.add_step("step1", Value::String("result1".to_string()));
        ctx.add_step("step2", Value::Number(42.into()));
        ctx.add_step("step3", Value::Bool(true));
        ctx.add_step("step4", Value::String("result4".to_string()));

        // Should have evicted to keep within limit
        assert_eq!(ctx.intermediate_steps.len(), 3);
    }

    #[test]
    fn test_keep_first_eviction_retains_correct_count() {
        let mut ctx = ChainContext::new()
            .with_max_steps(3)
            .with_eviction_strategy(StepEvictionStrategy::KeepFirst);
        for i in 0..5usize {
            ctx.add_step(format!("step{i}"), Value::Number(i.into()));
        }
        // Must have exactly `max_steps` elements, not fewer.
        assert_eq!(ctx.intermediate_steps.len(), 3);
        // The *first* three steps must be retained.
        assert_eq!(ctx.intermediate_steps[0].0, "step0");
        assert_eq!(ctx.intermediate_steps[1].0, "step1");
        assert_eq!(ctx.intermediate_steps[2].0, "step2");
    }

    #[test]
    fn test_lightweight_mode() {
        let mut ctx = ChainContext::new().with_lightweight_mode(true);
        ctx.add_step("step1", Value::String("result1".to_string()));

        // In lightweight mode, value should be Null
        assert_eq!(ctx.get_step("step1"), Some(&Value::Null));
    }

    #[test]
    fn test_unbounded_context() {
        let mut ctx = ChainContext::unbounded();
        for i in 0..200 {
            ctx.add_step(format!("step{}", i), Value::Number(i.into()));
        }
        // Unbounded should keep all steps
        assert_eq!(ctx.intermediate_steps.len(), 200);
    }

    #[test]
    fn test_chain_result() {
        let result = ChainResult::new(Value::String("output".to_string()));
        assert_eq!(result.output, Value::String("output".to_string()));
    }
}
