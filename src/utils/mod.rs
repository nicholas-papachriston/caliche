//! Utility modules for common functionality.
//!
//! This module provides various utility functions and types:
//!
//! - **Parsing**: Text parsing utilities for extracting structured data from LLM responses
//! - **String Pool**: String interning for memory efficiency
//! - **User Context**: User context management for multi-tenant applications
//! - **Cost**: Token cost calculation and estimation
//! - **Health**: Health check utilities
//!
//! # Examples
//!
//! ## Parsing JSON from LLM Output
//!
//! Note: Parsing utilities are internal. Use structured output parsing from the `output` module instead.
//!
//! ## Cost Estimation
//!
//! ```rust,no_run
//! use caliche::cost::CostEstimator;
//! use caliche::llm::ProviderType;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let estimator = CostEstimator::new();
//! let estimate = estimator.estimate_chat(
//!     "What is 2+2?",
//!     ProviderType::OpenAI,
//!     Some("gpt-4o"),
//!     0, // context size
//!     &[], // tools
//! );
//! println!("Estimated cost: ${:.4}", estimate.estimated_cost_usd);
//! # Ok(())
//! # }
//! ```

pub mod beef_helpers;
pub mod cost;
pub mod health;
pub mod json_merge;
pub mod json_path;
pub mod json_schema_lite;
pub mod math;
#[cfg(feature = "reasoning-patterns")]
pub mod parsing;
pub mod string_pool;
pub mod time_id;
pub mod user_context;
