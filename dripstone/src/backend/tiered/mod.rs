//! N-tier chain backend for ordered cold-to-hot storage.
//!
//! Chain any number of backends in order from **cold** (source of truth, index 0)
//! to **hot** (fast cache, last index). Reads query the hottest tier first and
//! fall back to colder tiers; data that meets the per-tier `promotion_threshold`
//! is promoted. Capacity is enforced with per-tier `max_vectors` / `max_bytes`
//! and LRU-style eviction.
//!
//! # Ordering
//!
//! Tiers are added in cold-to-hot order. Examples:
//!
//! - Two-tier: `s3 -> memory`, `sqlite -> redis`
//! - Three-tier: `sqlite -> redis -> memory`, `s3 -> redis -> memory`
//!
//! # Write strategy
//!
//! - [WriteStrategy::WriteThrough](crate::backend::tiered::WriteStrategy::WriteThrough): write to all tiers (most durable).
//! - [WriteStrategy::WriteToSource](crate::backend::tiered::WriteStrategy::WriteToSource): write only to tier 0; promote on read.
//! - [WriteStrategy::WriteToSourceAndHot](crate::backend::tiered::WriteStrategy::WriteToSourceAndHot): write to tier 0 and the hottest tier (default).
//!
//! # Search strategy
//!
//! - [SearchStrategy::SearchAllTiers](crate::backend::tiered::SearchStrategy::SearchAllTiers): query every tier on each search (default).
//! - [SearchStrategy::HotThenSourceFallback](crate::backend::tiered::SearchStrategy::HotThenSourceFallback): hottest tier first. Search stops early only after a complete tier has produced enough hits. A capacity cap or a write strategy that does not fill the tier keeps colder tiers in the query.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use dripstone::traits::Backend;
//! # use dripstone::{TieredChainBuilder, TierConfig};
//!
//! # async fn example(
//! #     cold: Arc<dyn Backend>,
//! #     hot: Arc<dyn Backend>,
//! # ) -> Result<(), dripstone::DripstoneError> {
//! let backend: Arc<dyn Backend> = TieredChainBuilder::new()
//!     .tier(cold, TierConfig::new("cold"))
//!     .tier(hot, TierConfig::new("hot").max_vectors(10_000))
//!     .build()?;
//! # Ok(())
//! # }
//! ```

mod builder;
mod chain;
mod config;

pub use builder::TieredChainBuilder;
pub use chain::TieredChainBackend;
pub use config::{SearchStrategy, TierConfig, TierErrorPolicy, TieredChainConfig, WriteStrategy};
