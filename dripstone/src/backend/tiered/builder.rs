//! Fluent builder for N-tier chain backends.
//!
//! Add tiers in cold-to-hot order with [`TieredChainBuilder::tier`]. At least two
//! tiers are required. Call [`TieredChainBuilder::build`] to produce an `Arc<dyn Backend>`.

use std::sync::Arc;

use crate::errors::DripstoneError;
use crate::traits::Backend;

use super::chain::TieredChainBackend;
use super::config::{
    SearchStrategy, TierConfig, TierErrorPolicy, TieredChainConfig, WriteStrategy,
};

/// Builder for tiered chain backends with ordered tiers (cold -> hot).
pub struct TieredChainBuilder {
    tiers: Vec<Arc<dyn Backend>>,
    tier_configs: Vec<TierConfig>,
    write_strategy: Option<WriteStrategy>,
    search_strategy: Option<SearchStrategy>,
    background_sync: Option<bool>,
    sync_interval_secs: Option<u64>,
    error_policy: Option<TierErrorPolicy>,
    background_sync_semaphore: Option<usize>,
}

impl TieredChainBuilder {
    /// Create a new tiered chain builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tiers: Vec::new(),
            tier_configs: Vec::new(),
            write_strategy: None,
            search_strategy: None,
            background_sync: None,
            sync_interval_secs: None,
            error_policy: None,
            background_sync_semaphore: None,
        }
    }

    /// Add a tier. First added is coldest (source of truth), last is hottest.
    #[must_use]
    pub fn tier(mut self, backend: Arc<dyn Backend>, config: TierConfig) -> Self {
        self.tiers.push(backend);
        self.tier_configs.push(config);
        self
    }

    /// Set the write strategy.
    #[must_use]
    pub const fn write_strategy(mut self, strategy: WriteStrategy) -> Self {
        self.write_strategy = Some(strategy);
        self
    }

    /// Set how `search()` fans out across tiers.
    #[must_use]
    pub const fn search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.search_strategy = Some(strategy);
        self
    }

    /// Enable or disable background sync.
    #[must_use]
    pub const fn background_sync(mut self, enabled: bool) -> Self {
        self.background_sync = Some(enabled);
        self
    }

    /// Set sync interval in seconds.
    #[must_use]
    pub const fn sync_interval_secs(mut self, secs: u64) -> Self {
        self.sync_interval_secs = Some(secs);
        self
    }

    /// Set the tier error policy (strict / best-effort).
    #[must_use]
    pub const fn error_policy(mut self, policy: TierErrorPolicy) -> Self {
        self.error_policy = Some(policy);
        self
    }

    /// Set the maximum concurrent background promotion batches.
    #[must_use]
    pub const fn background_sync_semaphore(mut self, n: usize) -> Self {
        self.background_sync_semaphore = Some(n);
        self
    }

    /// Build the tiered chain backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        if self.tiers.len() < 2 {
            return Err(DripstoneError::Misconfiguration(
                "tiered chain requires at least 2 tiers".to_string(),
            ));
        }

        if self.tier_configs.len() != self.tiers.len() {
            return Err(DripstoneError::Misconfiguration(
                "number of tier configs must match number of backends".to_string(),
            ));
        }

        if let Some(cfg) = self.tier_configs.first()
            && (cfg.max_vectors.is_some() || cfg.max_bytes.is_some())
        {
            tracing::warn!(
                "Tier 0 (source of truth) has capacity limits; consider leaving it unlimited"
            );
        }

        let mut config = TieredChainConfig::new(self.tier_configs);
        if let Some(ws) = self.write_strategy {
            config = config.write_strategy(ws);
        }
        if let Some(ss) = self.search_strategy {
            config = config.search_strategy(ss);
        }
        if let Some(enabled) = self.background_sync {
            config = config.background_sync(enabled);
        }
        if let Some(secs) = self.sync_interval_secs {
            config = config.sync_interval_secs(secs);
        }
        if let Some(policy) = self.error_policy {
            config = config.error_policy(policy);
        }
        if let Some(n) = self.background_sync_semaphore {
            config = config.background_sync_semaphore(n);
        }

        let backend = TieredChainBackend::new(self.tiers, config)?;
        Ok(Arc::new(backend))
    }
}

impl Default for TieredChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}
