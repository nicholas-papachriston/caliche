//! Configuration for N-tier chain backends.
//!
//! - [`TierConfig`]: per-tier name, capacity limits (`max_vectors`, `max_bytes`), and
//!   `promotion_threshold` (access count required before promoting into this tier).
//! - [`WriteStrategy`]: how writes are distributed (all tiers, source only, or source + hot).
//! - [`SearchStrategy`]: how reads fan out across tiers during `search()`.
//! - [`TieredChainConfig`]: full chain config (tiers, write strategy, background sync, search strategy).

/// Per-tier configuration for capacity and promotion behavior.
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Label for logging and debugging.
    pub name: String,
    /// Maximum number of vectors in this tier (None = unlimited).
    pub max_vectors: Option<usize>,
    /// Maximum memory usage in bytes (None = unlimited).
    pub max_bytes: Option<usize>,
    /// Number of accesses needed before promoting to this tier.
    pub promotion_threshold: usize,
}

impl TierConfig {
    /// Create a new tier config with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_vectors: None,
            max_bytes: None,
            promotion_threshold: 1,
        }
    }

    /// Set the maximum number of vectors in this tier.
    #[must_use]
    pub const fn max_vectors(mut self, max: usize) -> Self {
        self.max_vectors = Some(max);
        self
    }

    /// Set the maximum memory usage in bytes for this tier.
    #[must_use]
    pub const fn max_bytes(mut self, bytes: usize) -> Self {
        self.max_bytes = Some(bytes);
        self
    }

    /// Set the access count threshold for promotion to this tier.
    #[must_use]
    pub const fn promotion_threshold(mut self, threshold: usize) -> Self {
        self.promotion_threshold = threshold;
        self
    }
}

/// Strategy for how writes flow through the tier chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WriteStrategy {
    /// Write to all tiers (most durable, highest write latency).
    WriteThrough,
    /// Write to tier 0 (source of truth) only; promote lazily on read.
    WriteToSource,
    /// Write to tier 0 and the hottest tier (tier N-1).
    #[default]
    WriteToSourceAndHot,
}

/// How `search()` consults tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchStrategy {
    /// Query every tier (hottest to coldest), merge, dedupe by chunk id (hotter wins), then sort.
    #[default]
    SearchAllTiers,
    /// Query the hottest tier first, then colder tiers.
    ///
    /// Early stop happens only after a tier that holds a complete copy of the corpus
    /// has already been searched and at least `limit` unique ids were collected.
    /// A tier is complete when it has no `max_vectors` / `max_bytes` cap and the write
    /// strategy fills it (`WriteThrough`, or the source and the hottest tier for
    /// `WriteToSourceAndHot`). A capped hot tier is a partial cache: search keeps
    /// going so those hits cannot hide a nearer neighbor on a colder tier.
    HotThenSourceFallback,
}

/// Policy for how a failing tier-search is handled during chain search.
///
/// The default (`fallback_on_tier_error = false`) propagates the first error
/// encountered so callers can distinguish "empty result" from "tier failure".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierErrorPolicy {
    /// If true, a cold-tier error in any search strategy is logged at `warn`
    /// and treated as "no hits" for that tier. Default: `false` (propagate).
    pub fallback_on_tier_error: bool,
}

impl TierErrorPolicy {
    /// Strict policy: propagate the first tier error. This is the default.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            fallback_on_tier_error: false,
        }
    }

    /// Best-effort policy: log and skip errors from individual tiers.
    #[must_use]
    pub const fn best_effort() -> Self {
        Self {
            fallback_on_tier_error: true,
        }
    }
}

/// Chain-level configuration for tiered storage.
#[derive(Debug, Clone)]
pub struct TieredChainConfig {
    /// Per-tier settings; length must match the number of backends.
    pub tiers: Vec<TierConfig>,
    /// How writes are distributed across tiers.
    pub write_strategy: WriteStrategy,
    /// How `search()` fans out across tiers.
    pub search_strategy: SearchStrategy,
    /// Enable periodic full-list sync from hotter tiers down to colder tiers.
    ///
    /// Default is off. Each tick lists every vector on the warmer tiers.
    pub background_sync: bool,
    /// Sync interval in seconds.
    pub sync_interval_secs: u64,
    /// Policy controlling how per-tier search errors are handled.
    pub error_policy: TierErrorPolicy,
    /// Maximum concurrent promotion batches.
    ///
    /// Default is 1. Higher values overlap promotions; capacity accounting is then
    /// best-effort because two promotions can evict and insert at the same time.
    pub background_sync_semaphore: usize,
}

impl TieredChainConfig {
    /// Create config with the given tier configs and defaults for the rest.
    #[must_use]
    pub fn new(tiers: Vec<TierConfig>) -> Self {
        Self {
            tiers,
            write_strategy: WriteStrategy::default(),
            search_strategy: SearchStrategy::default(),
            background_sync: false,
            sync_interval_secs: 300,
            error_policy: TierErrorPolicy::default(),
            background_sync_semaphore: 1,
        }
    }

    /// Set the write strategy.
    #[must_use]
    pub const fn write_strategy(mut self, strategy: WriteStrategy) -> Self {
        self.write_strategy = strategy;
        self
    }

    /// Set how `search()` consults tiers.
    #[must_use]
    pub const fn search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.search_strategy = strategy;
        self
    }

    /// Enable or disable background sync.
    #[must_use]
    pub const fn background_sync(mut self, enabled: bool) -> Self {
        self.background_sync = enabled;
        self
    }

    /// Set the sync interval in seconds.
    #[must_use]
    pub const fn sync_interval_secs(mut self, secs: u64) -> Self {
        self.sync_interval_secs = secs;
        self
    }

    /// Set the tier error policy.
    #[must_use]
    pub const fn error_policy(mut self, policy: TierErrorPolicy) -> Self {
        self.error_policy = policy;
        self
    }

    /// Set the maximum concurrent promotion batches.
    #[must_use]
    pub fn background_sync_semaphore(mut self, n: usize) -> Self {
        self.background_sync_semaphore = n.max(1);
        self
    }

    /// Whether tier `tier_idx` is filled with every write and is not capacity-capped.
    ///
    /// Search may stop after a complete tier. A capped tier, or a tier the write
    /// strategy does not fill, is a partial cache.
    #[must_use]
    pub fn tier_has_complete_corpus(&self, tier_idx: usize) -> bool {
        let Some(tier) = self.tiers.get(tier_idx) else {
            return false;
        };
        if tier.max_vectors.is_some() || tier.max_bytes.is_some() {
            return false;
        }
        let hottest = self.tiers.len().saturating_sub(1);
        match self.write_strategy {
            WriteStrategy::WriteThrough => true,
            WriteStrategy::WriteToSourceAndHot => tier_idx == 0 || tier_idx == hottest,
            WriteStrategy::WriteToSource => tier_idx == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_hot_tier_is_not_a_complete_corpus() {
        let config = TieredChainConfig::new(vec![
            TierConfig::new("cold"),
            TierConfig::new("hot").max_vectors(1),
        ]);
        assert!(config.tier_has_complete_corpus(0));
        assert!(!config.tier_has_complete_corpus(1));
    }

    #[test]
    fn write_to_source_leaves_an_uncapped_hot_tier_partial() {
        let config = TieredChainConfig::new(vec![TierConfig::new("cold"), TierConfig::new("hot")])
            .write_strategy(WriteStrategy::WriteToSource);
        assert!(config.tier_has_complete_corpus(0));
        assert!(!config.tier_has_complete_corpus(1));
    }

    #[test]
    fn write_through_treats_uncapped_tiers_as_complete() {
        let config = TieredChainConfig::new(vec![
            TierConfig::new("cold"),
            TierConfig::new("warm"),
            TierConfig::new("hot"),
        ])
        .write_strategy(WriteStrategy::WriteThrough);
        assert!(config.tier_has_complete_corpus(0));
        assert!(config.tier_has_complete_corpus(1));
        assert!(config.tier_has_complete_corpus(2));
    }
}
