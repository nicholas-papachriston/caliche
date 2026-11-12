//! N-tier chain backend implementation.
//!
//! [`TieredChainBackend`] holds an ordered list of backends (tier 0 = coldest, last = hottest).
//! Search consults tiers from hottest to coldest so the fast path wins first; writes follow
//! [`super::config::WriteStrategy`]. Promotion and eviction use [`crate::backend::shared::AccessTracker`].
//!
//! ## Background sync ([`TieredChainConfig::background_sync`](super::config::TieredChainConfig::background_sync))
//!
//! When enabled, a Tokio task runs every [`sync_interval_secs`](super::config::TieredChainConfig::sync_interval_secs)
//! and walks tiers from hot to cold (indices `N-1` .. `1`). For each adjacent pair it lists every
//! namespace on the hotter tier, loads **all** chunks via `list`, and upserts the full batch into
//! the next colder tier. Per-tick cost is proportional to **total vectors on warm/hot tiers** (list
//! + upsert I/O), not incremental deltas.
//!
//! Off by default. Enable when colder tiers must stay aligned with warmer tiers without relying on search-time
//! promotion (e.g. DR copies, shared warm caches). For steady-state query latency, prefer
//! [`SearchStrategy::HotThenSourceFallback`](super::config::SearchStrategy::HotThenSourceFallback)
//! or a single hot indexed tier so reads hit the fast path first; disable background sync when
//! promotion-on-read and write strategies already keep the hot tier warm enough.

use async_trait::async_trait;
use futures::future::try_join_all;
use parking_lot::RwLock;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};

use crate::backend::shared::{
    AccessTracker, BackgroundTask, build_composite_key, parse_composite_key,
};
use crate::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::traits::Backend;

use super::config::{SearchStrategy, TierErrorPolicy, TieredChainConfig, WriteStrategy};

/// Shared state for [`TieredChainBackend`] so search can spawn deferred promotion work.
pub struct TieredChainInner {
    pub tiers: Vec<Arc<dyn Backend>>,
    pub config: TieredChainConfig,
    pub access_tracker: Arc<AccessTracker>,
    pub tier_sizes: Vec<Arc<RwLock<usize>>>,
    /// Composite keys known to have been written through this chain.
    pub tier_residents: Vec<Arc<RwLock<HashSet<String>>>>,
    pub promotions_in_flight: Arc<AsyncMutex<HashSet<String>>>,
    pub promotion_batch_gate: Arc<Semaphore>,
}

impl TieredChainInner {
    fn estimate_chunk_size(chunk: &Chunk) -> usize {
        let base_size = std::mem::size_of::<Chunk>();
        let vector_size = chunk.vector.len() * std::mem::size_of::<f32>();
        let id_size = chunk.id.len();
        let doc_id_size = chunk.doc_id.len();
        let content_size = chunk.content.len();
        let namespace_size = chunk.namespace.as_ref().map_or(0, std::string::String::len);
        let metadata_size = chunk.metadata.to_string().len();

        base_size
            + vector_size
            + id_size
            + doc_id_size
            + content_size
            + namespace_size
            + metadata_size
            + 100
    }

    #[expect(
        clippy::expect_used,
        reason = "TieredChain construction always installs at least one tier"
    )]
    fn source_tier(&self) -> &Arc<dyn Backend> {
        self.tiers.first().expect("at least one tier")
    }

    #[expect(
        clippy::expect_used,
        reason = "TieredChain construction always installs at least one tier"
    )]
    fn hottest_tier(&self) -> &Arc<dyn Backend> {
        self.tiers.last().expect("at least one tier")
    }

    fn should_promote_to_tier(&self, key: &str, tier_idx: usize) -> bool {
        let Some(tier_cfg) = self.config.tiers.get(tier_idx) else {
            return false;
        };
        self.access_tracker
            .should_promote(key, tier_cfg.promotion_threshold)
    }

    fn record_access(&self, key: String, size_bytes: usize) {
        self.access_tracker.record_access(key, size_bytes);
    }

    async fn evict_from_tier(
        &self,
        tier_idx: usize,
        required_bytes: usize,
    ) -> Result<(), DripstoneError> {
        let tier = &self.tiers[tier_idx];
        let tier_cfg = &self.config.tiers[tier_idx];
        let current_size = *self.tier_sizes[tier_idx].read();
        let count = tier.count(None).await.unwrap_or(0);

        let needs_eviction = match (tier_cfg.max_bytes, tier_cfg.max_vectors) {
            (Some(max_bytes), _) => current_size + required_bytes > max_bytes,
            (_, Some(max_vectors)) => count >= max_vectors,
            (None, None) => false,
        };

        if !needs_eviction {
            return Ok(());
        }

        let target_size = tier_cfg
            .max_bytes
            .map(|max| max.saturating_sub(required_bytes));
        let target_count = tier_cfg.max_vectors.map(|max| max.saturating_sub(1));

        let residents = self.tier_residents[tier_idx].read().clone();
        let mut candidates: Vec<(String, crate::backend::shared::AccessMetrics)> =
            self.access_tracker.get_all_metrics();
        if !residents.is_empty() {
            candidates.retain(|(key, _)| residents.contains(key));
        }
        candidates.sort_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed));

        let mut current = current_size;
        let mut evicted_count = 0;

        for (key, metrics) in &candidates {
            let under_size = target_size.is_none_or(|t| current <= t);
            let under_count = target_count.is_none_or(|t| count.saturating_sub(evicted_count) <= t);
            if under_size && under_count {
                break;
            }

            let (namespace, id) = parse_composite_key(key);
            tier.delete(id, namespace).await?;
            let freed = self.forget_resident(tier_idx, namespace, id);
            let freed = if freed == 0 {
                let mut size = self.tier_sizes[tier_idx].write();
                *size = size.saturating_sub(metrics.size_bytes);
                metrics.size_bytes
            } else {
                freed
            };
            let _ = self.access_tracker.remove(key);
            current = current.saturating_sub(freed);
            evicted_count += 1;
        }

        if evicted_count > 0 {
            tracing::debug!(
                "Evicted {} vectors from tier {} ({})",
                evicted_count,
                tier_idx,
                tier_cfg.name
            );
        }

        Ok(())
    }

    async fn promote_to_tier(&self, tier_idx: usize, chunk: &Chunk) -> Result<(), DripstoneError> {
        self.evict_from_tier(tier_idx, Self::estimate_chunk_size(chunk))
            .await?;

        self.tiers[tier_idx]
            .upsert(std::slice::from_ref(chunk))
            .await?;
        self.note_resident(tier_idx, chunk);

        Ok(())
    }

    fn note_resident(&self, tier_idx: usize, chunk: &Chunk) {
        let key = build_composite_key(chunk.namespace.as_deref(), &chunk.id);
        let mut residents = self.tier_residents[tier_idx].write();
        if residents.insert(key) {
            *self.tier_sizes[tier_idx].write() += Self::estimate_chunk_size(chunk);
        }
    }

    fn forget_resident(&self, tier_idx: usize, namespace: Option<&str>, id: &str) -> usize {
        let key = build_composite_key(namespace, id);
        let removed = self.tier_residents[tier_idx].write().remove(&key);
        if !removed {
            return 0;
        }
        let size = self
            .access_tracker
            .get_metrics(&key)
            .map_or(0, |metrics| metrics.size_bytes);
        let mut tier_size = self.tier_sizes[tier_idx].write();
        *tier_size = tier_size.saturating_sub(size);
        size
    }

    /// Promote a chunk upward through the chain (from cold toward hot) as far as
    /// access thresholds allow. Start at tier 1 (first tier above source).
    pub(crate) async fn promote_chain(&self, chunk: &Chunk) -> Result<(), DripstoneError> {
        let key = build_composite_key(chunk.namespace.as_deref(), &chunk.id);

        for tier_idx in 1..self.tiers.len() {
            let should_promote = self.should_promote_to_tier(&key, tier_idx);
            if should_promote && let Err(e) = self.promote_to_tier(tier_idx, chunk).await {
                tracing::warn!(
                    "Failed to promote chunk {} to tier {}: {}",
                    chunk.id,
                    tier_idx,
                    e
                );
            }
        }

        Ok(())
    }

    async fn reserve_promotions(&self, chunks: Vec<Chunk>) -> Vec<Chunk> {
        let mut in_flight = self.promotions_in_flight.lock().await;
        let mut reserved = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let key = build_composite_key(chunk.namespace.as_deref(), &chunk.id);
            if in_flight.insert(key) {
                reserved.push(chunk);
            }
        }
        reserved
    }

    async fn release_promotion(&self, chunk: &Chunk) {
        let key = build_composite_key(chunk.namespace.as_deref(), &chunk.id);
        let mut in_flight = self.promotions_in_flight.lock().await;
        in_flight.remove(&key);
    }

    /// Periodic hot-to-cold copy; see module docs for cost model and when to enable.
    ///
    /// The task holds its own clones of the tiers, so it runs until the
    /// returned guard drops.
    fn start_background_sync(inner: &Self) -> Option<BackgroundTask> {
        let interval = std::time::Duration::from_secs(inner.config.sync_interval_secs);
        let tiers: Vec<Arc<dyn Backend>> = inner.tiers.iter().map(Arc::clone).collect();

        // Avoid spinning up a dedicated OS thread + single-threaded runtime;
        // this task cooperates with the caller's runtime and sleeps via
        // `tokio::time::interval`.
        if tokio::runtime::Handle::try_current().is_err() {
            tracing::warn!(
                "TieredChainBackend::start_background_sync called outside a Tokio runtime; \
                 background tier sync will not run"
            );
            return None;
        }

        Some(BackgroundTask::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Skip the immediate first tick so we start by waiting `interval`.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                for i in (1..tiers.len()).rev() {
                    let from_tier = &tiers[i];
                    let to_tier = &tiers[i - 1];

                    let namespaces = match from_tier.namespaces().await {
                        Ok(ns) => ns,
                        Err(e) => {
                            tracing::warn!(
                                "Failed to list namespaces on tier {i} for background sync: {e}"
                            );
                            continue;
                        }
                    };
                    for namespace in namespaces {
                        let chunks = match from_tier.list(Some(&namespace), None).await {
                            Ok(chunks) => chunks,
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to list chunks in tier {i} namespace {namespace}: {e}"
                                );
                                continue;
                            }
                        };
                        match to_tier.upsert(&chunks).await {
                            Ok(()) => tracing::debug!(
                                "Synced {} chunks from tier {} to tier {} for namespace {}",
                                chunks.len(),
                                i,
                                i - 1,
                                namespace
                            ),
                            Err(e) => tracing::warn!(
                                "Failed to sync namespace {} from tier {} to tier {}: {}",
                                namespace,
                                i,
                                i - 1,
                                e
                            ),
                        }
                    }
                }
                tracing::debug!("Completed background sync");
            }
        }))
    }
}

/// N-tier chain backend: ordered list of backends from cold (index 0) to hot (last).
pub struct TieredChainBackend {
    inner: Arc<TieredChainInner>,
    /// Stops the background sync when the backend drops.
    _sync: Option<BackgroundTask>,
}

impl TieredChainBackend {
    /// Create a new tiered chain backend.
    ///
    /// Returns an error if `tiers` is empty (at least one tier is required for
    /// source and hottest tier accessors to be infallible).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(
        tiers: Vec<Arc<dyn Backend>>,
        config: TieredChainConfig,
    ) -> Result<Self, DripstoneError> {
        if tiers.is_empty() {
            return Err(DripstoneError::configuration(
                "TieredChainBackend requires at least one tier",
            ));
        }
        let n = tiers.len();
        let tier_sizes = (0..n).map(|_| Arc::new(RwLock::new(0usize))).collect();
        let tier_residents = (0..n)
            .map(|_| Arc::new(RwLock::new(HashSet::new())))
            .collect();
        let promotion_permits = config.background_sync_semaphore.max(1);

        let inner = Arc::new(TieredChainInner {
            tiers,
            config,
            access_tracker: Arc::new(AccessTracker::new()),
            tier_sizes,
            tier_residents,
            promotions_in_flight: Arc::new(AsyncMutex::new(HashSet::new())),
            promotion_batch_gate: Arc::new(Semaphore::new(promotion_permits)),
        });

        let sync = if inner.config.background_sync {
            TieredChainInner::start_background_sync(&inner)
        } else {
            None
        };

        Ok(Self { inner, _sync: sync })
    }

    /// Convert a per-tier search error into either an empty result (best-effort
    /// policy) or a propagated error (strict policy). This centralises the
    /// decision so all five search paths honour the configured [`TierErrorPolicy`].
    fn handle_tier_error<T: Default>(
        policy: TierErrorPolicy,
        tier_idx: usize,
        tier_name: &str,
        err: DripstoneError,
    ) -> Result<T, DripstoneError> {
        if policy.fallback_on_tier_error {
            tracing::warn!(
                tier = tier_idx,
                tier_name = tier_name,
                error = %err,
                "tier search failed; policy=best_effort so tier is skipped"
            );
            Ok(T::default())
        } else {
            Err(err)
        }
    }

    /// Record hits from one tier into the merge map (hotter tier wins on duplicate ids).
    fn ingest_tier_search_results(
        inner: &TieredChainInner,
        tier_idx: usize,
        hottest: usize,
        results: Vec<SearchResult>,
        merged: &mut HashMap<String, (usize, SearchResult)>,
        to_promote: &mut Vec<Chunk>,
    ) {
        for result in results {
            let key = build_composite_key(result.chunk.namespace.as_deref(), &result.chunk.id);
            let size = TieredChainInner::estimate_chunk_size(&result.chunk);
            inner.record_access(key, size);

            let id = result.chunk.id.clone();
            match merged.entry(id) {
                Entry::Occupied(_) => {}
                Entry::Vacant(v) => {
                    if tier_idx < hottest {
                        to_promote.push(result.chunk.clone());
                    }
                    v.insert((tier_idx, result));
                }
            }
        }
    }

    /// Fan out search hottest-to-coldest, merge/dedupe, and schedule promotion.
    ///
    /// `search_tier` performs the per-tier query. When `exhaustive` is false
    /// ([`SearchStrategy::HotThenSourceFallback`](super::config::SearchStrategy::HotThenSourceFallback)),
    /// colder tiers are skipped only after a complete tier has already contributed
    /// at least `limit` unique ids. A partial hot tier does not end the search.
    async fn merge_tier_search_results<F, Fut>(
        &self,
        limit: usize,
        exhaustive: bool,
        mut search_tier: F,
    ) -> Result<Vec<SearchResult>, DripstoneError>
    where
        F: FnMut(&Arc<dyn Backend>, usize, &str) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<SearchResult>, DripstoneError>>,
    {
        let inner = &self.inner;
        let n = inner.tiers.len();
        let hottest = n.saturating_sub(1);
        let mut merged: HashMap<String, (usize, SearchResult)> = HashMap::new();
        let mut to_promote: Vec<Chunk> = Vec::new();
        let mut saw_complete_tier = false;

        for tier_idx in (0..n).rev() {
            if !exhaustive && saw_complete_tier && merged.len() >= limit {
                break;
            }
            let tier = inner.tiers.get(tier_idx).ok_or_else(|| {
                DripstoneError::Misconfiguration("tiered chain has no tiers".to_string())
            })?;
            let tier_name = inner
                .config
                .tiers
                .get(tier_idx)
                .map_or("", |c| c.name.as_str());
            let searched_ok = match search_tier(tier, tier_idx, tier_name).await {
                Ok(results) => {
                    Self::ingest_tier_search_results(
                        inner,
                        tier_idx,
                        hottest,
                        results,
                        &mut merged,
                        &mut to_promote,
                    );
                    true
                }
                Err(e) => {
                    let empty: Vec<SearchResult> =
                        Self::handle_tier_error(inner.config.error_policy, tier_idx, tier_name, e)?;
                    Self::ingest_tier_search_results(
                        inner,
                        tier_idx,
                        hottest,
                        empty,
                        &mut merged,
                        &mut to_promote,
                    );
                    false
                }
            };
            if searched_ok && inner.config.tier_has_complete_corpus(tier_idx) {
                saw_complete_tier = true;
            }
        }

        self.apply_search_merge(merged, to_promote, limit).await
    }

    async fn search_all_tiers(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.merge_tier_search_results(query.limit, true, |tier, _, _| {
            let tier = Arc::clone(tier);
            async move { tier.search(query).await }
        })
        .await
    }

    /// Hottest tier first; if fewer than `limit` unique hits, query progressively colder tiers
    /// (N-2 .. 0) until enough ids are collected or every tier has been searched.
    async fn search_hot_then_source(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.merge_tier_search_results(query.limit, false, |tier, _, _| {
            let tier = Arc::clone(tier);
            async move { tier.search(query).await }
        })
        .await
    }

    async fn search_all_tiers_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.merge_tier_search_results(query.limit, true, |tier, _, _| {
            let tier = Arc::clone(tier);
            async move { tier.search_full_text(query).await }
        })
        .await
    }

    async fn search_hot_then_source_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.merge_tier_search_results(query.limit, false, |tier, _, _| {
            let tier = Arc::clone(tier);
            async move { tier.search_full_text(query).await }
        })
        .await
    }

    async fn search_all_tiers_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &crate::search::FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let limit = dense.limit.min(lexical.limit);
        self.merge_tier_search_results(limit, true, |tier, _, _| {
            let tier = Arc::clone(tier);
            async move { tier.search_hybrid_full_text(dense, lexical, fusion).await }
        })
        .await
    }

    async fn search_hot_then_source_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &crate::search::FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let limit = dense.limit.min(lexical.limit);
        self.merge_tier_search_results(limit, false, |tier, _, _| {
            let tier = Arc::clone(tier);
            async move { tier.search_hybrid_full_text(dense, lexical, fusion).await }
        })
        .await
    }

    async fn apply_search_merge(
        &self,
        merged: HashMap<String, (usize, SearchResult)>,
        to_promote: Vec<Chunk>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let inner = &self.inner;
        let mut results: Vec<SearchResult> = merged.into_values().map(|(_, r)| r).collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        let to_promote = inner.reserve_promotions(to_promote).await;
        if !to_promote.is_empty() {
            let inner_spawn = Arc::clone(inner);
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let Ok(permit) = inner_spawn.promotion_batch_gate.acquire().await else {
                        return;
                    };
                    for chunk in to_promote {
                        if let Err(e) = inner_spawn.promote_chain(&chunk).await {
                            tracing::warn!(
                                "Deferred promote after search failed for chunk {}: {}",
                                chunk.id,
                                e
                            );
                        }
                        inner_spawn.release_promotion(&chunk).await;
                    }
                    drop(permit);
                });
            }
        }

        Ok(results)
    }
}

impl std::fmt::Debug for TieredChainBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sizes: Vec<usize> = self.inner.tier_sizes.iter().map(|s| *s.read()).collect();
        f.debug_struct("TieredChainBackend")
            .field("tier_count", &self.inner.tiers.len())
            .field("config", &self.inner.config)
            .field("tier_sizes", &sizes)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Backend for TieredChainBackend {
    fn capabilities(&self) -> Capabilities {
        let mut caps = self.inner.source_tier().capabilities();
        for tier in self.inner.tiers.iter().skip(1) {
            let t = tier.capabilities();
            caps.native_hybrid = caps.native_hybrid || t.native_hybrid;
            caps.native_full_text = caps.native_full_text || t.native_full_text;
            caps.native_hybrid_full_text =
                caps.native_hybrid_full_text || t.native_hybrid_full_text;
            caps.native_filter = caps.native_filter || t.native_filter;
            caps.requires_exact_filter_pass =
                caps.requires_exact_filter_pass || t.requires_exact_filter_pass;
            caps.supports_namespaces = caps.supports_namespaces && t.supports_namespaces;
            caps.max_dimension = caps.max_dimension.or(t.max_dimension);
            caps.max_batch = caps.max_batch.or(t.max_batch);
            caps.supports_streaming = caps.supports_streaming || t.supports_streaming;
        }
        caps
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let written = match self.inner.config.write_strategy {
            WriteStrategy::WriteThrough => {
                for tier in &self.inner.tiers {
                    tier.upsert(chunks).await?;
                }
                (0..self.inner.tiers.len()).collect::<Vec<_>>()
            }
            WriteStrategy::WriteToSource => {
                self.inner.source_tier().upsert(chunks).await?;
                vec![0]
            }
            WriteStrategy::WriteToSourceAndHot => {
                self.inner.source_tier().upsert(chunks).await?;
                if self.inner.tiers.len() > 1 {
                    self.inner.hottest_tier().upsert(chunks).await?;
                    vec![0, self.inner.tiers.len() - 1]
                } else {
                    vec![0]
                }
            }
        };

        for chunk in chunks {
            let key = build_composite_key(chunk.namespace.as_deref(), &chunk.id);
            let size = TieredChainInner::estimate_chunk_size(chunk);
            self.inner.record_access(key, size);
            for tier_idx in &written {
                self.inner.note_resident(*tier_idx, chunk);
            }
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let key = build_composite_key(namespace, id);

        for (tier_idx, tier) in self.inner.tiers.iter().enumerate().rev() {
            if let Ok(Some(chunk)) = tier.get(id, namespace).await {
                let size = TieredChainInner::estimate_chunk_size(&chunk);
                self.inner.record_access(key.clone(), size);
                if tier_idx < self.inner.tiers.len() - 1 {
                    self.inner.promote_chain(&chunk).await?;
                }
                return Ok(Some(chunk));
            }
        }

        Ok(None)
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let key = build_composite_key(namespace, id);

        for (tier_idx, tier) in self.inner.tiers.iter().enumerate() {
            tier.delete(id, namespace).await?;
            self.inner.forget_resident(tier_idx, namespace, id);
        }
        let _ = self.inner.access_tracker.remove(&key);

        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let mut max_count = 0usize;
        for tier in &self.inner.tiers {
            let c = tier.delete_document(doc_id, namespace).await?;
            max_count = max_count.max(c);
        }
        if let Some(ns) = namespace {
            self.inner.access_tracker.clear_namespace(ns);
        }
        Ok(max_count)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        match self.inner.config.search_strategy {
            SearchStrategy::SearchAllTiers => self.search_all_tiers(query).await,
            SearchStrategy::HotThenSourceFallback => self.search_hot_then_source(query).await,
        }
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        match self.inner.config.search_strategy {
            SearchStrategy::SearchAllTiers => self.search_all_tiers_full_text(query).await,
            SearchStrategy::HotThenSourceFallback => {
                self.search_hot_then_source_full_text(query).await
            }
        }
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &crate::search::FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        match self.inner.config.search_strategy {
            SearchStrategy::SearchAllTiers => {
                self.search_all_tiers_hybrid_full_text(dense, lexical, fusion)
                    .await
            }
            SearchStrategy::HotThenSourceFallback => {
                self.search_hot_then_source_hybrid_full_text(dense, lexical, fusion)
                    .await
            }
        }
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let mut all: std::collections::HashSet<String> = std::collections::HashSet::new();
        for tier in &self.inner.tiers {
            if let Ok(ns) = tier.namespaces().await {
                all.extend(ns);
            }
        }
        Ok(all.into_iter().collect())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let ns = namespace.to_string();
        let deletes = self.inner.tiers.iter().map(|tier| {
            let t = Arc::clone(tier);
            let ns = ns.clone();
            async move { t.delete_namespace(&ns).await }
        });
        try_join_all(deletes).await?;
        self.inner.access_tracker.clear_namespace(namespace);
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.inner.source_tier().count(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        self.inner.source_tier().list(namespace, limit).await
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        for tier in &self.inner.tiers {
            tier.flush().await?;
        }
        Ok(())
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        for tier in &self.inner.tiers {
            tier.close().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tier_error_policy_tests {
    use super::*;

    #[test]
    fn strict_policy_propagates_error() {
        let err = DripstoneError::storage("boom");
        let policy = TierErrorPolicy::strict();
        let result: Result<Vec<SearchResult>, _> =
            TieredChainBackend::handle_tier_error(policy, 0, "hot", err);
        assert!(result.is_err(), "strict policy must propagate");
    }

    #[test]
    fn best_effort_policy_converts_to_empty_default() {
        let err = DripstoneError::storage("boom");
        let policy = TierErrorPolicy::best_effort();
        let result: Result<Vec<SearchResult>, _> =
            TieredChainBackend::handle_tier_error(policy, 0, "hot", err);
        let v = result.expect("best_effort policy must swallow tier error");
        assert!(v.is_empty(), "best_effort returns Default::default()");
    }

    #[test]
    fn best_effort_on_scalar_default() {
        // Confirms the generic `T: Default` signature works for non-Vec returns.
        let policy = TierErrorPolicy::best_effort();
        let res: Result<Option<SearchResult>, _> =
            TieredChainBackend::handle_tier_error(policy, 2, "cold", DripstoneError::storage("e"));
        assert!(matches!(res, Ok(None)));
    }
}

#[cfg(all(test, feature = "memory-backend"))]
mod background_sync_tests {
    use super::*;
    use crate::backend::memory::MemoryBackend;
    use crate::backend::tiered::config::TierConfig;
    use std::time::Duration;

    const INTERVAL_SECS: u64 = 60;
    const NAMESPACE: &str = "ns";

    /// A cold and a hot tier, with one chunk only in the hot tier.
    #[expect(
        clippy::expect_used,
        reason = "test setup fails the test when seeding fails"
    )]
    async fn tiers() -> (Arc<MemoryBackend>, Vec<Arc<dyn Backend>>) {
        let cold = Arc::new(MemoryBackend::new());
        let hot = Arc::new(MemoryBackend::new());
        let chunk = Chunk::new(
            "doc",
            0,
            "hot only",
            vec![1.0_f32],
            0,
            8,
            1,
            serde_json::json!({}),
            Some(NAMESPACE.to_string()),
        );
        hot.upsert(&[chunk]).await.expect("seed hot tier");
        let tiers: Vec<Arc<dyn Backend>> = vec![cold.clone(), hot];
        (cold, tiers)
    }

    fn syncing() -> TieredChainConfig {
        TieredChainConfig::new(vec![TierConfig::new("cold"), TierConfig::new("hot")])
            .background_sync(true)
            .sync_interval_secs(INTERVAL_SECS)
    }

    #[expect(
        clippy::expect_used,
        reason = "test helper fails the test when listing fails"
    )]
    async fn cold_ids(cold: &MemoryBackend) -> Vec<String> {
        cold.list(Some(NAMESPACE), None)
            .await
            .expect("list cold tier")
            .into_iter()
            .map(|chunk| chunk.id)
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn a_live_backend_copies_hot_chunks_to_the_cold_tier() {
        let (cold, tiers) = tiers().await;
        let backend = TieredChainBackend::new(tiers, syncing()).expect("backend");

        tokio::time::sleep(Duration::from_secs(INTERVAL_SECS + 1)).await;

        assert_eq!(cold_ids(&cold).await, vec!["doc#0".to_string()]);
        drop(backend);
    }

    #[tokio::test(start_paused = true)]
    async fn a_dropped_backend_writes_nothing_after_three_intervals() {
        let (cold, tiers) = tiers().await;
        let backend = TieredChainBackend::new(tiers, syncing()).expect("backend");

        drop(backend);
        tokio::time::sleep(Duration::from_secs(3 * INTERVAL_SECS + 1)).await;

        assert_eq!(cold_ids(&cold).await, Vec::<String>::new());
    }
}
