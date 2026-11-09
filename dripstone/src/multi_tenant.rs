//! Tenant quota helpers.
//!
//! # Ordering contract
//!
//! The quota accounting layer is deliberately a pair of independent helpers:
//!
//! - [`MultiTenantManager::check_quota`] inspects current usage and returns
//!   whether the caller is _currently_ under its limits.
//! - [`MultiTenantManager::record_request`] / [`MultiTenantManager::record_embedding`]
//!   mutate the usage counters.
//!
//! Callers MUST use a **check-then-record** pattern at the edge of each
//! tenant-facing operation:
//!
//! ```text
//! if !manager.check_quota(tenant_id) { return Err(QuotaExceeded); }
//! manager.record_request(tenant_id.to_string());
//! // ... perform the tenant-scoped work
//! ```
//!
//! Reversing the order ("record then check") turns the rate limiter into
//! an off-by-one leak: the very request that should have been refused still
//! counts toward the allowance. The window-trimming code in both
//! `check_quota` and `record_request` is resilient to either ordering, so
//! the contract is purely a semantic invariant enforced by callers.

use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct TenantQuota {
    pub max_embeddings: usize,
    pub max_requests_per_minute: u32,
    pub max_storage_bytes: u64,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_embeddings: 100_000,
            max_requests_per_minute: 1000,
            max_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TenantUsage {
    embedding_count: usize,
    storage_bytes: u64,
    // Use VecDeque for efficient removal of old entries from front
    requests: VecDeque<Instant>,
}

/// Multi-tenant isolation and quota management
pub struct MultiTenantManager {
    quotas: Arc<DashMap<String, TenantQuota>>,
    usage: Arc<DashMap<String, TenantUsage>>,
}

impl MultiTenantManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            quotas: Arc::new(DashMap::new()),
            usage: Arc::new(DashMap::new()),
        }
    }

    pub fn set_quota(&self, tenant_id: String, quota: TenantQuota) {
        self.quotas.insert(tenant_id, quota);
    }

    /// Returns whether `tenant_id` is currently under all configured limits.
    ///
    /// Always evaluates the quota against a usage row: if one does not exist
    /// yet it is created via `entry().or_default()`, which keeps the rate
    /// limiter self-consistent after a cold start (the first `check_quota`
    /// for a tenant no longer silently bypasses the rate limit because the
    /// row hasn't been initialised yet).
    #[must_use]
    pub fn check_quota(&self, tenant_id: &str) -> bool {
        let quota = self
            .quotas
            .get(tenant_id)
            .map(|q| q.value().clone())
            .unwrap_or_default();

        let mut usage = self.usage.entry(tenant_id.to_string()).or_default();

        if usage.embedding_count >= quota.max_embeddings {
            return false;
        }
        if usage.storage_bytes >= quota.max_storage_bytes {
            return false;
        }

        let now = Instant::now();
        let window_start = rate_limit_window_start(now, RATE_LIMIT_WINDOW);
        while usage.requests.front().is_some_and(|&t| t < window_start) {
            usage.requests.pop_front();
        }
        let recent_requests = usage.requests.len();
        let max_requests = usize::try_from(quota.max_requests_per_minute).unwrap_or(usize::MAX);
        if recent_requests >= max_requests {
            return false;
        }
        drop(usage);
        true
    }

    pub fn record_request(&self, tenant_id: String) {
        let mut entry = self.usage.entry(tenant_id).or_default();

        let now = Instant::now();
        let window_start = rate_limit_window_start(now, RATE_LIMIT_WINDOW);
        while entry.requests.front().is_some_and(|&t| t < window_start) {
            entry.requests.pop_front();
        }
        entry.requests.push_back(now);
    }

    pub fn record_embedding(&self, tenant_id: String, size_bytes: u64) {
        let mut entry = self.usage.entry(tenant_id).or_default();
        entry.embedding_count = entry.embedding_count.saturating_add(1);
        entry.storage_bytes = entry.storage_bytes.saturating_add(size_bytes);
    }

    #[must_use]
    pub fn get_usage(&self, tenant_id: &str) -> Option<(usize, u64)> {
        self.usage
            .get(tenant_id)
            .map(|u| (u.embedding_count, u.storage_bytes))
    }
}

fn rate_limit_window_start(now: Instant, window: Duration) -> Instant {
    now.checked_sub(window).unwrap_or(now)
}

impl Default for MultiTenantManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Get namespaced key for tenant isolation
#[must_use]
pub fn tenant_namespace(tenant_id: &str, key: &str) -> String {
    format!("tenant:{tenant_id}:{key}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::{MultiTenantManager, TenantQuota, rate_limit_window_start};

    #[test]
    fn rate_limit_window_saturates_when_duration_is_too_large() {
        let now = Instant::now();
        assert_eq!(rate_limit_window_start(now, Duration::MAX), now);
    }

    #[test]
    fn request_limit_rejects_at_exact_boundary() {
        let manager = MultiTenantManager::new();
        manager.set_quota(
            "tenant".to_string(),
            TenantQuota {
                max_requests_per_minute: 1,
                ..TenantQuota::default()
            },
        );

        assert!(manager.check_quota("tenant"));
        manager.record_request("tenant".to_string());
        assert!(!manager.check_quota("tenant"));
    }

    #[test]
    fn concurrent_embedding_updates_are_not_lost() {
        const THREADS: usize = 8;
        const UPDATES_PER_THREAD: usize = 250;

        let manager = Arc::new(MultiTenantManager::new());
        thread::scope(|scope| {
            for _ in 0..THREADS {
                let manager = Arc::clone(&manager);
                scope.spawn(move || {
                    for _ in 0..UPDATES_PER_THREAD {
                        manager.record_embedding("tenant".to_string(), 2);
                    }
                });
            }
        });

        let expected_updates = THREADS * UPDATES_PER_THREAD;
        let expected_bytes = u64::try_from(expected_updates)
            .unwrap_or(u64::MAX)
            .saturating_mul(2);
        assert_eq!(
            manager.get_usage("tenant"),
            Some((expected_updates, expected_bytes))
        );
    }
}
