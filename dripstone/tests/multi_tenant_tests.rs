//! Behavioural tests for `MultiTenantManager` quota handling.
//!
//! These exist primarily to lock in the check-then-record ordering
//! contract and the `entry().or_default()` fix applied in the review
//! remediation: a first `check_quota` on a tenant that had no prior usage
//! row must no longer silently bypass limits.

use dripstone::multi_tenant::{MultiTenantManager, TenantQuota};

#[test]
fn check_quota_with_no_usage_is_true_for_default_quota() {
    let mgr = MultiTenantManager::new();
    // Without any configured quota, check_quota should return true (no limit).
    assert!(mgr.check_quota("tenant-a"));
    assert!(mgr.check_quota("tenant-a"), "second call stays true");
}

#[test]
fn embedding_limit_is_enforced_on_next_check() {
    let mgr = MultiTenantManager::new();
    mgr.set_quota(
        "t1".to_string(),
        TenantQuota {
            max_embeddings: 2,
            max_requests_per_minute: 10_000,
            max_storage_bytes: u64::MAX,
        },
    );

    assert!(mgr.check_quota("t1"));
    mgr.record_embedding("t1".to_string(), 10);
    assert!(mgr.check_quota("t1"));
    mgr.record_embedding("t1".to_string(), 10);

    // We have now stored 2 embeddings which is == max_embeddings;
    // the next check must report "over quota".
    assert!(!mgr.check_quota("t1"), "should be over embedding quota");
}

#[test]
fn storage_limit_is_enforced() {
    let mgr = MultiTenantManager::new();
    mgr.set_quota(
        "t2".to_string(),
        TenantQuota {
            max_embeddings: usize::MAX,
            max_requests_per_minute: 10_000,
            max_storage_bytes: 100,
        },
    );

    mgr.record_embedding("t2".to_string(), 50);
    assert!(mgr.check_quota("t2"));
    mgr.record_embedding("t2".to_string(), 60);
    assert!(
        !mgr.check_quota("t2"),
        "storage_bytes (110) >= max_storage_bytes (100)"
    );
}

#[test]
fn rate_limit_is_enforced_within_window() {
    let mgr = MultiTenantManager::new();
    mgr.set_quota(
        "t3".to_string(),
        TenantQuota {
            max_embeddings: usize::MAX,
            max_requests_per_minute: 3,
            max_storage_bytes: u64::MAX,
        },
    );

    for _ in 0..3 {
        assert!(mgr.check_quota("t3"));
        mgr.record_request("t3".to_string());
    }

    assert!(
        !mgr.check_quota("t3"),
        "4th request within the minute must be refused"
    );
}

#[test]
fn check_quota_materializes_usage_row_via_or_default() {
    // The remediation replaced `get_mut` with `entry().or_default()` so
    // that a brand new tenant with no prior writes still evaluates against
    // its configured ceiling. The observable consequence is that setting
    // `max_embeddings = 0` must immediately refuse the very first check.
    let mgr = MultiTenantManager::new();
    mgr.set_quota(
        "t4".to_string(),
        TenantQuota {
            max_embeddings: 0,
            max_requests_per_minute: 10_000,
            max_storage_bytes: u64::MAX,
        },
    );

    assert!(
        !mgr.check_quota("t4"),
        "max_embeddings = 0 must refuse immediately, not after first record"
    );
}
