use super::branch_timeout_from_metadata;
use std::time::Duration;

#[test]
fn branch_timeout_ms_parses_u64_millis() {
    let metadata = serde_json::json!({ "branch_timeout_ms": 1500 });
    assert_eq!(
        branch_timeout_from_metadata(&metadata),
        Some(Duration::from_millis(1500))
    );
}

#[test]
fn branch_timeout_ms_absent_or_non_u64_is_none() {
    assert_eq!(branch_timeout_from_metadata(&serde_json::json!({})), None);
    assert_eq!(
        branch_timeout_from_metadata(&serde_json::json!({ "branch_timeout_ms": "1000" })),
        None
    );
    assert_eq!(
        branch_timeout_from_metadata(&serde_json::json!({ "branch_timeout_ms": 1000.5 })),
        None
    );
}
