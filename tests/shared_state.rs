/// SharedState tests
use caliche::SharedState;
use serde_json::json;
use std::sync::Arc;
use tokio::task;

#[tokio::test]
async fn test_shared_state_get_set() {
    let state = SharedState::new();

    // Set a value
    state.set("/test_key", json!("test_value")).await.unwrap();

    // Get the value
    let value = state.get("/test_key").await.unwrap();
    assert_eq!(value, Some(json!("test_value")));

    // Test nested paths
    state.set("/nested/path/value", json!(42)).await.unwrap();

    let nested_value = state.get("/nested/path/value").await.unwrap();
    assert_eq!(nested_value, Some(json!(42)));
}

#[tokio::test]
async fn test_shared_state_merge() {
    let state = SharedState::new();

    state.set("/customer/name", json!("Alice")).await.unwrap();

    state
        .merge(json!({
            "customer": {
                "email": "alice@example.com"
            }
        }))
        .await
        .unwrap();

    let name = state.get("/customer/name").await.unwrap();
    assert_eq!(name, Some(json!("Alice")));

    let email = state.get("/customer/email").await.unwrap();
    assert_eq!(email, Some(json!("alice@example.com")));
}

#[tokio::test]
async fn test_shared_state_merge_preserves_existing() {
    let state = SharedState::new();

    // Set initial values
    state.set("/customer/name", json!("Alice")).await.unwrap();
    state.set("/customer/age", json!(30)).await.unwrap();

    // Merge new values
    state
        .merge(json!({
            "customer": {
                "email": "alice@example.com"
            }
        }))
        .await
        .unwrap();

    // Verify existing values are preserved
    let name = state.get("/customer/name").await.unwrap();
    assert_eq!(name, Some(json!("Alice")));

    let age = state.get("/customer/age").await.unwrap();
    assert_eq!(age, Some(json!(30)));

    // Verify new value was added
    let email = state.get("/customer/email").await.unwrap();
    assert_eq!(email, Some(json!("alice@example.com")));
}

#[tokio::test]
async fn test_shared_state_edge_cases() {
    let state = SharedState::new();

    // Test root path
    state.set("/", json!({"root": "value"})).await.unwrap();
    let root_value = state.get("/").await.unwrap();
    assert!(root_value.is_some());

    // Test empty path
    state.set("", json!({"empty": "path"})).await.unwrap();
    let empty_value = state.get("").await.unwrap();
    assert!(empty_value.is_some());

    // Test non-existent path
    let missing = state.get("/nonexistent/path").await.unwrap();
    assert_eq!(missing, None);

    // Test array indices
    state.set("/array/0", json!("first")).await.unwrap();
    state.set("/array/1", json!("second")).await.unwrap();
    let first = state.get("/array/0").await.unwrap();
    assert_eq!(first, Some(json!("first")));
}

#[tokio::test]
async fn test_shared_state_get_all() {
    let state = SharedState::new();

    state.set("/key1", json!("value1")).await.unwrap();
    state.set("/key2", json!("value2")).await.unwrap();
    state
        .set("/nested/key", json!("nested_value"))
        .await
        .unwrap();

    let all = state.get_all().await;

    assert_eq!(all["key1"], "value1");
    assert_eq!(all["key2"], "value2");
    assert_eq!(all["nested"]["key"], "nested_value");
}

#[tokio::test]
async fn test_shared_state_deep_nested_merge() {
    let state = SharedState::new();

    state
        .merge(json!({
            "a": {
                "b": {
                    "x": 1
                }
            }
        }))
        .await
        .unwrap();

    state
        .merge(json!({
            "a": {
                "b": {
                    "y": 2
                }
            }
        }))
        .await
        .unwrap();

    let x = state.get("/a/b/x").await.unwrap();
    assert_eq!(x, Some(json!(1)));

    let y = state.get("/a/b/y").await.unwrap();
    assert_eq!(y, Some(json!(2)));
}

#[tokio::test]
async fn test_shared_state_merge_root_replaces_non_object() {
    // Deep merge only recurses when both sides are objects; merging an object
    // into a non-object root replaces the entire root with the incoming value.
    let state = SharedState::from_value(json!("scalar"));

    state.merge(json!({ "replaced": true })).await.unwrap();

    let all = state.get_all().await;
    assert_eq!(all, json!({ "replaced": true }));
}

#[tokio::test]
async fn test_shared_state_concurrent_access() {
    let state = Arc::new(SharedState::new());

    // Spawn multiple tasks that set different values concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let state_clone = Arc::clone(&state);
        let handle = task::spawn(async move {
            state_clone
                .set(&format!("/key_{}", i), json!(i))
                .await
                .unwrap();
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all values were set
    for i in 0..10 {
        let value = state.get(&format!("/key_{}", i)).await.unwrap();
        assert_eq!(value, Some(json!(i)));
    }
}
