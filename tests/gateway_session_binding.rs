//! Gateway session binding store tests (CAS get_or_insert).

use caliche::gateway::{
    BindingKey, ChannelKind, MemorySessionBindingStore, SessionBinding, SessionBindingStore,
};
use chrono::Utc;
use std::collections::HashMap;

fn sample_binding(session_id: &str, channel_id: &str) -> (BindingKey, SessionBinding) {
    let key = BindingKey {
        tenant_id: Some("t1".to_string()),
        channel_id: channel_id.to_string(),
        channel_kind: ChannelKind::Slack,
        user_id: "U1".to_string(),
        thread_id: Some("th".to_string()),
    };
    let now = Utc::now();
    let binding = SessionBinding {
        key: key.clone(),
        session_id: session_id.to_string(),
        created_at: now,
        updated_at: now,
        metadata: HashMap::new(),
    };
    (key, binding)
}

#[tokio::test]
async fn memory_store_get_put_delete_list() {
    let store = MemorySessionBindingStore::new();
    let (key, binding) = sample_binding("sess-99", "C1");

    assert!(
        store.get(&key).await.expect("get").is_none(),
        "new store should have no bindings"
    );

    store.put(binding.clone()).await.expect("put");
    let got = store
        .get(&key)
        .await
        .expect("get after put")
        .expect("binding should exist");
    assert_eq!(got.session_id, "sess-99");

    let listed = store
        .list_for_session("sess-99")
        .await
        .expect("list for session");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].key.channel_id, "C1");

    store.delete(&key).await.expect("delete");
    assert!(
        store.get(&key).await.expect("get after delete").is_none(),
        "binding should be removed"
    );
    assert!(
        store
            .list_for_session("sess-99")
            .await
            .expect("list after delete")
            .is_empty(),
        "session list should be empty after delete"
    );
}

#[tokio::test]
async fn memory_store_get_or_insert_is_atomic() {
    let store = MemorySessionBindingStore::new();
    let key = BindingKey {
        tenant_id: None,
        channel_kind: ChannelKind::Slack,
        channel_id: "C-race".to_string(),
        user_id: "U-race".to_string(),
        thread_id: None,
    };
    let now = Utc::now();
    let first = SessionBinding {
        key: key.clone(),
        session_id: "sess-first".to_string(),
        created_at: now,
        updated_at: now,
        metadata: HashMap::new(),
    };
    let second = SessionBinding {
        key: key.clone(),
        session_id: "sess-second".to_string(),
        created_at: now,
        updated_at: now,
        metadata: HashMap::new(),
    };

    let (got_first, created_first) = store
        .get_or_insert(first)
        .await
        .expect("first get_or_insert");
    assert!(created_first, "first insert should create binding");
    assert_eq!(got_first.session_id, "sess-first");

    let (got_second, created_second) = store
        .get_or_insert(second)
        .await
        .expect("second get_or_insert");
    assert!(
        !created_second,
        "second insert with same key should not create"
    );
    assert_eq!(
        got_second.session_id, "sess-first",
        "CAS should return existing session id"
    );
}
