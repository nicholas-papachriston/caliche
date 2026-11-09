//! Integration coverage for JSONL session trees and branch restore.

use caliche::agent::session::{JsonlSessionEntry, JsonlSessionHeader, JsonlSessionStore};
use serde_json::json;
use tempfile::tempdir;

#[tokio::test]
async fn jsonl_session_branch_restore_roundtrip() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("chat.jsonl");
    let store = JsonlSessionStore::new(&path);
    store
        .create(&JsonlSessionHeader::new("sess", Some("/tmp".into())))
        .await
        .expect("create");

    store
        .append_entry(&JsonlSessionEntry::message(
            "u1",
            None,
            json!({"role":"user","content":"one"}),
        ))
        .await
        .expect("append");
    store
        .append_entry(&JsonlSessionEntry::message(
            "a1",
            Some("u1".into()),
            json!({"role":"assistant","content":"two"}),
        ))
        .await
        .expect("append");

    // Branch from first user turn.
    let mut tree = store.load().await.expect("load");
    let branch = tree.branch_from("u1", "br", "retry").expect("branch");
    store
        .append_entry(tree.entries.last().expect("branch entry"))
        .await
        .expect("persist branch");
    store
        .append_entry(&JsonlSessionEntry::message(
            "a2",
            Some(branch),
            json!({"role":"assistant","content":"alt"}),
        ))
        .await
        .expect("append alt");

    let reloaded = store.load().await.expect("reload");
    let main = reloaded.restore_branch_messages("a1").expect("main");
    let alt = reloaded.restore_branch_messages("a2").expect("alt");
    assert_eq!(main.len(), 2);
    assert_eq!(alt.len(), 2);
    assert_eq!(main[1]["content"], "two");
    assert_eq!(alt[1]["content"], "alt");
}
