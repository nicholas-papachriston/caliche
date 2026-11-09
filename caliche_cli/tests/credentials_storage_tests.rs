//! SecureCredentialManager storage and scope resolution tests.

use caliche_cli::storage::SecureCredentialManager;
use tempfile::NamedTempFile;

#[tokio::test]
async fn get_api_key_reads_keyring_cache_without_db_metadata() {
    let db_file = NamedTempFile::new().expect("temp db file");
    let manager = SecureCredentialManager::new_for_test(db_file.path())
        .await
        .expect("test credential manager");
    manager.test_seed_scope_cache("global", "openai", "sk-test-keyring-only");

    let key = manager.get_api_key("openai").await;
    assert_eq!(key.as_deref(), Some("sk-test-keyring-only"));

    let in_db = manager.has_db_credential("openai").await.expect("db read");
    assert!(!in_db, "regression guard: lookup must not depend on DB row");
}
