//! A key the shell exported at launch reaches a gateway child unchanged:
//! neither the keychain at startup nor a key saved mid-session (through
//! `/creds`) replaces it.
//!
//! A gateway child inherits the process environment, plus every name in
//! the in-process credential store (`credential_env::snapshot`). So the
//! shell value reaches the child exactly when the store does not hold that
//! name.
//!
//! The shell-key set is recorded once per process, so the test runs its
//! checks in a child process of this binary, started with the launch
//! environment, which records the set first as startup does. The keychain
//! is never touched: the manager's cache is seeded instead.

use caliche_cli::storage::{SecureCredentialManager, record_shell_env_keys, shell_env_keys};
use caliche_contracts::credential_env;
use std::process::Command;

/// Set in the child process that runs the checks.
const CHILD: &str = "CALICHE_TEST_GATEWAY_SHELL_KEY_CHILD";
const TEST_NAME: &str = "the_shell_key_beats_the_keychain_and_a_key_saved_mid_session";

/// The value a gateway child would get for `name` from the store, which
/// overrides the inherited environment.
fn child_override(name: &str) -> Option<String> {
    credential_env::snapshot()
        .into_iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

#[test]
fn the_shell_key_beats_the_keychain_and_a_key_saved_mid_session() {
    if std::env::var_os(CHILD).is_some() {
        checks_at_launch();
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", TEST_NAME, "--test-threads=1", "--nocapture"])
        .env(CHILD, "1")
        .env("HOME", dir.path())
        .env("CALICHE_NO_KEYRING", "1")
        .env_remove("CALICHE_USE_KEYRING")
        .env("ANTHROPIC_API_KEY", "shell-anthropic")
        .env_remove("GOOGLE_API_KEY")
        .output()
        .expect("run the checks");

    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Runs in the child, whose environment is the launch environment.
fn checks_at_launch() {
    // Startup records the names the shell exported before anything else.
    record_shell_env_keys();
    let home = std::env::var_os("HOME").expect("HOME");
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    runtime.block_on(async {
        let manager =
            SecureCredentialManager::new_for_test(&std::path::Path::new(&home).join("creds.db"))
                .await
                .expect("credential manager");
        manager.test_seed_scope_cache("global", "anthropic", "keychain-anthropic");
        manager.test_seed_scope_cache("global", "google", "keychain-google");

        caliche_cli::bootstrap::load_startup_credentials(&manager, &shell_env_keys()).await;
        let at_startup = (
            child_override("ANTHROPIC_API_KEY"),
            child_override("GOOGLE_API_KEY"),
        );

        // `/creds` saves a key, then puts it in the store the same way.
        manager.test_seed_scope_cache("global", "anthropic", "creds-anthropic");
        manager.publish_to_credential_store().await;
        let mid_session = child_override("ANTHROPIC_API_KEY");

        assert_eq!(at_startup, (None, Some("keychain-google".to_owned())));
        assert_eq!(mid_session, None);
        assert_eq!(
            std::env::var("ANTHROPIC_API_KEY").as_deref(),
            Ok("shell-anthropic")
        );
    });
}
