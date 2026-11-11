//! Credentials from `/creds` and the keychain reach readers through the
//! in-process store, and the process environment is never written.
//!
//! Unit tests use the in-memory mock keychain (see `storage::credentials`).
//! The store is process-wide, so each test uses its own credential names.

use super::{SlashCommandExecutor, SlashCommandResult};
use crate::storage::SecureCredentialManager;
use caliche::traits::credentials::{CredentialProvider, EnvironmentCredentialProvider};
use caliche_contracts::credential_env;
use caliche_test_env::EnvGuard;
use std::collections::HashSet;
use std::env::VarError;

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(ToString::to_string).collect()
}

/// Point the credential database at `home` and enable the (mock) keychain.
fn isolated_keychain(env: &mut EnvGuard, home: &std::path::Path) {
    env.set("HOME", home);
    env.remove("CALICHE_NO_KEYRING");
    env.set("CALICHE_USE_KEYRING", "1");
}

#[tokio::test]
#[serial_test::serial(env)]
async fn creds_set_key_resolves_for_the_provider_and_env_stays_unset() {
    let home = tempfile::tempdir().expect("tempdir");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { EnvGuard::acquire() };
    isolated_keychain(&mut env, home.path());
    env.remove("COHERE_API_KEY");
    env.remove("API_KEY");

    let result =
        SlashCommandExecutor::handle_creds(&args(&["set", "cohere", "cohere-test-key"]), None)
            .await
            .expect("creds set");

    match result {
        SlashCommandResult::StatusMessage(message) => assert_eq!(
            message,
            "Stored API key for 'cohere' in global keychain.\n\
             Run '/provider cohere' to start using it in this session."
        ),
        other => panic!("unexpected /creds set result: {other:?}"),
    }
    assert_eq!(
        credential_env::var("COHERE_API_KEY").as_deref(),
        Ok("cohere-test-key")
    );
    assert!(crate::credentials::provider_has_env_credentials("cohere"));
    let resolved = EnvironmentCredentialProvider::new()
        .get_credentials("cohere")
        .await
        .expect("cohere credentials");
    assert_eq!(resolved.as_api_key(), Some("cohere-test-key"));
    assert_eq!(std::env::var("COHERE_API_KEY"), Err(VarError::NotPresent));
}

#[tokio::test]
#[serial_test::serial(env)]
async fn gateway_twilio_values_resolve_and_env_stays_unset() {
    let home = tempfile::tempdir().expect("tempdir");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { EnvGuard::acquire() };
    isolated_keychain(&mut env, home.path());
    for key in [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM",
        "TWILIO_PHONE_NUMBER",
    ] {
        env.remove(key);
    }
    let manager = SecureCredentialManager::new_with_scope(None)
        .await
        .expect("credential manager");
    manager
        .set_api_key("twilio_account_sid", "AC-test-sid")
        .await
        .expect("store sid");
    manager
        .set_api_key("twilio_auth_token", "twilio-test-token")
        .await
        .expect("store token");
    manager
        .set_api_key("twilio_from", "+15550100")
        .await
        .expect("store from");

    // An unknown action loads the credentials, then stops before any network call.
    let result = SlashCommandExecutor::handle_creds(
        &args(&["gateway", "twilio", "webhook", "unknown-action"]),
        None,
    )
    .await
    .expect("gateway twilio");

    assert!(
        matches!(&result, SlashCommandResult::Error(message) if message.starts_with("Unknown /gateway twilio subcommand.")),
        "unexpected result: {result:?}"
    );
    for (key, value) in [
        ("TWILIO_ACCOUNT_SID", "AC-test-sid"),
        ("TWILIO_AUTH_TOKEN", "twilio-test-token"),
        ("TWILIO_FROM", "+15550100"),
        ("TWILIO_PHONE_NUMBER", "+15550100"),
    ] {
        assert_eq!(credential_env::var(key).as_deref(), Ok(value), "{key}");
        assert_eq!(std::env::var(key), Err(VarError::NotPresent), "{key}");
    }
}

#[tokio::test]
#[serial_test::serial(env)]
async fn shell_value_beats_the_keychain_at_startup() {
    let home = tempfile::tempdir().expect("tempdir");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { EnvGuard::acquire() };
    isolated_keychain(&mut env, home.path());
    env.set("PERPLEXITY_API_KEY", "shell-perplexity");
    let manager = SecureCredentialManager::new_with_scope(None)
        .await
        .expect("credential manager");
    manager
        .set_api_key("perplexity", "keychain-perplexity")
        .await
        .expect("store key");
    let shell_keys: HashSet<String> = HashSet::from(["PERPLEXITY_API_KEY".to_string()]);

    crate::bootstrap::load_startup_credentials(&manager, &shell_keys).await;

    assert_eq!(
        credential_env::var("PERPLEXITY_API_KEY").as_deref(),
        Ok("shell-perplexity")
    );
    assert!(
        !credential_env::snapshot()
            .iter()
            .any(|(name, _)| name == "PERPLEXITY_API_KEY")
    );
}

#[tokio::test]
#[serial_test::serial(env)]
async fn directory_scope_beats_global_keychain_at_startup() {
    let home = tempfile::tempdir().expect("tempdir");
    let project = tempfile::tempdir().expect("project dir");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { EnvGuard::acquire() };
    isolated_keychain(&mut env, home.path());
    env.remove("TELEGRAM_BOT_TOKEN");
    let manager = SecureCredentialManager::new_with_scope(Some(project.path()))
        .await
        .expect("credential manager");
    manager
        .set_api_key("telegram_bot_token", "global-telegram")
        .await
        .expect("store global");
    let scope_hash = manager.current_scope_hash().to_string();
    manager
        .set_api_key_in_scope("telegram_bot_token", "scoped-telegram", &scope_hash)
        .await
        .expect("store scoped");

    crate::bootstrap::load_startup_credentials(&manager, &HashSet::new()).await;

    assert_eq!(
        credential_env::var("TELEGRAM_BOT_TOKEN").as_deref(),
        Ok("scoped-telegram")
    );
    assert_eq!(
        std::env::var("TELEGRAM_BOT_TOKEN"),
        Err(VarError::NotPresent)
    );
}

#[tokio::test]
#[serial_test::serial(env)]
async fn keychain_replaces_an_env_file_value_at_startup() {
    let home = tempfile::tempdir().expect("tempdir");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { EnvGuard::acquire() };
    isolated_keychain(&mut env, home.path());
    // In the process env but not in the shell snapshot: it came from --env-file.
    env.set("AZURE_OPENAI_API_KEY", "env-file-azure");
    let manager = SecureCredentialManager::new_with_scope(None)
        .await
        .expect("credential manager");
    manager
        .set_api_key("azure-openai", "keychain-azure")
        .await
        .expect("store key");

    crate::bootstrap::load_startup_credentials(&manager, &HashSet::new()).await;

    assert_eq!(
        credential_env::var("AZURE_OPENAI_API_KEY").as_deref(),
        Ok("keychain-azure")
    );
    assert_eq!(
        std::env::var("AZURE_OPENAI_API_KEY").as_deref(),
        Ok("env-file-azure")
    );
}

#[tokio::test]
#[serial_test::serial(env)]
async fn creds_set_keeps_a_value_already_in_the_environment() {
    let home = tempfile::tempdir().expect("tempdir");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { EnvGuard::acquire() };
    isolated_keychain(&mut env, home.path());
    env.set("DISCORD_BOT_TOKEN", "shell-discord");

    let result = SlashCommandExecutor::handle_creds(
        &args(&["set", "discord_bot_token", "keychain-discord"]),
        None,
    )
    .await
    .expect("creds set");

    assert!(
        matches!(result, SlashCommandResult::StatusMessage(_)),
        "unexpected result: {result:?}"
    );
    assert_eq!(
        credential_env::var("DISCORD_BOT_TOKEN").as_deref(),
        Ok("shell-discord")
    );
    assert!(
        !credential_env::snapshot()
            .iter()
            .any(|(name, _)| name == "DISCORD_BOT_TOKEN")
    );
    // The keychain still holds the new value for the next launch.
    let manager = SecureCredentialManager::new_with_scope(None)
        .await
        .expect("credential manager");
    assert_eq!(
        manager
            .get_scope_credentials("global")
            .get("discord_bot_token")
            .map(String::as_str),
        Some("keychain-discord")
    );
}
