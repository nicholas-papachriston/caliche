//! Shared fixtures for API integration tests.
//!
//! `hold` points `CALICHE_CONFIG_DIR` at one temp directory for this test
//! process. The directory stays for the process lifetime, so a later test
//! cannot fall back to the real user data directory. Every test in a binary
//! that includes this module must be `#[serial]`: the first `hold` changes
//! the process environment.

use std::sync::{Arc, Mutex, OnceLock};

use caliche::AgentBuilder;
use caliche::llm::ProviderType;
use caliche::testing::MockProvider;
use caliche_server::{RouterOptions, create_router};

struct InstalledConfigDir {
    _dir: tempfile::TempDir,
}

static CONFIG_DIR: OnceLock<InstalledConfigDir> = OnceLock::new();
static CONFIG_DIR_LOCK: Mutex<()> = Mutex::new(());

/// Keeps the process config directory installed while the test runs.
pub struct ConfigDirGuard;

/// Install the process config directory before router bootstrap.
pub fn hold() -> ConfigDirGuard {
    let _lock = CONFIG_DIR_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    CONFIG_DIR.get_or_init(|| {
        let dir = tempfile::tempdir().expect("CALICHE_CONFIG_DIR temp directory");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: every test in each binary that includes this harness is
        // `#[serial]`, so no other test thread reads or writes the
        // environment during this call.
        unsafe {
            caliche_test_env::set_for_process(caliche_server::paths::CONFIG_DIR_ENV, dir.path());
        }
        InstalledConfigDir { _dir: dir }
    });
    ConfigDirGuard
}

/// Router whose chat path uses [`MockProvider`] and does not call a live model.
pub async fn mock_router() -> axum::Router {
    let _config = hold();
    let mock = MockProvider::new(vec!["Hello from mock!".to_string()]);
    let service = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("mock-model")
        .provider_instance(Arc::new(mock))
        .build()
        .await
        .expect("mock service should build");
    let options = RouterOptions {
        shared_service: Some(Arc::new(tokio::sync::RwLock::new(service))),
        ..RouterOptions::default()
    };
    create_router(
        ProviderType::OpenAI,
        "mock-model".to_string(),
        Some(options),
    )
    .await
    .expect("router should build")
}
