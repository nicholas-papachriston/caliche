//! SecurityPolicy fail-closed startup validation tests.

use caliche_server::security::{
    BindHostKind, OpenBindPosture, SecurityPolicy, classify_bind_host, env_bool,
    open_unspecified_bind_posture,
};
use serial_test::serial;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tempfile::NamedTempFile;
use tracing::field::Visit;
use tracing::span::{Attributes, Id, Record};
use tracing::{Metadata, Subscriber};

fn clear_security_env() {
    unsafe {
        for key in [
            "JWT_ENABLED",
            "JWT_ALGORITHM",
            "JWT_PUBLIC_KEY",
            "JWT_PUBLIC_KEY_PATH",
            "JWT_JWKS_URL",
            "JWT_AUDIENCE",
            "JWT_ISSUER",
            "CALICHE_REQUIRE_AUTH",
            "CALICHE_ENV",
            "CALICHE_ALLOW_INSECURE_PRODUCTION",
            "CALICHE_TRUSTED_PROXY",
            "CALICHE_TRUSTED_PROXY_CIDRS",
            "CORS_PERMISSIVE",
            "API_HOST",
            "BIND_ADDRESS",
            "RUST_ENV",
        ] {
            std::env::remove_var(key);
        }
    }
}

#[test]
#[serial]
fn env_bool_parses_truthy_values() {
    unsafe {
        std::env::set_var("CALICHE_TEST_BOOL", "true");
        assert!(env_bool("CALICHE_TEST_BOOL"));
        std::env::set_var("CALICHE_TEST_BOOL", "YES");
        assert!(env_bool("CALICHE_TEST_BOOL"));
        std::env::set_var("CALICHE_TEST_BOOL", "0");
        assert!(!env_bool("CALICHE_TEST_BOOL"));
        std::env::remove_var("CALICHE_TEST_BOOL");
    }
    assert!(!env_bool("CALICHE_TEST_BOOL_MISSING"));
}

#[test]
#[serial]
fn require_auth_without_jwt_fails_closed() {
    clear_security_env();
    unsafe {
        std::env::set_var("CALICHE_REQUIRE_AUTH", "true");
    }

    let err = SecurityPolicy::from_env().expect_err("require_auth without JWT should fail");
    assert!(
        err.to_string()
            .contains("CALICHE_REQUIRE_AUTH is set but JWT_ENABLED is false")
    );
}

#[test]
#[serial]
fn jwt_enabled_without_audience_fails_closed() {
    clear_security_env();
    unsafe {
        std::env::set_var("JWT_ENABLED", "true");
        std::env::set_var("JWT_ALGORITHM", "HS256");
        std::env::set_var("JWT_PUBLIC_KEY", "test-secret-key");
    }

    let err = SecurityPolicy::from_env().expect_err("JWT without audience should fail");
    assert!(err.to_string().contains("JWT_AUDIENCE"));
}

#[test]
#[serial]
fn jwt_enabled_without_key_source_fails_closed() {
    clear_security_env();
    unsafe {
        std::env::set_var("JWT_ENABLED", "true");
    }

    let err = SecurityPolicy::from_env().expect_err("JWT without keys should fail");
    assert!(err.to_string().contains("no key source provided"));
}

#[test]
#[serial]
fn jwt_enabled_with_secret_initializes() {
    clear_security_env();
    unsafe {
        std::env::set_var("JWT_ENABLED", "true");
        std::env::set_var("JWT_ALGORITHM", "HS256");
        std::env::set_var("JWT_PUBLIC_KEY", "test-secret-key");
        std::env::set_var("JWT_AUDIENCE", "caliche");
        std::env::set_var("JWT_ISSUER", "caliche");
    }

    let policy = SecurityPolicy::from_env().expect("JWT with inline secret should initialize");
    assert!(policy.jwt_enabled);
    assert!(policy.jwt_state.is_some());
    assert!(!policy.require_auth);
    assert!(policy.session_auth_required());
}

#[test]
#[serial]
fn jwt_init_failure_surfaces_as_policy_error() {
    clear_security_env();
    let key_dir = tempfile::tempdir().expect("temp key dir");
    unsafe {
        std::env::set_var("JWT_ENABLED", "true");
        std::env::set_var("JWT_ALGORITHM", "RS256");
        std::env::set_var(
            "JWT_PUBLIC_KEY_PATH",
            key_dir.path().to_string_lossy().to_string(),
        );
        std::env::set_var("JWT_AUDIENCE", "caliche");
        std::env::set_var("JWT_ISSUER", "caliche");
    }

    let err = SecurityPolicy::from_env().expect_err("directory key path should fail closed");
    assert!(
        err.to_string()
            .contains("JWT authentication failed to initialize")
    );
}

#[test]
#[serial]
fn session_auth_required_when_require_auth_set() {
    clear_security_env();
    let key_file = NamedTempFile::new().expect("temp key file");
    fs::write(key_file.path(), "test-secret-key").expect("write key file");
    unsafe {
        std::env::set_var("JWT_ENABLED", "true");
        std::env::set_var("JWT_ALGORITHM", "HS256");
        std::env::set_var(
            "JWT_PUBLIC_KEY_PATH",
            key_file.path().to_string_lossy().to_string(),
        );
        std::env::set_var("CALICHE_REQUIRE_AUTH", "true");
        std::env::set_var("JWT_AUDIENCE", "caliche");
        std::env::set_var("JWT_ISSUER", "caliche");
    }

    let policy = SecurityPolicy::from_env().expect("policy should initialize");
    assert!(policy.require_auth);
    assert!(policy.session_auth_required());
}

#[test]
#[serial]
fn trusted_proxy_without_cidrs_fails_closed() {
    clear_security_env();
    unsafe {
        std::env::set_var("CALICHE_TRUSTED_PROXY", "true");
    }
    let err = SecurityPolicy::from_env().expect_err("trusted proxy without CIDRs");
    assert!(err.to_string().contains("TRUSTED_PROXY_CIDRS"));
}

#[test]
#[serial]
fn cors_permissive_fails_closed_in_production() {
    clear_security_env();
    unsafe {
        std::env::set_var("CALICHE_ENV", "production");
        std::env::set_var("CORS_PERMISSIVE", "true");
        std::env::remove_var("CALICHE_ALLOW_INSECURE_PRODUCTION");
    }
    let err = caliche_server::security::cors_permissive_from_env()
        .expect_err("permissive CORS in production");
    assert!(err.to_string().contains("CORS_PERMISSIVE"));
}

#[test]
fn unspecified_bind_hosts_are_classified() {
    assert_eq!(classify_bind_host("0.0.0.0"), BindHostKind::Unspecified);
    assert_eq!(classify_bind_host("::"), BindHostKind::Unspecified);
    assert_eq!(classify_bind_host("[::]"), BindHostKind::Unspecified);
    assert_eq!(classify_bind_host(" 0.0.0.0 "), BindHostKind::Unspecified);
    assert_eq!(classify_bind_host("127.0.0.1"), BindHostKind::Specific);
    assert_eq!(classify_bind_host("::1"), BindHostKind::Specific);
    assert_eq!(classify_bind_host("localhost"), BindHostKind::Specific);
}

#[test]
fn open_unspecified_bind_warns_only_without_production_or_require_auth() {
    assert_eq!(
        open_unspecified_bind_posture("0.0.0.0", false, false),
        OpenBindPosture::Warn
    );
    assert_eq!(
        open_unspecified_bind_posture("::", false, false),
        OpenBindPosture::Warn
    );
    assert_eq!(
        open_unspecified_bind_posture("127.0.0.1", false, false),
        OpenBindPosture::Quiet
    );
    assert_eq!(
        open_unspecified_bind_posture("0.0.0.0", true, false),
        OpenBindPosture::Quiet
    );
    assert_eq!(
        open_unspecified_bind_posture("0.0.0.0", false, true),
        OpenBindPosture::Quiet
    );
}

#[test]
#[serial]
fn unspecified_api_host_warns_and_still_starts() {
    clear_security_env();
    unsafe {
        std::env::set_var("API_HOST", "0.0.0.0");
    }

    let hit = capture_unspecified_bind_warn(|| {
        let policy = SecurityPolicy::from_env().expect("open defaults still start");
        assert!(!policy.jwt_enabled);
        assert!(!policy.require_auth);
    });
    assert!(hit, "API_HOST=0.0.0.0 should warn when auth is open");
}

#[test]
#[serial]
fn unspecified_ipv6_bind_address_warns() {
    clear_security_env();
    unsafe {
        std::env::set_var("BIND_ADDRESS", "::");
    }

    let hit = capture_unspecified_bind_warn(|| {
        SecurityPolicy::from_env().expect("open defaults still start");
    });
    assert!(hit, "BIND_ADDRESS=:: should warn when auth is open");
}

#[test]
#[serial]
fn loopback_api_host_does_not_warn() {
    clear_security_env();
    unsafe {
        std::env::set_var("API_HOST", "127.0.0.1");
    }

    let hit = capture_unspecified_bind_warn(|| {
        SecurityPolicy::from_env().expect("loopback open API still starts");
    });
    assert!(!hit, "loopback API_HOST must stay quiet");
}

#[test]
#[serial]
fn rust_env_does_not_select_the_production_profile() {
    clear_security_env();
    unsafe {
        std::env::set_var("RUST_ENV", "production");
    }
    SecurityPolicy::from_env().expect("production profile reads CALICHE_ENV");
    unsafe {
        std::env::remove_var("RUST_ENV");
    }
}

#[test]
#[serial]
fn production_profile_still_fails_closed_when_api_host_is_unspecified() {
    clear_security_env();
    unsafe {
        std::env::set_var("CALICHE_ENV", "production");
        std::env::set_var("API_HOST", "0.0.0.0");
    }

    let hit = capture_unspecified_bind_warn(|| {
        let err = SecurityPolicy::from_env().expect_err("production without JWT");
        assert!(err.to_string().contains("JWT_ENABLED"));
    });
    assert!(
        !hit,
        "production fail-closed must not be replaced by the bind warning"
    );
}

struct UnspecifiedBindWarn {
    hit: Arc<AtomicBool>,
}

struct MessageProbe {
    hit: bool,
}

impl Visit for MessageProbe {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.record_text(field.name(), &format!("{value:?}"));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.record_text(field.name(), value);
    }
}

impl MessageProbe {
    fn record_text(&mut self, name: &str, value: &str) {
        if name == "message" && value.contains("unspecified bind") {
            self.hit = true;
        }
    }
}

impl Subscriber for UnspecifiedBindWarn {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target() == "caliche_server::security"
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut probe = MessageProbe { hit: false };
        event.record(&mut probe);
        if probe.hit {
            self.hit.store(true, Ordering::SeqCst);
        }
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

fn capture_unspecified_bind_warn(body: impl FnOnce()) -> bool {
    let hit = Arc::new(AtomicBool::new(false));
    let subscriber = UnspecifiedBindWarn {
        hit: Arc::clone(&hit),
    };
    tracing::subscriber::with_default(subscriber, body);
    hit.load(Ordering::SeqCst)
}
