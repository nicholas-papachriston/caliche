//! Integration tests for gateway cluster registration security.

use axum::http::{HeaderMap, HeaderValue};
use caliche_server::handlers::gateway_validation::{
    CLUSTER_REGISTER_HEADER, authorize_cluster_register, validate_cluster_base_url,
};
use serial_test::serial;

fn clear_env(env: &mut caliche_test_env::EnvGuard) {
    env.remove("CALICHE_CLUSTER_HOST_ALLOWLIST");
    env.remove("CALICHE_CLUSTER_ALLOW_HTTP");
    env.remove("CALICHE_CLUSTER_ALLOW_PRIVATE");
    env.remove("CALICHE_CLUSTER_REGISTER_TOKEN");
}

#[test]
#[serial]
fn rejects_registration_without_token_or_role() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    let headers = HeaderMap::new();
    assert!(authorize_cluster_register(&headers, None).is_err());
}

#[test]
#[serial]
fn accepts_valid_register_token() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    env.set("CALICHE_CLUSTER_REGISTER_TOKEN", "super-s3cret");
    let mut headers = HeaderMap::new();
    headers.insert(
        CLUSTER_REGISTER_HEADER,
        HeaderValue::from_static("super-s3cret"),
    );
    authorize_cluster_register(&headers, None).expect("token should authorize");
}

#[test]
#[serial]
fn rejects_wrong_scheme_without_opt_in() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
    let err =
        validate_cluster_base_url("ftp://trusted.example/").expect_err("ftp must be rejected");
    assert!(err.to_string().contains("unsupported scheme"));
}

#[test]
#[serial]
fn rejects_private_ip_without_opt_in() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "192.168.1.1");
    env.set("CALICHE_CLUSTER_ALLOW_HTTP", "true");
    let err = validate_cluster_base_url("http://192.168.1.1:9090/")
        .expect_err("private ip must be rejected");
    assert!(err.to_string().contains("ALLOW_PRIVATE"));
}

#[test]
#[serial]
fn allows_public_https_in_allowlist() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
    validate_cluster_base_url("https://trusted.example:8443/agent")
        .expect("should accept trusted https");
}

#[test]
#[serial]
fn rejects_non_allowlisted_host() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
    let err = validate_cluster_base_url("https://evil.example/")
        .expect_err("non-allowlisted host must be rejected");
    assert!(err.to_string().contains("allowlist"));
}

#[test]
#[serial]
fn rejects_registration_when_host_allowlist_unset() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_env(&mut env);
    let err = validate_cluster_base_url("https://anything.example/")
        .expect_err("empty allowlist must reject");
    assert!(err.to_string().contains("HOST_ALLOWLIST"));
}
