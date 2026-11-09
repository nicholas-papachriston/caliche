use caliche::plugins::http_policy::HttpPolicy;
use serial_test::serial;

#[test]
fn blocks_loopback_after_interpolation() {
    let policy = HttpPolicy::new(None);
    let err = policy
        .validate_resolved_url("http://127.0.0.1/latest/meta-data/")
        .expect_err("loopback should be blocked");
    assert!(
        err.to_string().contains("blocked"),
        "expected blocked host error, got: {err}"
    );
}

#[test]
fn enforces_allowlist_on_resolved_host() {
    let policy = HttpPolicy::new(Some(vec!["example.com".to_string()]));
    policy
        .validate_resolved_url("https://example.com/v1")
        .expect("allowed host");
    let err = policy
        .validate_resolved_url("https://notexample.com/v1")
        .expect_err("host not on allowlist");
    assert!(
        err.to_string().contains("allowlist"),
        "expected allowlist error, got: {err}"
    );
}

#[test]
fn blocks_mapped_ipv6_and_cgnat() {
    let policy = HttpPolicy::new(None);
    let mapped = policy
        .validate_resolved_url("http://[::ffff:127.0.0.1]/")
        .expect_err("mapped loopback");
    assert!(mapped.to_string().contains("blocked"));
    let cgnat = policy
        .validate_resolved_url("http://100.64.0.1/")
        .expect_err("cgnat");
    assert!(cgnat.to_string().contains("blocked"));
}

#[test]
fn rejects_non_http_scheme() {
    let policy = HttpPolicy::new(None);
    let err = policy
        .validate_resolved_url("file:///etc/passwd")
        .expect_err("file scheme");
    assert!(
        err.to_string().contains("scheme"),
        "expected scheme error, got: {err}"
    );
}

#[test]
#[serial]
#[expect(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "test sets and restores process env vars"
)]
fn allowlist_from_env_parses_comma_separated_hosts() {
    // SAFETY: test runs serially; env mutation is isolated to this process.
    unsafe {
        std::env::set_var("PLUGIN_HTTP_ALLOWLIST", " example.com , cdn.example.com , ");
    }
    let list = HttpPolicy::allowlist_from_env().expect("non-empty allowlist from env");
    assert_eq!(list, vec!["example.com", "cdn.example.com"]);

    let policy = HttpPolicy::new(HttpPolicy::allowlist_from_env());
    policy
        .validate_resolved_url("https://example.com/health")
        .expect("env allowlist host should pass");

    unsafe {
        std::env::remove_var("PLUGIN_HTTP_ALLOWLIST");
    }
    assert!(
        HttpPolicy::allowlist_from_env().is_none(),
        "empty env var should yield no allowlist"
    );
}

#[test]
#[serial]
#[expect(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "test sets and restores process env vars"
)]
fn from_env_requires_allowlist_in_production() {
    let previous_env = std::env::var("CALICHE_ENV").ok();
    let previous_allow = std::env::var("PLUGIN_HTTP_ALLOWLIST").ok();
    let previous_insecure = std::env::var("CALICHE_ALLOW_INSECURE_PRODUCTION").ok();
    unsafe {
        std::env::set_var("CALICHE_ENV", "production");
        std::env::remove_var("PLUGIN_HTTP_ALLOWLIST");
        std::env::remove_var("CALICHE_ALLOW_INSECURE_PRODUCTION");
    }
    let err = HttpPolicy::from_env().expect_err("production allowlist");
    assert!(err.to_string().contains("PLUGIN_HTTP_ALLOWLIST"));
    unsafe {
        std::env::set_var("PLUGIN_HTTP_ALLOWLIST", "*");
    }
    HttpPolicy::from_env().expect("star allowlist");
    unsafe {
        match previous_env {
            Some(value) => std::env::set_var("CALICHE_ENV", value),
            None => std::env::remove_var("CALICHE_ENV"),
        }
        match previous_allow {
            Some(value) => std::env::set_var("PLUGIN_HTTP_ALLOWLIST", value),
            None => std::env::remove_var("PLUGIN_HTTP_ALLOWLIST"),
        }
        match previous_insecure {
            Some(value) => std::env::set_var("CALICHE_ALLOW_INSECURE_PRODUCTION", value),
            None => std::env::remove_var("CALICHE_ALLOW_INSECURE_PRODUCTION"),
        }
    }
}
