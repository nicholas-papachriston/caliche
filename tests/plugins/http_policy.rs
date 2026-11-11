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
fn allowlist_from_env_parses_comma_separated_hosts() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("PLUGIN_HTTP_ALLOWLIST", " example.com , cdn.example.com , ");
    let list = HttpPolicy::allowlist_from_env().expect("non-empty allowlist from env");
    assert_eq!(list, vec!["example.com", "cdn.example.com"]);

    let policy = HttpPolicy::new(HttpPolicy::allowlist_from_env());
    policy
        .validate_resolved_url("https://example.com/health")
        .expect("env allowlist host should pass");

    env.remove("PLUGIN_HTTP_ALLOWLIST");
    assert!(
        HttpPolicy::allowlist_from_env().is_none(),
        "empty env var should yield no allowlist"
    );
}

#[test]
#[serial]
fn from_env_requires_allowlist_in_production() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_ENV", "production");
    env.remove("PLUGIN_HTTP_ALLOWLIST");
    env.remove("CALICHE_ALLOW_INSECURE_PRODUCTION");
    let err = HttpPolicy::from_env().expect_err("production allowlist");
    assert!(err.to_string().contains("PLUGIN_HTTP_ALLOWLIST"));
    env.set("PLUGIN_HTTP_ALLOWLIST", "*");
    HttpPolicy::from_env().expect("star allowlist");
}
