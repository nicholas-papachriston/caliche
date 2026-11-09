use caliche_sandbox::config::CPU_CAP_REQUIRES_WINDOWS_JOB;
use caliche_sandbox::{
    NamedPolicy, NetworkPolicy, SandboxBackendType, SandboxMode, policy_to_config,
    validate_sandbox_config,
};
use std::path::Path;
use std::str::FromStr;

#[test]
fn restricted_network_rejected() {
    let mut config = policy_to_config(
        NamedPolicy::NetworkRestricted,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    config.mode = SandboxMode::Off;
    config.network = NetworkPolicy::Restricted {
        allowed_hosts: vec!["example.com".into()],
    };
    let err = validate_sandbox_config(&config).expect_err("restricted egress");
    assert!(err.to_string().contains("not enforced"));
}

#[test]
fn restricted_network_empty_allowlist_rejected() {
    let mut config = policy_to_config(
        NamedPolicy::NetworkRestricted,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    config.mode = SandboxMode::Off;
    config.network = NetworkPolicy::Restricted {
        allowed_hosts: vec![],
    };
    let err = validate_sandbox_config(&config).expect_err("empty allowlist");
    assert!(err.to_string().contains("not enforced"));
}

#[test]
fn network_restricted_policy_builds_restricted_network_config() {
    let config = policy_to_config(
        NamedPolicy::NetworkRestricted,
        Path::new("/tmp"),
        Some(&["example.com".into()]),
        SandboxBackendType::Noop,
    );
    assert!(matches!(config.network, NetworkPolicy::Restricted { .. }));
    let err = validate_sandbox_config(&config).expect_err("validation rejects restricted egress");
    assert!(err.to_string().contains("not enforced"));
}

#[test]
fn workspace_write_policy_allows_validation() {
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    validate_sandbox_config(&config).expect("workspace-write has no restricted egress");
}

#[test]
fn cpu_rate_hard_cap_defaults_to_off() {
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    assert_eq!(config.resource_limits.cpu_rate_hard_cap_percent, None);
    validate_sandbox_config(&config).expect("default cap is off");
}

#[test]
fn cpu_rate_outside_one_through_one_hundred_fails_validation() {
    let mut config = policy_to_config(
        NamedPolicy::ReadOnly,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    config.resource_limits.cpu_rate_hard_cap_percent = Some(0);
    let err = validate_sandbox_config(&config).expect_err("0");
    assert!(err.to_string().contains("1 through 100"));
    config.resource_limits.cpu_rate_hard_cap_percent = Some(100);
    let err = validate_sandbox_config(&config).expect_err("noop ignores a cap");
    assert_eq!(err.to_string(), CPU_CAP_REQUIRES_WINDOWS_JOB);
}

#[test]
fn create_backend_rejects_a_cpu_cap_on_noop() {
    let mut config = policy_to_config(
        NamedPolicy::ReadOnly,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    config.resource_limits.cpu_rate_hard_cap_percent = Some(20);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let err = runtime.block_on(caliche_sandbox::create_backend(&config));
    let err = err.expect_err("noop must not ignore a cpu cap");
    assert_eq!(err.to_string(), CPU_CAP_REQUIRES_WINDOWS_JOB);
    config.resource_limits.cpu_rate_hard_cap_percent = Some(0);
    let err = runtime.block_on(caliche_sandbox::create_backend(&config));
    let err = err.expect_err("0 is not a cap");
    assert!(err.to_string().contains("1 through 100"));
}

#[test]
fn cpu_rate_hard_cap_is_accepted_on_the_windows_job_backend() {
    let mut config = policy_to_config(
        NamedPolicy::ReadOnly,
        Path::new("/tmp"),
        None,
        SandboxBackendType::WindowsJob {
            policy: NamedPolicy::ReadOnly,
        },
    );
    config.resource_limits.cpu_rate_hard_cap_percent = Some(100);
    validate_sandbox_config(&config).expect("windows job accepts 100");
}

#[test]
fn named_policy_from_str_accepts_kebab_and_snake_case() {
    assert_eq!(
        NamedPolicy::from_str("read-only").expect("read-only"),
        NamedPolicy::ReadOnly
    );
    assert_eq!(
        NamedPolicy::from_str("workspace_write").expect("workspace_write"),
        NamedPolicy::WorkspaceWrite
    );
    assert_eq!(
        NamedPolicy::from_str("networkrestricted").expect("networkrestricted"),
        NamedPolicy::NetworkRestricted
    );
    assert!(NamedPolicy::from_str("unknown-policy").is_err());
}
