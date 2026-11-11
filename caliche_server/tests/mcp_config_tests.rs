//! MCP URL and stdio command allowlist validation tests.

use caliche_server::handlers::mcp_config::validate_stdio_command;
use caliche_server::handlers::mcp_policy::validate_mcp_remote_url;

fn clear_mcp_env(env: &mut caliche_test_env::EnvGuard) {
    env.remove("CALICHE_MCP_COMMAND_ALLOWLIST");
}

#[test]
#[serial_test::serial]
fn blocks_private_mcp_url() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_mcp_env(&mut env);
    let err =
        validate_mcp_remote_url("http://127.0.0.1/mcp").expect_err("loopback should be blocked");
    assert!(err.to_string().contains("blocked"));
}

#[test]
#[serial_test::serial]
fn blocks_localhost_mcp_url() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_mcp_env(&mut env);
    let err =
        validate_mcp_remote_url("https://localhost/mcp").expect_err("localhost should be blocked");
    assert!(err.to_string().contains("blocked"));
}

#[test]
#[serial_test::serial]
fn allows_public_mcp_url() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_mcp_env(&mut env);
    validate_mcp_remote_url("https://example.com/sse").expect("public host should pass");
}

#[test]
#[serial_test::serial]
fn rejects_stdio_without_allowlist() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_mcp_env(&mut env);
    let err = validate_stdio_command("npx").expect_err("allowlist should be required");
    assert!(err.to_string().contains("CALICHE_MCP_COMMAND_ALLOWLIST"));
}

#[test]
#[serial_test::serial]
fn accepts_allowlisted_stdio_command() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_mcp_env(&mut env);
    env.set("CALICHE_MCP_COMMAND_ALLOWLIST", "npx,uvx");
    validate_stdio_command("npx").expect("basename allowlist should pass");
    let err = validate_stdio_command("/usr/local/bin/uvx")
        .expect_err("path must be listed exactly, not by basename");
    assert!(
        err.to_string().contains("uses a path"),
        "unexpected error: {err}"
    );
    let err = validate_stdio_command("/tmp/npx").expect_err("arbitrary npx path");
    assert!(err.to_string().contains("uses a path"));
    let err = validate_stdio_command("./npx").expect_err("relative npx");
    assert!(err.to_string().contains("uses a path"));
    let err = validate_stdio_command("bash").expect_err("non-allowlisted command should fail");
    assert!(
        err.to_string()
            .contains("not in CALICHE_MCP_COMMAND_ALLOWLIST")
    );
}

#[test]
#[serial_test::serial]
fn accepts_exact_path_when_allowlisted() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    clear_mcp_env(&mut env);
    env.set("CALICHE_MCP_COMMAND_ALLOWLIST", "/usr/bin/npx,uvx");
    validate_stdio_command("/usr/bin/npx").expect("exact path allowlist should pass");
    validate_stdio_command("uvx").expect("bare name still matches");
    let err = validate_stdio_command("/usr/local/bin/npx").expect_err("different path");
    assert!(err.to_string().contains("uses a path"));
}
