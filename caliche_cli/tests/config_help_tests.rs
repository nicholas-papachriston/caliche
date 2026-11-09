//! `caliche config <action> --help` prints help without looking up credentials.
//!
//! These spawn the package binary with a cleared environment and a temp HOME,
//! so no provider key is set and the user's own config stays out of the result.

use caliche_contracts::child_env::SYSTEMROOT;
use std::process::{Command, Output};

fn help(args: &[&str]) -> Output {
    let home = tempfile::tempdir().expect("temp home");
    let mut command = Command::new(env!("CARGO_BIN_EXE_caliche"));
    command
        .args(args)
        .arg("--help")
        .current_dir(home.path())
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("APPDATA", home.path().join("appdata"))
        .env("LOCALAPPDATA", home.path().join("localappdata"))
        .env("XDG_CONFIG_HOME", home.path().join("xdg-config"))
        .env("XDG_DATA_HOME", home.path().join("xdg-data"));
    if let Some(root) = std::env::var_os(SYSTEMROOT) {
        command.env(SYSTEMROOT, root);
    }
    command.output().expect("spawn caliche")
}

#[test]
fn config_set_help_prints_usage_without_keys() {
    let output = help(&["config", "set"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "stderr: {stderr}");
    assert!(
        stdout.starts_with(
            "Set an API key for a provider\n\nUsage: caliche config set [OPTIONS] <PROVIDER>\n"
        ),
        "stdout: {stdout}"
    );
    assert!(!stderr.contains("No credentials found"), "stderr: {stderr}");
}

#[test]
fn every_config_action_help_prints_usage_without_keys() {
    for action in [
        "setup", "set", "get", "remove", "list", "default", "model", "aws", "show", "schema",
        "path", "export", "vault", "gateway",
    ] {
        let output = help(&["config", action]);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(0), "{action} stderr: {stderr}");
        assert!(
            stdout.contains(&format!("Usage: caliche config {action}")),
            "{action} stdout: {stdout}"
        );
        assert!(!stderr.contains("Error"), "{action} stderr: {stderr}");
    }
}
