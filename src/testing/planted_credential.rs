//! Re-run one test in a child process that holds a credential-shaped
//! variable, so a spawn path can be checked for passing it on.
//!
//! Tests may not mutate this process's environment, so the variable is set
//! on a fresh copy of the test binary instead.

/// Credential-shaped variable planted in the re-run test process.
pub(crate) const PLANTED_CREDENTIAL: &str = "CALICHE_SCRUB_PROBE_TOKEN";

/// A `sh` snippet that prints the planted variable, or `unset`.
pub(crate) const PRINT_PLANTED: &str = "printf %s \"${CALICHE_SCRUB_PROBE_TOKEN-unset}\"";

/// Run `test` (its full path) again with [`PLANTED_CREDENTIAL`] set and
/// assert that it passed.
///
/// Returns `true` in the outer process, which should then return. Returns
/// `false` inside the re-run, where the test body runs.
pub(crate) fn rerun_with_planted_credential(test: &str) -> bool {
    rerun_in_child(test, &[], None)
}

/// Run `test` again in a child process with extra variables (for example
/// `PATH`) and, when given, a working directory. The child also holds
/// [`PLANTED_CREDENTIAL`], which marks it as the re-run.
pub(crate) fn rerun_in_child(
    test: &str,
    envs: &[(&str, &std::ffi::OsStr)],
    cwd: Option<&std::path::Path>,
) -> bool {
    if std::env::var_os(PLANTED_CREDENTIAL).is_some() {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().expect("test binary"));
    command
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(PLANTED_CREDENTIAL, "planted")
        .envs(envs.iter().copied());
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output().expect("re-run test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}
