//! Run curl without putting header values on the process command line.
//!
//! `curl --config -` reads header lines from stdin. `ps` then shows `--config -`
//! and not the bearer token.

use std::io::Write;
use std::process::{Command, Stdio};

/// Execute curl. `headers` are sent as config-file `header` lines on stdin.
///
/// # Errors
///
/// Returns the spawn or stdin write error. The curl exit status is on the
/// returned [`std::process::Output`].
pub(crate) fn run_curl(
    args: &[&str],
    headers: &[(String, String)],
) -> std::io::Result<std::process::Output> {
    let mut command = Command::new("curl");
    command.args(args);
    let config = header_config(headers);
    if config.is_some() {
        command.arg("--config");
        command.arg("-");
    }
    command.stdin(Stdio::piped());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    let mut child = command.spawn()?;
    if let (Some(mut stdin), Some(body)) = (child.stdin.take(), config.as_deref()) {
        stdin.write_all(body.as_bytes())?;
    }
    child.wait_with_output()
}

fn header_config(headers: &[(String, String)]) -> Option<String> {
    if headers.is_empty() {
        return None;
    }
    let mut body = String::new();
    for (name, value) in headers {
        let escaped = value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace(['\n', '\r'], "");
        body.push_str("header = \"");
        body.push_str(name);
        body.push_str(": ");
        body.push_str(&escaped);
        body.push_str("\"\n");
    }
    Some(body)
}
