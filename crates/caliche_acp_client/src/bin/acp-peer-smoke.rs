//! Drive an ACP v1 peer and write a JSONL transcript plus a summary JSON file.
//!
//! ```text
//! cargo run -p caliche_acp_client --bin acp-peer-smoke -- --peer fixture
//! cargo run -p caliche_acp_client --bin acp-peer-smoke -- --peer opencode --transcript /tmp/acp-opencode.jsonl --summary /tmp/acp-opencode.json
//! cargo run -p caliche_acp_client --bin acp-peer-smoke -- --peer hermes --transcript /tmp/acp-hermes.jsonl --summary /tmp/acp-hermes.json
//! cargo run -p caliche_acp_client --bin acp-peer-smoke -- --matrix
//! ```
//!
//! Exit `0` is a finished turn, `1` is a protocol or spawn failure, `2` is an
//! auth refusal (`auth_rejected` or `auth_required`), and `3` means the
//! selected peer binary was not found.
//! `--peer auto` prefers `OpenCode`, then Hermes, and does not use the fixture.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use caliche_acp_client::smoke::{
    PeerKind, PeerQuery, SmokeConfig, SmokeReport, peer_matrix, run_smoke,
};

const HELP: &str = "\
acp-peer-smoke — ACP v1 peer client smoke

Usage:
  acp-peer-smoke [--peer auto|fixture|opencode|hermes]
                 [--cwd PATH] [--transcript PATH] [--summary PATH]
                 [--prompt TEXT] [--timeout SECS]
                 [--opencode PATH] [--hermes PATH]
                 [--fixture PATH] [--fixture-mode MODE]
                 [--print-logs] [--inherit-env | --no-inherit-env]
                 [--matrix]

--peer auto uses OpenCode when installed, otherwise Hermes. It does not
fall through to the fixture. The peer gets this process's PATH, HOME, locale,
and temp dirs, never a *_KEY, *_SECRET, or *_TOKEN. Pass --inherit-env to
hand it the full environment, provider keys included. The summary and
transcript redact credential-shaped values.

Exit codes: 0 completed, 1 failed, 2 auth rejected or auth_required, 3 peer missing.
Login hints (`opencode-login`, `hermes-setup`) do not abort initialize.
This process does not run `opencode auth login`, `hermes setup`, or `hermes acp --setup`.
";

struct Args {
    peer: PeerKind,
    cwd: PathBuf,
    transcript: PathBuf,
    summary: PathBuf,
    prompt: String,
    opencode_bin: Option<PathBuf>,
    hermes_bin: Option<PathBuf>,
    fixture_script: PathBuf,
    fixture_mode: String,
    print_logs: bool,
    matrix_only: bool,
    help: bool,
    timeout_secs: u64,
    inherit_env: Option<bool>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            write_status(&err);
            return ExitCode::from(1);
        }
    };
    if args.help {
        write_status(HELP);
        return ExitCode::SUCCESS;
    }
    if args.matrix_only {
        return write_matrix_exit(&args.summary);
    }
    let config = SmokeConfig {
        peer: PeerQuery {
            peer: args.peer,
            opencode_bin: args.opencode_bin,
            hermes_bin: args.hermes_bin,
            fixture_script: args.fixture_script,
            fixture_mode: args.fixture_mode,
            opencode_print_logs: args.print_logs,
        },
        cwd: args.cwd,
        prompt: args.prompt,
        transcript: args.transcript,
        timeout: Duration::from_secs(args.timeout_secs),
        inherit_env: args.inherit_env,
        env: Vec::new(),
    };
    let report = run_smoke(config).await;
    if let Err(err) = write_summary(&args.summary, &report) {
        write_status(&err.to_string());
        return ExitCode::from(1);
    }
    write_status(&format!(
        "status={} peer={} summary={} detail={}",
        report.status.as_str(),
        report.peer_id,
        args.summary.display(),
        one_line(&report.detail)
    ));
    ExitCode::from(report.status.code())
}

fn write_matrix_exit(summary: &std::path::Path) -> ExitCode {
    match write_json(summary, &matrix_value()) {
        Ok(()) => {
            write_status(&format!("wrote command matrix to {}", summary.display()));
            ExitCode::SUCCESS
        }
        Err(err) => {
            write_status(&err.to_string());
            ExitCode::from(1)
        }
    }
}

fn matrix_value() -> serde_json::Value {
    let rows: Vec<_> = peer_matrix()
        .iter()
        .map(|entry| {
            serde_json::json!({
                "id": entry.id,
                "argv": entry.argv,
                "auth": entry.auth,
            })
        })
        .collect();
    serde_json::Value::Array(rows)
}

fn write_summary(path: &std::path::Path, report: &SmokeReport) -> std::io::Result<()> {
    write_json(path, report)
}

fn write_json(path: &std::path::Path, value: &impl serde::Serialize) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, value).map_err(std::io::Error::other)
}

fn one_line(text: &str) -> String {
    text.lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(240)
        .collect()
}

#[expect(
    clippy::print_stderr,
    reason = "the smoke binary reports status to the operator"
)]
fn write_status(text: &str) {
    eprintln!("acp-peer-smoke: {text}");
}

fn parse_args() -> Result<Args, String> {
    parse_arg_list(std::env::args().skip(1))
}

fn parse_arg_list(values: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let cwd = std::env::current_dir().map_err(|err| err.to_string())?;
    let mut args = Args {
        peer: PeerKind::Auto,
        transcript: cwd.join("acp-peer-smoke.jsonl"),
        summary: cwd.join("acp-peer-smoke.json"),
        cwd,
        prompt: "Reply with the single word pong.".to_owned(),
        opencode_bin: None,
        hermes_bin: None,
        fixture_script: default_fixture(),
        fixture_mode: "happy".to_owned(),
        print_logs: false,
        matrix_only: false,
        help: false,
        timeout_secs: 30,
        inherit_env: None,
    };
    let mut raw = values.into_iter();
    while let Some(arg) = raw.next() {
        apply_arg(&mut args, &arg, &mut raw)?;
    }
    Ok(args)
}

fn apply_arg(
    args: &mut Args,
    arg: &str,
    raw: &mut impl Iterator<Item = String>,
) -> Result<(), String> {
    match arg {
        "--help" | "-h" => {
            args.help = true;
            Ok(())
        }
        "--matrix" => {
            args.matrix_only = true;
            Ok(())
        }
        "--print-logs" => {
            args.print_logs = true;
            Ok(())
        }
        "--inherit-env" => {
            args.inherit_env = Some(true);
            Ok(())
        }
        "--no-inherit-env" => {
            args.inherit_env = Some(false);
            Ok(())
        }
        "--cwd" => set_path(raw, arg, &mut args.cwd),
        "--transcript" => set_path(raw, arg, &mut args.transcript),
        "--summary" => set_path(raw, arg, &mut args.summary),
        "--fixture" => set_path(raw, arg, &mut args.fixture_script),
        "--opencode" => set_optional_path(raw, arg, &mut args.opencode_bin),
        "--hermes" => set_optional_path(raw, arg, &mut args.hermes_bin),
        "--peer" => set_peer(args, raw),
        "--prompt" => set_text(raw, arg, &mut args.prompt),
        "--fixture-mode" => set_text(raw, arg, &mut args.fixture_mode),
        "--timeout" => set_timeout(args, raw),
        other => Err(format!("unknown argument {other}")),
    }
}

fn set_path(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
    slot: &mut PathBuf,
) -> Result<(), String> {
    *slot = PathBuf::from(next_value(raw, flag)?);
    Ok(())
}

fn set_optional_path(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
    slot: &mut Option<PathBuf>,
) -> Result<(), String> {
    *slot = Some(PathBuf::from(next_value(raw, flag)?));
    Ok(())
}

fn set_text(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
    slot: &mut String,
) -> Result<(), String> {
    *slot = next_value(raw, flag)?;
    Ok(())
}

fn set_peer(args: &mut Args, raw: &mut impl Iterator<Item = String>) -> Result<(), String> {
    args.peer = parse_peer(&next_value(raw, "--peer")?)?;
    Ok(())
}

fn set_timeout(args: &mut Args, raw: &mut impl Iterator<Item = String>) -> Result<(), String> {
    args.timeout_secs = parse_timeout(&next_value(raw, "--timeout")?)?;
    Ok(())
}

fn parse_peer(text: &str) -> Result<PeerKind, String> {
    PeerKind::parse(text)
        .ok_or_else(|| format!("unknown peer {text}; expected auto, fixture, opencode, or hermes"))
}

fn parse_timeout(text: &str) -> Result<u64, String> {
    let secs: u64 = text
        .parse()
        .map_err(|_| format!("invalid timeout {text}"))?;
    if secs == 0 {
        return Err("timeout must be at least 1 second".to_owned());
    }
    Ok(secs)
}

fn next_value(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn default_fixture() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixture_agent.py"
    ))
}

#[cfg(test)]
mod tests {
    use super::parse_arg_list;
    use std::path::PathBuf;

    fn owned(values: &[&str]) -> Vec<String> {
        values.iter().copied().map(str::to_owned).collect()
    }

    #[test]
    fn valued_flags_consume_the_following_token() {
        let args = match parse_arg_list(owned(&[
            "--cwd",
            "/tmp/work",
            "--prompt",
            "pong",
            "--timeout",
            "4",
            "--matrix",
            "--opencode",
            "/bin/opencode",
        ])) {
            Ok(args) => args,
            Err(error) => panic!("parse failed: {error}"),
        };
        assert_eq!(args.cwd, PathBuf::from("/tmp/work"));
        assert_eq!(args.prompt, "pong");
        assert_eq!(args.timeout_secs, 4);
        assert!(args.matrix_only);
        assert_eq!(args.opencode_bin, Some(PathBuf::from("/bin/opencode")));
    }

    #[test]
    fn unknown_flag_reports_that_flag() {
        let error = match parse_arg_list(owned(&["--nope", "--matrix"])) {
            Ok(_) => panic!("unknown flag was accepted"),
            Err(error) => error,
        };
        assert_eq!(error, "unknown argument --nope");
    }

    #[test]
    fn missing_flag_value_names_the_flag() {
        let error = match parse_arg_list(owned(&["--cwd"])) {
            Ok(_) => panic!("missing value was accepted"),
            Err(error) => error,
        };
        assert_eq!(error, "--cwd needs a value");
    }
}
