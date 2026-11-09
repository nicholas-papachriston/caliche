//! A `.env` in the working directory is read only through `--env-file`.
//!
//! These spawn the package binary from a project dir that holds a planted
//! `.env`, with a cleared environment and a temp HOME, so the user's own
//! config and keychain scope stay out of the result.

use caliche_contracts::child_env::SYSTEMROOT;
use caliche_server::paths::CONFIG_DIR_ENV;
use std::fs;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Output};

const PLANTED_TEMPERATURE: &str = "0.42";

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_caliche")
}

fn run(project: &Path, home: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(bin());
    command
        .args(args)
        .current_dir(project)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("USERPROFILE", home)
        .env("APPDATA", home.join("appdata"))
        .env("LOCALAPPDATA", home.join("localappdata"))
        .env("XDG_CONFIG_HOME", home.join("xdg-config"))
        .env("XDG_DATA_HOME", home.join("xdg-data"));
    if let Some(root) = std::env::var_os(SYSTEMROOT) {
        command.env(SYSTEMROOT, root);
    }
    command.output().expect("spawn caliche")
}

struct Planted {
    _root: tempfile::TempDir,
    project: std::path::PathBuf,
    home: std::path::PathBuf,
}

/// A project whose `.env` sets the provider, model and config dir. The
/// planted config dir holds a `config.toml` that sets the temperature.
fn planted_project() -> Planted {
    let root = tempfile::tempdir().expect("temp dir");
    let project = root.path().join("project");
    let home = root.path().join("home");
    let planted_config = root.path().join("planted-config");
    for dir in [&project, &home, &planted_config] {
        fs::create_dir_all(dir).expect("create dir");
    }
    fs::write(
        planted_config.join("config.toml"),
        format!("temperature = {PLANTED_TEMPERATURE}\n"),
    )
    .expect("write planted config");
    fs::write(
        project.join(".env"),
        format!(
            "AGENT_PROVIDER=planted_provider\nAGENT_MODEL=planted-model\n{CONFIG_DIR_ENV}={}\n",
            planted_config.display()
        ),
    )
    .expect("write .env");
    Planted {
        _root: root,
        project,
        home,
    }
}

fn shown_config(output: &Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "config show failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("config show prints JSON")
}

#[test]
fn a_project_env_file_is_ignored_without_the_flag() {
    let planted = planted_project();
    let shown = shown_config(&run(
        &planted.project,
        &planted.home,
        &["config", "show", "--format", "json"],
    ));
    assert_eq!(shown["provider"], serde_json::Value::Null);
    assert_eq!(shown["model"], serde_json::Value::Null);
    assert_eq!(shown["temperature"], serde_json::Value::Null);
}

#[test]
fn a_project_env_file_is_loaded_with_the_flag() {
    let planted = planted_project();
    let shown = shown_config(&run(
        &planted.project,
        &planted.home,
        &["--env-file", ".env", "config", "show", "--format", "json"],
    ));
    assert_eq!(shown["provider"], "planted_provider");
    assert_eq!(shown["model"], "planted-model");
    assert_eq!(shown["temperature"], serde_json::json!(0.42));
}

/// Runs `--list-models ollama` with `OLLAMA_BASE_URL` in the project `.env`
/// pointing at a local listener, and reports whether a request reached it.
fn base_url_reached(with_flag: bool) -> bool {
    let planted = planted_project();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let addr = listener.local_addr().expect("listener addr");
    fs::write(
        planted.project.join(".env"),
        format!("OLLAMA_BASE_URL=http://{addr}\n"),
    )
    .expect("write .env");
    let mut args = vec!["--list-models", "ollama"];
    if with_flag {
        args.splice(0..0, ["--env-file", ".env"]);
    }
    // The listener takes the first connection and drops it, so a request
    // from the child fails at once. After the child exits, a probe connects;
    // the first connection came from the child exactly when it is not the
    // probe. The thread hands the listener back so it stays open for the
    // probe.
    let serve = std::thread::spawn(move || {
        let (stream, peer) = listener.accept().expect("accept");
        drop(stream);
        (peer, listener)
    });
    run(&planted.project, &planted.home, &args);
    let probe = TcpStream::connect(addr).expect("probe connection");
    let (first, _listener) = serve.join().expect("listener thread");
    first != probe.local_addr().expect("probe addr")
}

#[test]
fn a_project_base_url_is_ignored_without_the_flag() {
    assert!(!base_url_reached(false));
}

#[test]
fn a_project_base_url_is_used_with_the_flag() {
    assert!(base_url_reached(true));
}
