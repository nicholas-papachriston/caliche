//! Live `caliche --server` checks for the Windows registry.
//!
//! These spawn the package binary, so they see `caliche.exe`, a real
//! listener, and `TerminateProcess`. `CALICHE_CONFIG_DIR` keeps the
//! records, lock, and plugin store out of the user profile.

#![cfg(target_os = "windows")]

use caliche_server::paths::CONFIG_DIR_ENV;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_caliche")
}

struct Server {
    child: Child,
    config: PathBuf,
    port: u16,
    log: PathBuf,
}

impl Server {
    fn start(config: &Path) -> Self {
        let port = free_port();
        let log = config.join("server.log");
        let log_file = fs::File::create(&log).expect("server log");
        let child = Command::new(bin())
            .args([
                "--server",
                "--bind",
                "127.0.0.1",
                "--port",
                &port.to_string(),
            ])
            .env(CONFIG_DIR_ENV, config)
            .stdout(Stdio::null())
            .stderr(Stdio::from(log_file))
            .spawn()
            .expect("spawn server");
        Self {
            child,
            config: config.to_path_buf(),
            port,
            log,
        }
    }

    async fn wait_until_running(&mut self) {
        let ready = wait_for(Duration::from_secs(45), || {
            if self.child.try_wait().ok().flatten().is_some() {
                return true;
            }
            registry_command(&self.config, &["server", "list"])
                .map(|output| String::from_utf8_lossy(&output.stdout).contains("running"))
                .unwrap_or(false)
        })
        .await;
        if self.child.try_wait().ok().flatten().is_some() {
            panic!(
                "server exited before it was listed as running\n{}",
                read_log(&self.log)
            );
        }
        assert!(ready, "server was not running\n{}", read_log(&self.log));
        assert!(
            self.config.join("plugins.db").is_file(),
            "plugins.db was not created under CALICHE_CONFIG_DIR"
        );
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test]
async fn live_server_is_running_in_list_and_status_and_stop_removes_it() {
    let root = tempfile::tempdir().expect("temp");
    let config = root.path().join("config");
    fs::create_dir_all(&config).expect("config dir");
    let mut server = Server::start(&config);
    server.wait_until_running().await;

    let list = registry_command(&config, &["server", "list"]).expect("list");
    let list_text = String::from_utf8_lossy(&list.stdout);
    assert!(
        list.status.success(),
        "{list_text}\n{}",
        read_log(&server.log)
    );
    assert!(list_text.contains("running"), "{list_text}");

    let status = registry_command(
        &config,
        &["server", "status", "--port", &server.port.to_string()],
    )
    .expect("status");
    let status_text = String::from_utf8_lossy(&status.stdout);
    assert!(
        status.status.success(),
        "{status_text}\n{}",
        read_log(&server.log)
    );
    assert!(status_text.contains("state: running"), "{status_text}");

    let stop = registry_command(
        &config,
        &["server", "stop", "--port", &server.port.to_string()],
    )
    .expect("stop");
    let stop_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&stop.stdout),
        String::from_utf8_lossy(&stop.stderr)
    );
    assert!(
        stop.status.success(),
        "{stop_text}\n{}",
        read_log(&server.log)
    );
    assert!(
        json_records(&config).is_empty(),
        "stop left records: {:?}",
        json_records(&config)
    );
    let exited = wait_for(Duration::from_secs(10), || {
        server.child.try_wait().ok().flatten().is_some()
    })
    .await;
    assert!(exited, "stop left the server running");
}

#[tokio::test]
async fn stop_keeps_the_record_of_a_still_running_server() {
    let root = tempfile::tempdir().expect("temp");
    let config = root.path().join("config");
    fs::create_dir_all(&config).expect("config dir");
    let mut server = Server::start(&config);
    server.wait_until_running().await;
    let records = json_records(&config);
    assert_eq!(records.len(), 1, "{records:?}");
    rewrite_exe(&records[0], r"C:\Windows\System32\notepad.exe");

    let stop = registry_command(
        &config,
        &["server", "stop", "--port", &server.port.to_string()],
    )
    .expect("stop");
    assert!(
        !stop.status.success(),
        "stop should refuse a mismatched executable"
    );
    assert!(
        records[0].is_file(),
        "stop deleted the record of a running server"
    );
    assert!(
        server.child.try_wait().expect("poll").is_none(),
        "stop signaled a server it could not prove"
    );
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("addr").port()
}

fn registry_command(config: &Path, args: &[&str]) -> std::io::Result<std::process::Output> {
    Command::new(bin())
        .args(args)
        .env(CONFIG_DIR_ENV, config)
        .output()
}

fn json_records(config: &Path) -> Vec<PathBuf> {
    let dir = config.join("servers");
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect()
}

fn rewrite_exe(path: &Path, exe: &str) {
    let raw = fs::read_to_string(path).expect("read record");
    let mut value: serde_json::Value = serde_json::from_str(&raw).expect("parse record");
    value["exe"] = serde_json::Value::String(exe.to_string());
    fs::write(
        path,
        serde_json::to_string_pretty(&value).expect("serialize"),
    )
    .expect("rewrite");
}

fn read_log(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

async fn wait_for(limit: Duration, mut probe: impl FnMut() -> bool) -> bool {
    let started = Instant::now();
    while started.elapsed() < limit {
        if probe() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    probe()
}
