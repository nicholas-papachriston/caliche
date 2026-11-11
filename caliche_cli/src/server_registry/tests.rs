use super::inspect::{self, Inspection};
use super::{
    ListedServer, SIGKILL, SIGTERM, ServerOrigin, ServerRecord, ServerSelector, ServerState,
    format_list, format_status, list_with, prune_dead_records, register_current,
    remove_records_for_pid_if_dead, selector_from_cli, stop_with, write_record,
};
use crate::client::{ServerLock, write_lock};
use crate::commands::{Args, Commands, ServerAction};
use crate::process_liveness::REASON_TOKEN_MISMATCH;
use clap::Parser;
use std::net::TcpListener;
use std::path::Path;
use std::process;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

fn sample(id: &str, pid: u32, port: u16) -> ServerRecord {
    ServerRecord {
        version: 1,
        id: id.to_string(),
        pid,
        port,
        bind: "127.0.0.1".to_string(),
        build_id: "abc".to_string(),
        started_at: 1_700_000_000,
        cwd: "/work".to_string(),
        exe: "/usr/local/bin/caliche".to_string(),
        origin: ServerOrigin::Foreground,
        start_token: "token".to_string(),
    }
}

fn owned(record: &ServerRecord) -> Inspection {
    Inspection::Owned {
        token: record.start_token.clone(),
    }
}

async fn stop_port(
    dir: &Path,
    lock: &Path,
    port: u16,
    inspector: impl Fn(&ServerRecord) -> Inspection,
    signal: impl FnMut(u32, i32, &str) -> anyhow::Result<()>,
) -> super::StopReport {
    stop_with(
        dir,
        lock,
        ServerSelector::Port(port),
        inspector,
        signal,
        || std::future::ready(()),
    )
    .await
    .expect("stop")
}

#[test]
fn origin_env_accepts_only_daemon() {
    assert_eq!(
        ServerOrigin::from_origin_env(Some("daemon")),
        ServerOrigin::Daemon
    );
    assert_eq!(
        ServerOrigin::from_origin_env(Some(" daemon ")),
        ServerOrigin::Daemon
    );
    assert_eq!(
        ServerOrigin::from_origin_env(Some("foreground")),
        ServerOrigin::Foreground
    );
    assert_eq!(
        ServerOrigin::from_origin_env(None),
        ServerOrigin::Foreground
    );
    assert_eq!(ServerOrigin::Embedded.as_str(), "embedded");
}

#[test]
fn register_roundtrip_keeps_identity_fields() {
    let dir = tempdir().unwrap();
    let record = register_current(dir.path(), ServerOrigin::Foreground, "127.0.0.1", 3060)
        .expect("register");
    assert_eq!(record.pid, process::id());
    assert_eq!(record.port, 3060);
    assert_eq!(record.bind, "127.0.0.1");
    assert_eq!(record.origin, ServerOrigin::Foreground);
    assert!(!record.start_token.is_empty());
    assert!(record.id.starts_with("foreground-3060-"));
    let listed = list_with(dir.path(), |_| Inspection::Stale).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].record.id, record.id);
    assert_eq!(listed[0].state, ServerState::Stale);
}

#[test]
fn list_marks_scripted_running_record() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("foreground-3000-7-1", 7, 3000)).unwrap();
    let listed = list_with(dir.path(), owned).unwrap();
    assert_eq!(listed[0].state, ServerState::Running);
    assert!(listed[0].detail.contains("listening"));
    let text = format_list(dir.path(), &listed, None);
    assert!(text.contains("foreground-3000-7-1"));
    assert!(text.contains("running"));
    assert!(text.contains("server.lock: absent"));
    let status = format_status(dir.path(), &listed, None);
    assert!(status.contains("origin: foreground"));
    assert!(status.contains("port: 3000"));
}

#[tokio::test]
async fn stop_removes_stale_record_and_matching_lock_without_signaling() {
    let dir = tempdir().unwrap();
    let record = sample("foreground-3000-9-1", 9, 3000);
    write_record(dir.path(), &record).unwrap();
    let lock_path = dir.path().join("server.lock");
    let mut lock = ServerLock::new("127.0.0.1", 3000);
    lock.pid = 9;
    write_lock(&lock_path, &lock).unwrap();
    let other = sample("foreground-3060-8-1", 8, 3060);
    write_record(dir.path(), &other).unwrap();

    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &lock_path,
        3000,
        |_| Inspection::Stale,
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;

    assert!(!report.failed);
    assert!(signals.lock().expect("signals").is_empty());
    assert!(report.lines[0].contains("removed stale"));
    assert!(!dir.path().join("foreground-3000-9-1.json").exists());
    assert!(dir.path().join("foreground-3060-8-1.json").exists());
    assert!(!lock_path.exists());
}

#[tokio::test]
async fn stop_refuses_to_signal_a_foreign_exe_and_keeps_the_record() {
    let dir = tempdir().unwrap();
    let record = sample("foreground-3000-9-1", 9, 3000);
    write_record(dir.path(), &record).unwrap();
    let lock_path = dir.path().join("server.lock");
    let mut lock = ServerLock::new("127.0.0.1", 3000);
    lock.pid = 9;
    write_lock(&lock_path, &lock).unwrap();
    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &lock_path,
        3000,
        |_| Inspection::Foreign {
            reason: "process executable is not caliche".to_string(),
        },
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;
    assert!(report.failed);
    assert!(signals.lock().expect("signals").is_empty());
    assert!(report.lines[0].contains("not caliche"));
    assert!(!report.lines[0].contains("removed the record"));
    assert!(dir.path().join("foreground-3000-9-1.json").exists());
    assert!(lock_path.exists());
}

#[tokio::test]
async fn stop_removes_a_recycled_inspection_without_matching_the_reason_text() {
    let dir = tempdir().unwrap();
    let record = sample("foreground-3000-9-1", 9, 3000);
    write_record(dir.path(), &record).unwrap();
    let lock_path = dir.path().join("server.lock");
    let mut lock = ServerLock::new("127.0.0.1", 3000);
    lock.pid = 9;
    write_lock(&lock_path, &lock).unwrap();
    let report = stop_port(
        dir.path(),
        &lock_path,
        3000,
        |_| Inspection::Recycled {
            reason: "rewritten reused-pid text".to_string(),
        },
        |_pid, _sig, _token| panic!("a reused pid must not be signaled"),
    )
    .await;
    assert!(report.failed);
    assert!(
        report.lines[0].contains("rewritten reused-pid text"),
        "{:?}",
        report.lines
    );
    assert!(report.lines[0].contains("removed the record"));
    assert!(!dir.path().join("foreground-3000-9-1.json").exists());
    assert!(!lock_path.exists());
}

#[tokio::test]
async fn stop_keeps_a_foreign_record_that_repeats_the_reused_pid_text() {
    let dir = tempdir().unwrap();
    let record = sample("foreground-3000-9-1", 9, 3000);
    write_record(dir.path(), &record).unwrap();
    let lock_path = dir.path().join("server.lock");
    let mut lock = ServerLock::new("127.0.0.1", 3000);
    lock.pid = 9;
    write_lock(&lock_path, &lock).unwrap();
    let report = stop_port(
        dir.path(),
        &lock_path,
        3000,
        |_| Inspection::Foreign {
            reason: REASON_TOKEN_MISMATCH.to_string(),
        },
        |_pid, _sig, _token| panic!("foreign text must not be signaled"),
    )
    .await;
    assert!(report.failed);
    assert!(report.lines[0].contains(REASON_TOKEN_MISMATCH));
    assert!(!report.lines[0].contains("removed the record"));
    assert!(dir.path().join("foreground-3000-9-1.json").exists());
    assert!(lock_path.exists());
}

#[tokio::test]
async fn stop_keeps_an_unproven_record_and_does_not_signal() {
    let dir = tempdir().unwrap();
    let record = sample("foreground-3000-9-1", 9, 3000);
    write_record(dir.path(), &record).unwrap();
    let lock_path = dir.path().join("server.lock");
    let mut lock = ServerLock::new("127.0.0.1", 4444);
    lock.pid = 44;
    write_lock(&lock_path, &lock).unwrap();
    let report = stop_port(
        dir.path(),
        &lock_path,
        3000,
        |_| Inspection::Unproven {
            reason: "pid is not listening on the recorded port".to_string(),
        },
        |_pid, _sig, _token| panic!("unproven record must not be signaled"),
    )
    .await;
    assert!(report.failed);
    assert!(report.lines[0].contains("not listening"));
    assert!(dir.path().join("foreground-3000-9-1.json").exists());
    assert!(lock_path.exists());
}

#[tokio::test]
async fn stop_signals_an_owned_server_and_removes_its_lock() {
    let dir = tempdir().unwrap();
    let record = sample("foreground-3060-15-1", 15, 3060);
    write_record(dir.path(), &record).unwrap();
    let lock_path = dir.path().join("server.lock");
    let mut lock = ServerLock::new("127.0.0.1", 3060);
    lock.pid = 15;
    write_lock(&lock_path, &lock).unwrap();
    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &lock_path,
        3060,
        {
            let signals = Arc::clone(&signals);
            move |record| {
                if signals.lock().expect("signals").is_empty() {
                    owned(record)
                } else {
                    Inspection::Stale
                }
            }
        },
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;
    assert!(!report.failed, "{:?}", report.lines);
    assert_eq!(
        signals.lock().expect("signals").as_slice(),
        &[(15, SIGTERM)]
    );
    assert!(report.lines[0].contains("stopped"));
    assert!(!dir.path().join("foreground-3060-15-1.json").exists());
    assert!(!lock_path.exists());
}

#[tokio::test]
async fn stop_sends_sigkill_when_the_process_stays_owned() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("foreground-3060-15-1", 15, 3060)).unwrap();
    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &dir.path().join("missing.lock"),
        3060,
        {
            let signals = Arc::clone(&signals);
            move |record| {
                if signals.lock().expect("signals").len() >= 2 {
                    Inspection::Stale
                } else {
                    owned(record)
                }
            }
        },
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;
    assert!(!report.failed, "{:?}", report.lines);
    assert_eq!(
        signals.lock().expect("signals").as_slice(),
        &[(15, SIGTERM), (15, SIGKILL)]
    );
}

#[tokio::test]
async fn stop_does_not_signal_when_ownership_changes_before_the_signal() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("foreground-3000-9-1", 9, 3000)).unwrap();
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let report = stop_port(
        dir.path(),
        &dir.path().join("missing.lock"),
        3000,
        move |record| {
            let seen = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if seen == 0 {
                owned(record)
            } else {
                Inspection::Unproven {
                    reason: "changed".to_string(),
                }
            }
        },
        |_pid, _sig, _token| panic!("changed identity must not be signaled"),
    )
    .await;
    assert!(report.failed);
    assert!(dir.path().join("foreground-3000-9-1.json").exists());
}

#[tokio::test]
async fn stop_refuses_two_owned_servers_on_the_same_port() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("foreground-3000-1-1", 1, 3000)).unwrap();
    write_record(dir.path(), &sample("foreground-3000-2-2", 2, 3000)).unwrap();
    let err = stop_with(
        dir.path(),
        &dir.path().join("missing.lock"),
        ServerSelector::Port(3000),
        owned,
        |_pid, _sig, _token| panic!("ambiguous servers must not be signaled"),
        || std::future::ready(()),
    )
    .await
    .expect_err("ambiguous");
    assert!(err.to_string().contains("multiple running"));
    assert!(dir.path().join("foreground-3000-1-1.json").exists());
    assert!(dir.path().join("foreground-3000-2-2.json").exists());
}

#[tokio::test]
async fn stop_by_id_does_not_require_a_port_match_with_another_record() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("keep-me", 4, 3060)).unwrap();
    write_record(dir.path(), &sample("drop-me", 5, 3000)).unwrap();
    let report = stop_with(
        dir.path(),
        &dir.path().join("missing.lock"),
        ServerSelector::Id("drop-me".to_string()),
        |_| Inspection::Stale,
        |_pid, _sig, _token| panic!("stale"),
        || std::future::ready(()),
    )
    .await
    .unwrap();
    assert!(!report.failed);
    assert!(dir.path().join("keep-me.json").exists());
    assert!(!dir.path().join("drop-me.json").exists());
}

#[tokio::test]
async fn selector_rejects_an_id_on_a_different_port() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("foreground-3000-1-1", 1, 3000)).unwrap();
    let error = stop_with(
        dir.path(),
        &dir.path().join("missing.lock"),
        ServerSelector::Both {
            id: "foreground-3000-1-1".to_string(),
            port: 3060,
        },
        |_| Inspection::Stale,
        |_pid, _sig, _token| panic!("mismatch"),
        || std::future::ready(()),
    )
    .await
    .expect_err("port mismatch");
    assert!(error.to_string().contains("not 3060"));
    assert!(dir.path().join("foreground-3000-1-1.json").exists());
}

#[test]
fn missing_selector_fails() {
    let err = selector_from_cli(None, None).expect_err("missing");
    assert!(err.to_string().contains("--port"));
}

#[test]
fn a_newer_record_version_is_skipped_and_left_in_place() {
    let dir = tempdir().unwrap();
    let newer = dir.path().join("newer.json");
    let raw = r#"{"version":2,"id":"newer","pid":1,"port":1,"bind":"127.0.0.1","build_id":"x","started_at":1,"cwd":"/","exe":"/usr/local/bin/caliche","origin":"foreground","start_token":"t"}"#;
    fs_write(&newer, raw);
    assert!(super::list_servers(dir.path()).expect("list").is_empty());
    assert_eq!(prune_dead_records(dir.path()).expect("prune"), 0);
    assert_eq!(std::fs::read_to_string(&newer).expect("still there"), raw);
    assert!(!dir.path().join("newer.json.bad").exists());
}

#[test]
fn a_newer_record_with_a_new_shape_is_skipped_not_quarantined() {
    let dir = tempdir().unwrap();
    let newer = dir.path().join("reshaped.json");
    let raw = r#"{"version":2,"id":"reshaped","pid":1,"port":1,"origin":"service","listen":{"addr":"127.0.0.1"}}"#;
    fs_write(&newer, raw);
    assert!(super::list_servers(dir.path()).expect("list").is_empty());
    assert_eq!(prune_dead_records(dir.path()).expect("prune"), 0);
    assert_eq!(std::fs::read_to_string(&newer).expect("still there"), raw);
    assert!(!dir.path().join("reshaped.json.bad").exists());
}

#[cfg(all(unix, not(target_os = "linux")))]
#[test]
fn a_ps_token_mismatch_keeps_the_record() {
    let mut record =
        super::capture_record(super::ServerOrigin::Foreground, "127.0.0.1", 1).expect("capture");
    record.start_token = "Mon Jan  1 00:00:00 2001".to_string();
    assert_eq!(
        inspect::inspect_live(&record),
        Inspection::Foreign {
            reason: REASON_TOKEN_MISMATCH.to_string()
        }
    );
}

#[cfg(all(unix, not(target_os = "linux")))]
#[test]
fn the_ps_start_token_ignores_the_caller_locale_and_zone() {
    let command = inspect::start_token_command(7).expect("ps");
    let envs: Vec<(String, Option<String>)> = command
        .get_envs()
        .map(|(key, value)| {
            (
                key.to_string_lossy().into_owned(),
                value.map(|value| value.to_string_lossy().into_owned()),
            )
        })
        .collect();
    assert_eq!(
        envs,
        [
            ("LC_ALL".to_string(), Some("C".to_string())),
            ("TZ".to_string(), Some("UTC".to_string())),
        ]
    );
}

#[test]
fn an_unreadable_record_is_skipped_and_left_in_place() {
    let dir = tempdir().unwrap();
    // Reading a directory fails with an I/O error other than NotFound, as a
    // permission or sharing violation would.
    let unreadable = dir.path().join("held.json");
    std::fs::create_dir(&unreadable).unwrap();
    assert!(super::list_servers(dir.path()).expect("list").is_empty());
    assert_eq!(prune_dead_records(dir.path()).expect("prune"), 0);
    assert!(unreadable.is_dir());
    assert!(!dir.path().join("held.json.bad").exists());
}

#[test]
fn prune_removes_dead_pids_and_keeps_this_process() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("dead-record", u32::MAX - 1, 3000)).unwrap();
    let mut live = sample("live-record", process::id(), 3060);
    live.start_token = inspect::current_start_token().expect("token");
    write_record(dir.path(), &live).unwrap();
    let removed = prune_dead_records(dir.path()).unwrap();
    assert_eq!(removed, 1);
    assert!(!dir.path().join("dead-record.json").exists());
    assert!(dir.path().join("live-record.json").exists());
    assert_eq!(
        remove_records_for_pid_if_dead(dir.path(), process::id()).unwrap(),
        0
    );
    assert!(dir.path().join("live-record.json").exists());
}

#[test]
fn prune_with_keeps_unproven_and_foreign_records() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("stale-record", 9, 3000)).unwrap();
    write_record(dir.path(), &sample("unproven-record", 9, 3001)).unwrap();
    write_record(dir.path(), &sample("foreign-record", 9, 3002)).unwrap();
    write_record(dir.path(), &sample("recycled-record", 9, 3003)).unwrap();
    let removed = super::prune_with(dir.path(), |record| match record.id.as_str() {
        "stale-record" => Inspection::Stale,
        "foreign-record" => Inspection::Foreign {
            reason: "process executable is not caliche".to_string(),
        },
        "recycled-record" => Inspection::Recycled {
            reason: REASON_TOKEN_MISMATCH.to_string(),
        },
        _ => Inspection::Unproven {
            reason: "could not prove the process".to_string(),
        },
    })
    .unwrap();
    assert_eq!(removed, 1);
    assert!(!dir.path().join("stale-record.json").exists());
    assert!(dir.path().join("unproven-record.json").exists());
    assert!(dir.path().join("foreign-record.json").exists());
    assert!(dir.path().join("recycled-record.json").exists());
}

#[tokio::test]
async fn corrupt_record_does_not_block_list_status_stop_or_prune() {
    let dir = tempdir().unwrap();
    write_record(dir.path(), &sample("good-record", 9, 3000)).unwrap();
    fs_write(dir.path().join("bad-record.json"), "{not json");
    let listed = super::list_servers(dir.path()).expect("list");
    assert_eq!(listed.len(), 1);
    assert!(format_status(dir.path(), &listed, None).contains("good-record"));
    assert!(!dir.path().join("bad-record.json").exists());
    assert!(dir.path().join("bad-record.json.bad").exists());

    let lock_path = dir.path().join("server.lock");
    let report = stop_port(
        dir.path(),
        &lock_path,
        3000,
        |_| Inspection::Stale,
        |_pid, _sig, _token| panic!("a stale record is not signaled"),
    )
    .await;
    assert!(!report.failed);
    assert!(!dir.path().join("good-record.json").exists());

    write_record(dir.path(), &sample("kept-record", 9, 3060)).unwrap();
    fs_write(dir.path().join("also-bad.json"), "nope");
    let removed = super::prune_with(dir.path(), |_| Inspection::Unproven {
        reason: "could not prove the process".to_string(),
    })
    .unwrap();
    assert_eq!(removed, 0);
    assert!(dir.path().join("kept-record.json").exists());
    assert!(!dir.path().join("also-bad.json").exists());
    assert!(dir.path().join("also-bad.json.bad").exists());
}

#[test]
fn missing_record_file_is_not_an_error() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("gone.json");
    assert!(matches!(
        super::read_stored(&missing),
        super::RecordRead::Missing
    ));
    assert!(super::list_servers(dir.path()).expect("list").is_empty());
}

#[tokio::test]
async fn stop_removes_a_dead_pid_through_live_inspection() {
    let dir = tempdir().unwrap();
    let dead = u32::MAX - 1;
    write_record(dir.path(), &sample("dead-live-inspect", dead, 3000)).unwrap();
    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &dir.path().join("missing.lock"),
        3000,
        inspect::inspect_live,
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;
    assert!(!report.failed, "{:?}", report.lines);
    assert!(signals.lock().expect("signals").is_empty());
    assert!(
        report
            .lines
            .iter()
            .any(|line| line.contains("removed stale")),
        "{:?}",
        report.lines
    );
    assert!(!dir.path().join("dead-live-inspect.json").exists());
}

#[tokio::test]
async fn stop_does_not_treat_a_recycled_pid_as_the_original() {
    let dir = tempdir().unwrap();
    let mut record = sample("recycled-self", process::id(), 9);
    record.start_token = "not-this-process".to_string();
    write_record(dir.path(), &record).unwrap();
    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &dir.path().join("missing.lock"),
        9,
        inspect::inspect_live,
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;
    assert!(signals.lock().expect("signals").is_empty());
    assert!(report.failed);
    assert!(
        report
            .lines
            .iter()
            .any(|line| line.contains(REASON_TOKEN_MISMATCH)),
        "{:?}",
        report.lines
    );
    // Only a Linux or Windows token proves reuse; a ps start time keeps the record.
    assert_eq!(
        dir.path().join("recycled-self.json").exists(),
        cfg!(all(unix, not(target_os = "linux")))
    );
}

#[tokio::test]
async fn live_inspector_does_not_signal_this_process() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = tempdir().unwrap();
    let mut record = sample("foreground-self", process::id(), port);
    let token = inspect::current_start_token().expect("token");
    record.start_token = token;
    record.exe = "/usr/local/bin/caliche".to_string();
    write_record(dir.path(), &record).unwrap();
    let signals = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&signals);
    let report = stop_port(
        dir.path(),
        &dir.path().join("missing.lock"),
        port,
        inspect::inspect_live,
        move |pid, sig, _token| {
            recorded.lock().expect("signals").push((pid, sig));
            Ok(())
        },
    )
    .await;
    assert!(signals.lock().expect("signals").is_empty());
    assert!(report.failed);
    assert!(
        report.lines.iter().any(|line| line.contains("caliche")),
        "{:?}",
        report.lines
    );
    assert!(
        dir.path().join("foreground-self.json").exists(),
        "a live process must keep its record when the executable does not match"
    );
    assert!(
        TcpListener::bind(("127.0.0.1", port)).is_err(),
        "listener should still hold the port"
    );
}

#[test]
fn server_stop_parses_without_a_runtime_provider() {
    let args = Args::try_parse_from(["caliche", "server", "stop", "--port", "3060"]).unwrap();
    assert!(!crate::commands::invocation_uses_provider(
        args.command.as_ref()
    ));
    match args.command {
        Some(Commands::Server {
            action: ServerAction::Stop { id, port },
        }) => {
            assert!(id.is_none());
            assert_eq!(port, Some(3060));
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[cfg(target_os = "linux")]
#[test]
fn parses_listen_inodes_and_stat_start_token() {
    let table = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:0BB8 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 12345 1 0000000000000000 100 0 0 10 0
   1: 0100007F:0BB8 00000000:0000 01 00000000:00000000 00:00000000 00000000     0        0 99999 1 0000000000000000 100 0 0 10 0
   2: 00000000000000000000000000000000:1F90 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 54321 1 0000000000000000 100 0 0 10 0
";
    assert_eq!(
        inspect::parse_listen_inodes(table, 3000),
        vec!["12345".to_string()]
    );
    assert_eq!(
        inspect::parse_listen_inodes(table, 8080),
        vec!["54321".to_string()]
    );
    let stat = "123 (caliche) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 999 0 0";
    assert_eq!(inspect::parse_start_token(stat).as_deref(), Some("999"));

    let fd_dir = tempdir().unwrap();
    std::os::unix::fs::symlink("socket:[12345]", fd_dir.path().join("3")).unwrap();
    std::os::unix::fs::symlink("/dev/null", fd_dir.path().join("4")).unwrap();
    let mut inodes = std::collections::HashSet::new();
    inodes.insert("12345".to_string());
    assert!(inspect::fd_dir_has_inode(fd_dir.path(), &inodes).unwrap());
    inodes.clear();
    inodes.insert("1".to_string());
    assert!(!inspect::fd_dir_has_inode(fd_dir.path(), &inodes).unwrap());
}

fn fs_write(path: impl AsRef<Path>, contents: &str) {
    std::fs::write(path, contents).unwrap();
}

#[test]
fn empty_record_token_on_this_process_is_unproven() {
    let mut record = sample("empty-token", process::id(), 9);
    record.start_token.clear();
    assert_eq!(
        inspect::inspect_live(&record),
        Inspection::Unproven {
            reason: crate::process_liveness::REASON_EMPTY_TOKEN.to_string(),
        }
    );
}

#[test]
fn windows_exe_name_is_caliche() {
    assert!(inspect::exe_name_is_server("caliche"));
    assert!(inspect::exe_name_is_server("caliche.exe"));
    assert!(inspect::exe_name_is_server("CALICHE.EXE"));
    assert!(inspect::exe_name_is_server(r"C:\Program Files\caliche.exe"));
    assert!(inspect::exe_name_is_server("/usr/local/bin/caliche"));
    assert!(!inspect::exe_name_is_server("notepad.exe"));
    assert!(!inspect::exe_name_is_server("caliche.cmd"));
    assert!(!inspect::exe_name_is_server("caliche_helper"));
}

#[test]
fn bom_record_is_listed_and_a_bad_record_is_skipped() {
    let dir = tempdir().unwrap();
    let good = sample("good-record", 9, 3000);
    let json = serde_json::to_string(&good).unwrap();
    fs_write(
        dir.path().join("good-record.json"),
        &format!("\u{feff}{json}"),
    );
    fs_write(dir.path().join("bad-record.json"), "{not json");
    let listed = super::list_servers(dir.path()).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].record.id, "good-record");
    let status = format_status(dir.path(), &listed, None);
    assert!(status.contains("good-record"));
}

#[test]
#[serial_test::serial]
fn config_dir_override_redirects_lock_registry_plugins_and_config() {
    let dir = tempdir().unwrap();
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.remove(caliche_server::paths::CONFIG_DIR_ENV);
    let unset_lock = crate::client::ServerLock::default_path();
    env.set(caliche_server::paths::CONFIG_DIR_ENV, dir.path());
    assert_eq!(
        crate::client::ServerLock::default_path(),
        dir.path().join("server.lock")
    );
    assert_eq!(super::default_registry_dir(), dir.path().join("servers"));
    assert_eq!(
        caliche_server::paths::plugin_store_path(),
        dir.path().join("plugins.db")
    );
    assert_eq!(
        crate::config::FileConfigSource::user_config_path(),
        dir.path().join("config.toml")
    );
    assert_eq!(
        caliche_server::paths::mcp_config_path(),
        dir.path().join("mcp.json")
    );
    // An empty CALICHE_CONFIG_DIR is ignored.
    env.set(caliche_server::paths::CONFIG_DIR_ENV, "");
    assert_eq!(crate::client::ServerLock::default_path(), unset_lock);
}

#[test]
fn format_lock_names_the_matching_record() {
    let record = sample("foreground-3000-9-1", 9, 3000);
    let row = ListedServer {
        record,
        state: ServerState::Running,
        detail: "listening on the recorded port".to_string(),
    };
    let lock = ServerLock {
        host: "127.0.0.1".to_string(),
        port: 3000,
        url: "http://127.0.0.1:3000".to_string(),
        pid: 9,
        started_at: 1,
    };
    let text = format_list(
        Path::new("/records"),
        std::slice::from_ref(&row),
        Some(&lock),
    );
    assert!(text.contains("record foreground-3000-9-1"));
}
