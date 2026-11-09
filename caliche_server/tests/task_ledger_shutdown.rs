//! The server binary keeps a task it records while it drains after SIGTERM,
//! even when the ledger database refuses the write.
//!
//! The binary runs against an in-test model server that asks for one bash
//! call, so the bash tool records a real task.

#![cfg(unix)]

use axum::Json;
use axum::extract::State;
use caliche::persistence::task_ledger::{TASKS_DB_FILE, TaskLedger};
use caliche_server::shutdown::run_with_task_ledger;
use serde_json::{Value, json};
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, mpsc};

const BASH_COMMAND: &str = "echo ledger-probe";
const WAIT: Duration = Duration::from_secs(60);

/// The model's replies: a bash call, then an answer once the tool result is
/// in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reply {
    BashCall,
    Answer,
}

#[derive(Clone)]
struct Script {
    held: Option<Reply>,
    release: Arc<Notify>,
    arrivals: mpsc::UnboundedSender<Reply>,
}

async fn chat_completions(State(script): State<Script>, Json(request): Json<Value>) -> Json<Value> {
    let has_tool_result = request["messages"]
        .as_array()
        .is_some_and(|messages| messages.iter().any(|message| message["role"] == "tool"));
    let reply = if has_tool_result {
        Reply::Answer
    } else {
        Reply::BashCall
    };
    let _ = script.arrivals.send(reply);
    if script.held == Some(reply) {
        script.release.notified().await;
    }
    let (message, finish_reason) = match reply {
        Reply::BashCall => (
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json!({ "command": BASH_COMMAND }).to_string(),
                    },
                }],
            }),
            "tool_calls",
        ),
        Reply::Answer => (json!({ "role": "assistant", "content": "done" }), "stop"),
    };
    Json(json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "fake-model",
        "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
    }))
}

/// OpenAI-compatible model server. The `held` reply waits for `release`.
struct FakeModel {
    base_url: String,
    release: Arc<Notify>,
    arrivals: mpsc::UnboundedReceiver<Reply>,
}

impl FakeModel {
    async fn start(held: Option<Reply>) -> anyhow::Result<Self> {
        let (arrivals_tx, arrivals) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let script = Script {
            held,
            release: Arc::clone(&release),
            arrivals: arrivals_tx,
        };
        let app = axum::Router::new()
            .route(
                "/v1/chat/completions",
                axum::routing::post(chat_completions),
            )
            .with_state(script);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        tokio::spawn(async move { axum::serve(listener, app).await });
        Ok(Self {
            base_url,
            release,
            arrivals,
        })
    }

    async fn next_request(&mut self) -> anyhow::Result<Reply> {
        tokio::time::timeout(WAIT, self.arrivals.recv())
            .await?
            .ok_or_else(|| anyhow::anyhow!("model server stopped"))
    }
}

/// Where `dirs::data_dir()` puts the ledger for `home`.
#[cfg(target_os = "macos")]
fn ledger_dir(home: &Path) -> PathBuf {
    home.join("Library/Application Support/caliche")
}

#[cfg(not(target_os = "macos"))]
fn ledger_dir(home: &Path) -> PathBuf {
    home.join("xdg/caliche")
}

async fn ledger_pool(dir: &Path) -> anyhow::Result<SqlitePool> {
    let options = SqliteConnectOptions::new().filename(dir.join(TASKS_DB_FILE));
    Ok(SqlitePool::connect_with(options).await?)
}

async fn refuse_task_writes(dir: &Path) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TRIGGER refuse_tasks BEFORE INSERT ON tasks \
         BEGIN SELECT RAISE(ABORT, 'ledger refuses writes'); END",
    )
    .execute(&ledger_pool(dir).await?)
    .await?;
    Ok(())
}

/// Accept writes again, open the ledger as the next start does, and read
/// the bash task's state.
async fn state_after_restart(dir: &Path) -> anyhow::Result<Option<String>> {
    let pool = ledger_pool(dir).await?;
    sqlx::query("DROP TRIGGER refuse_tasks")
        .execute(&pool)
        .await?;
    TaskLedger::open(&dir.join(TASKS_DB_FILE))
        .await?
        .shutdown()
        .await?;
    Ok(sqlx::query_scalar("SELECT state FROM tasks WHERE name = ?")
        .bind(BASH_COMMAND)
        .fetch_optional(&pool)
        .await?)
}

fn free_port() -> anyhow::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}

async fn wait_until_listening(port: u16, listening: bool) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + WAIT;
    while tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_ok()
        != listening
    {
        anyhow::ensure!(
            tokio::time::Instant::now() < deadline,
            "port {port} listening != {listening}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Ok(())
}

async fn send_signal(pid: Option<u32>, signal: &str) -> anyhow::Result<()> {
    let pid = pid.ok_or_else(|| anyhow::anyhow!("server already exited"))?;
    let status = tokio::process::Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(pid.to_string())
        .status()
        .await?;
    anyhow::ensure!(status.success(), "kill -{signal} {pid} failed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_keeps_a_task_recorded_during_the_drain() -> anyhow::Result<()> {
    let home = tempfile::tempdir()?;
    let workspace = home.path().join("workspace");
    std::fs::create_dir(&workspace)?;
    let log_path = home.path().join("server.log");
    let mut model = FakeModel::start(Some(Reply::BashCall)).await?;
    let port = free_port()?;
    let mut server = tokio::process::Command::new(env!("CARGO_BIN_EXE_caliche_server"))
        .args(["--provider", "ollama", "--model", "fake-model", "--port"])
        .arg(port.to_string())
        .current_dir(&workspace)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home.path())
        .env("XDG_DATA_HOME", home.path().join("xdg"))
        .env("OLLAMA_BASE_URL", &model.base_url)
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(&log_path)?)
        .kill_on_drop(true)
        .spawn()?;
    let log = || std::fs::read_to_string(&log_path).unwrap_or_default();

    wait_until_listening(port, true).await?;
    let ledger = ledger_dir(home.path());
    assert!(
        ledger.join("tasks.db").is_file(),
        "no ledger at {}\n{}",
        ledger.display(),
        log()
    );
    refuse_task_writes(&ledger).await?;

    let chat = tokio::spawn(
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/api/chat"))
            .json(&json!({ "message": "run the probe" }))
            .send(),
    );
    assert_eq!(model.next_request().await?, Reply::BashCall);
    send_signal(server.id(), "TERM").await?;
    wait_until_listening(port, false).await?;
    model.release.notify_one();
    assert_eq!(model.next_request().await?, Reply::Answer);
    let response: Value = chat.await??.json().await?;
    assert_eq!(response["response"], "done");

    let status = tokio::time::timeout(WAIT, server.wait()).await??;
    assert_eq!(status.code(), Some(0), "{}", log());
    assert_eq!(
        state_after_restart(&ledger).await?,
        Some("succeeded".to_owned()),
        "{}",
        log()
    );
    Ok(())
}

#[tokio::test]
async fn no_data_dir_runs_the_server_without_the_ledger() -> anyhow::Result<()> {
    let served = run_with_task_ledger(Err(anyhow::anyhow!("no data directory")), async {
        Ok("served")
    })
    .await?;
    assert_eq!(served, "served");
    Ok(())
}
