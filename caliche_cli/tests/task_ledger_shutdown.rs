//! Every way the CLI binary ends keeps a task recorded before it, even when
//! the ledger database refuses the write: an early exit, SIGTERM, a second
//! Ctrl+C, and a server drain.
//!
//! The binary runs against an in-test model server that asks for one bash
//! call, so the bash tool records a real task.

#![cfg(unix)]

use axum::Json;
use axum::extract::State;
use caliche::persistence::task_ledger::{TASKS_DB_FILE, TaskLedger};
use caliche_cli::shutdown::run_with_task_ledger;
use serde_json::{Value, json};
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
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

/// A temporary `$HOME` for one CLI run, with the CLI's log beside it.
struct Home {
    dir: TempDir,
}

impl Home {
    fn new() -> anyhow::Result<Self> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join("workspace"))?;
        Ok(Self { dir })
    }

    fn ledger_dir(&self) -> PathBuf {
        self.dir.path().join(".caliche")
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.dir.path().join("cli.log")).unwrap_or_default()
    }

    /// The CLI binary with only this home and the fake model in its
    /// environment.
    fn cli(&self, model: &FakeModel) -> anyhow::Result<Command> {
        let mut command = Command::new(env!("CARGO_BIN_EXE_caliche"));
        command
            .current_dir(self.dir.path().join("workspace"))
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.dir.path())
            .env("AGENT_PROVIDER", "ollama")
            .env("OLLAMA_MODEL", "fake-model")
            .env("OLLAMA_BASE_URL", &model.base_url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(self.dir.path().join("cli.log"))?)
            .kill_on_drop(true);
        Ok(command)
    }

    /// `session list` exits through the CLI's early exit; the ledger it opens
    /// gets the trigger that refuses task writes.
    async fn list_sessions_then_refuse_task_writes(
        &self,
        model: &FakeModel,
    ) -> anyhow::Result<ExitStatus> {
        let status = self.cli(model)?.args(["session", "list"]).status().await?;
        assert!(
            self.ledger_dir().join("tasks.db").is_file(),
            "no ledger after session list\n{}",
            self.log()
        );
        refuse_task_writes(&self.ledger_dir()).await?;
        Ok(status)
    }

    /// `--pipe` with one line of input; the turn ends through the CLI's
    /// early exit.
    fn spawn_pipe(&self, model: &FakeModel) -> anyhow::Result<Child> {
        let mut child = self
            .cli(model)?
            .args(["--pipe", "run the probe"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("no stdin"))?;
        tokio::spawn(async move {
            let _ = stdin.write_all(b"input\n").await;
        });
        Ok(child)
    }
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

async fn send_signal(child: &Child, signal: &str) -> anyhow::Result<()> {
    let pid = child
        .id()
        .ok_or_else(|| anyhow::anyhow!("CLI already exited"))?;
    let status = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(pid.to_string())
        .status()
        .await?;
    anyhow::ensure!(status.success(), "kill -{signal} {pid} failed");
    Ok(())
}

async fn exit_code(child: &mut Child) -> anyhow::Result<Option<i32>> {
    Ok(tokio::time::timeout(WAIT, child.wait()).await??.code())
}

#[tokio::test(flavor = "multi_thread")]
async fn early_exit_keeps_a_task_recorded_before_it() -> anyhow::Result<()> {
    let home = Home::new()?;
    let model = FakeModel::start(None).await?;
    let listed = home.list_sessions_then_refuse_task_writes(&model).await?;
    assert_eq!(listed.code(), Some(0), "{}", home.log());

    let piped = home.spawn_pipe(&model)?.wait_with_output();
    let piped = tokio::time::timeout(WAIT, piped).await??;
    assert_eq!(piped.status.code(), Some(0), "{}", home.log());
    assert_eq!(String::from_utf8_lossy(&piped.stdout), "done\n");
    assert_eq!(
        state_after_restart(&home.ledger_dir()).await?,
        Some("succeeded".to_owned()),
        "{}",
        home.log()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_keeps_a_task_recorded_before_it() -> anyhow::Result<()> {
    let home = Home::new()?;
    let mut model = FakeModel::start(Some(Reply::Answer)).await?;
    home.list_sessions_then_refuse_task_writes(&model).await?;

    let mut pipe = home.spawn_pipe(&model)?;
    assert_eq!(model.next_request().await?, Reply::BashCall);
    assert_eq!(model.next_request().await?, Reply::Answer);
    send_signal(&pipe, "TERM").await?;
    assert_eq!(exit_code(&mut pipe).await?, Some(143), "{}", home.log());
    assert_eq!(
        state_after_restart(&home.ledger_dir()).await?,
        Some("succeeded".to_owned()),
        "{}",
        home.log()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn second_ctrl_c_keeps_a_task_recorded_before_it() -> anyhow::Result<()> {
    let home = Home::new()?;
    let mut model = FakeModel::start(Some(Reply::Answer)).await?;
    home.list_sessions_then_refuse_task_writes(&model).await?;

    let mut pipe = home.spawn_pipe(&model)?;
    assert_eq!(model.next_request().await?, Reply::BashCall);
    assert_eq!(model.next_request().await?, Reply::Answer);
    send_signal(&pipe, "INT").await?;
    tokio::time::sleep(Duration::from_millis(400)).await;
    send_signal(&pipe, "INT").await?;
    assert_eq!(exit_code(&mut pipe).await?, Some(130), "{}", home.log());
    assert_eq!(
        state_after_restart(&home.ledger_dir()).await?,
        Some("succeeded".to_owned()),
        "{}",
        home.log()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn server_sigterm_keeps_a_task_recorded_during_the_drain() -> anyhow::Result<()> {
    let home = Home::new()?;
    let mut model = FakeModel::start(Some(Reply::BashCall)).await?;
    let port = free_port()?;
    let mut server = home
        .cli(&model)?
        .args(["--server", "--port"])
        .arg(port.to_string())
        .spawn()?;
    wait_until_listening(port, true).await?;
    assert!(
        home.ledger_dir().join("tasks.db").is_file(),
        "no ledger after server start\n{}",
        home.log()
    );
    refuse_task_writes(&home.ledger_dir()).await?;

    let chat = tokio::spawn(
        reqwest::Client::new()
            .post(format!("http://127.0.0.1:{port}/api/chat"))
            .json(&json!({ "message": "run the probe" }))
            .send(),
    );
    assert_eq!(model.next_request().await?, Reply::BashCall);
    send_signal(&server, "TERM").await?;
    wait_until_listening(port, false).await?;
    model.release.notify_one();
    assert_eq!(model.next_request().await?, Reply::Answer);
    let response: Value = chat.await??.json().await?;
    assert_eq!(response["response"], "done");

    assert_eq!(exit_code(&mut server).await?, Some(0), "{}", home.log());
    assert_eq!(
        state_after_restart(&home.ledger_dir()).await?,
        Some("succeeded".to_owned()),
        "{}",
        home.log()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn no_home_runs_the_cli_without_the_ledger() -> anyhow::Result<()> {
    let ran = run_with_task_ledger(Err(anyhow::anyhow!("no home directory")), async {
        Ok("ran")
    })
    .await?;
    assert_eq!(ran, "ran");
    Ok(())
}
