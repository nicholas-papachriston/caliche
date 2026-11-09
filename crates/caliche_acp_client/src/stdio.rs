use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStderr, ChildStdin, ChildStdout};
use tokio::sync::{mpsc, oneshot};

use crate::error::AcpClientError;
use crate::notes::deliver;
use crate::rpc::{Incoming, LineClass, WaiterFail, classify_line, encode_frame};
use crate::transcript::Transcript;

pub const PHASE_SPAWNED: u8 = 0;
pub const PHASE_READY: u8 = 1;
pub const PHASE_CLOSED: u8 = 2;

const STDERR_LINES: usize = 32;
const STDERR_LINE_CHARS: usize = 240;

type Pending = HashMap<u64, oneshot::Sender<Result<Value, WaiterFail>>>;

pub struct Io {
    stdin: tokio::sync::Mutex<ChildStdin>,
    pending: StdMutex<Pending>,
    notes_tx: mpsc::Sender<Incoming>,
    next_id: AtomicU64,
    phase: AtomicU8,
    transcript: Option<Arc<Transcript>>,
}

impl Io {
    pub fn new(
        stdin: ChildStdin,
        notes_tx: mpsc::Sender<Incoming>,
        transcript: Option<Arc<Transcript>>,
    ) -> Self {
        Self {
            stdin: tokio::sync::Mutex::new(stdin),
            pending: StdMutex::new(HashMap::new()),
            notes_tx,
            next_id: AtomicU64::new(1),
            phase: AtomicU8::new(PHASE_SPAWNED),
            transcript,
        }
    }

    pub fn phase(&self) -> u8 {
        self.phase.load(Ordering::Acquire)
    }

    pub fn set_phase(&self, phase: u8) {
        self.phase.store(phase, Ordering::Release);
    }

    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, AcpClientError> {
        if self.phase() == PHASE_CLOSED {
            return Err(AcpClientError::ShutDown);
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        lock_pending(&self.pending).insert(id, tx);
        let frame = serde_json::json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "id": id,
            "method": method,
            "params": params,
        });
        if let Err(err) = self.send_frame(&frame).await {
            lock_pending(&self.pending).remove(&id);
            return Err(err);
        }
        let wait = async {
            match rx.await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(fail)) => Err(fail.into()),
                Err(_) => Err(AcpClientError::StdoutClosed),
            }
        };
        match timeout {
            Some(limit) => match tokio::time::timeout(limit, wait).await {
                Ok(result) => result,
                Err(_) => {
                    lock_pending(&self.pending).remove(&id);
                    Err(AcpClientError::TimedOut {
                        method: method.to_owned(),
                    })
                }
            },
            None => wait.await,
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<(), AcpClientError> {
        if self.phase() == PHASE_CLOSED {
            return Err(AcpClientError::ShutDown);
        }
        let frame = serde_json::json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "method": method,
            "params": params,
        });
        self.send_frame(&frame).await
    }

    pub fn fail_all(&self, fail: &WaiterFail) {
        for (_, tx) in lock_pending(&self.pending).drain() {
            let _ = tx.send(Err(fail.clone()));
        }
    }

    async fn send_frame(&self, value: &Value) -> Result<(), AcpClientError> {
        if let Some(transcript) = &self.transcript {
            transcript
                .record_client(value)
                .map_err(|source| AcpClientError::Transcript { source })?;
        }
        let bytes = encode_frame(value).map_err(|source| AcpClientError::Encode {
            what: "frame",
            source,
        })?;
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(&bytes)
            .await
            .map_err(|source| AcpClientError::Io { source })?;
        stdin
            .flush()
            .await
            .map_err(|source| AcpClientError::Io { source })?;
        drop(stdin);
        Ok(())
    }
}

pub async fn read_stdout(stdout: ChildStdout, io: std::sync::Arc<Io>) {
    let mut lines = BufReader::new(stdout).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                if handle_line(&io, &line).await.is_err() {
                    break;
                }
            }
            Ok(None) => {
                io.fail_all(&WaiterFail::Closed);
                io.set_phase(PHASE_CLOSED);
                break;
            }
            Err(err) => {
                io.fail_all(&WaiterFail::Protocol(format!("stdout read failed: {err}")));
                io.set_phase(PHASE_CLOSED);
                break;
            }
        }
    }
}

pub async fn drain_stderr(stderr: ChildStderr, tail: std::sync::Arc<StdMutex<StderrTail>>) {
    let mut lines = BufReader::new(stderr).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => push_stderr(&tail, &line),
            Ok(None) | Err(_) => break,
        }
    }
}

pub type StderrTail = VecDeque<String>;

pub fn stderr_text(tail: &StdMutex<StderrTail>) -> String {
    lock_tail(tail)
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
}

async fn handle_line(io: &Io, line: &str) -> Result<(), ()> {
    if let Some(transcript) = &io.transcript {
        if transcript.record_peer_line(line).is_err() {
            io.fail_all(&WaiterFail::Protocol(
                "ACP transcript write failed".to_owned(),
            ));
            io.set_phase(PHASE_CLOSED);
            return Err(());
        }
    }
    match classify_line(line) {
        LineClass::Ignore => Ok(()),
        LineClass::Fatal(fail) => {
            io.fail_all(&fail);
            io.set_phase(PHASE_CLOSED);
            Err(())
        }
        LineClass::Response { id, body } => {
            complete(io, id, body);
            Ok(())
        }
        LineClass::Incoming(incoming) => {
            deliver(&io.notes_tx, incoming).await;
            Ok(())
        }
        LineClass::Reply(value) => send_reply(io, value).await,
        LineClass::ReplyAndIncoming { reply, incoming } => {
            deliver(&io.notes_tx, incoming).await;
            send_reply(io, reply).await
        }
    }
}

async fn send_reply(io: &Io, value: Value) -> Result<(), ()> {
    match io.send_frame(&value).await {
        Ok(()) => Ok(()),
        Err(err) => {
            io.fail_all(&WaiterFail::Protocol(err.to_string()));
            io.set_phase(PHASE_CLOSED);
            Err(())
        }
    }
}

fn complete(io: &Io, id: u64, body: Result<Value, WaiterFail>) {
    let tx = lock_pending(&io.pending).remove(&id);
    if let Some(tx) = tx {
        let _ = tx.send(body);
    }
}

fn push_stderr(tail: &StdMutex<StderrTail>, line: &str) {
    let mut tail = lock_tail(tail);
    if tail.len() == STDERR_LINES {
        tail.pop_front();
    }
    tail.push_back(clip_chars(line, STDERR_LINE_CHARS));
}

fn clip_chars(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_owned();
    }
    let end = line
        .char_indices()
        .nth(max_chars)
        .map_or(line.len(), |(index, _)| index);
    format!("{}...", &line[..end])
}

fn lock_pending(mutex: &StdMutex<Pending>) -> MutexGuard<'_, Pending> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn lock_tail(mutex: &StdMutex<StderrTail>) -> MutexGuard<'_, StderrTail> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
