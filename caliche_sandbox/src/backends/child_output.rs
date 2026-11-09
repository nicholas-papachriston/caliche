//! Wait for a child and drain its stdout and stderr under one deadline.
//!
//! A background process that inherits the output pipes keeps them open after
//! the child exits, so EOF can arrive long after the child. The deadline covers
//! both the wait and the drain; when it passes, the whole process tree is
//! killed and the output captured so far is returned in an [`ExecTimeout`].
//!
//! Each stream keeps at most [`EXEC_OUTPUT_LIMIT`] bytes. Past that the readers
//! keep reading and discard, so the child never blocks on a full pipe and a
//! command that floods its output cannot grow host memory.
//!
//! A reader still running when `collect` returns stops at its next read and
//! closes its end of the pipe. A writer that escaped the kill then gets
//! `EPIPE` or `SIGPIPE` instead of keeping a host thread busy discarding.

use crate::traits::{EXEC_OUTPUT_LIMIT, ExecResult, ExecTimeout};
use anyhow::Result;
use std::io::{ErrorKind, Read};
use std::process::{Child, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// How often to check whether the child has exited.
const POLL: Duration = Duration::from_millis(50);
/// How long to wait for the pipes to reach EOF after the tree is killed.
const KILL_GRACE: Duration = Duration::from_millis(500);

/// One output stream, capped at [`EXEC_OUTPUT_LIMIT`] bytes.
#[derive(Debug, Default)]
pub(crate) struct CappedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

impl CappedOutput {
    /// Keep as much of `chunk` as fits under the cap and drop the rest.
    pub(crate) fn push(&mut self, chunk: &[u8]) {
        let room = EXEC_OUTPUT_LIMIT.saturating_sub(self.bytes.len());
        let kept = chunk.len().min(room);
        self.bytes.extend_from_slice(&chunk[..kept]);
        self.truncated |= kept < chunk.len();
    }

    /// Decode as lossy UTF-8 and return the text with its truncated flag.
    /// Replacement characters can make the text longer than the bytes, so it
    /// is cut back to the cap at a char boundary.
    pub(crate) fn into_text(self) -> (String, bool) {
        let mut text = String::from_utf8_lossy(&self.bytes).into_owned();
        if text.len() <= EXEC_OUTPUT_LIMIT {
            return (text, self.truncated);
        }
        let mut end = EXEC_OUTPUT_LIMIT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        (text, true)
    }
}

type Buffer = Arc<Mutex<CappedOutput>>;

/// Wait for `child` and read its piped stdout and stderr until both reach EOF,
/// all within `timeout` of `started`. On timeout, or if waiting on the child
/// fails, call `kill_tree`, kill and reap `child`, and return the error; a
/// timeout is an [`ExecTimeout`] with the partial output.
pub(crate) fn collect(
    child: Child,
    started: Instant,
    timeout: Duration,
    kill_tree: impl FnOnce(&Child),
) -> Result<ExecResult> {
    collect_with(child, started, timeout, kill_tree, Child::try_wait)
}

/// [`collect`] with the child polled by `try_wait`. Windows offers no way to
/// make `Child::try_wait` fail on a live process, so its test injects one.
pub(crate) fn collect_with(
    mut child: Child,
    started: Instant,
    timeout: Duration,
    kill_tree: impl FnOnce(&Child),
    try_wait: impl FnMut(&mut Child) -> std::io::Result<Option<ExitStatus>>,
) -> Result<ExecResult> {
    // `eof_tx` lives until the end so the channel stays connected and
    // `recv_timeout` waits out its tick after both readers finish.
    let (eof_tx, eof_rx) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stdout = spawn_reader(child.stdout.take(), eof_tx.clone(), Arc::clone(&stop));
    let stderr = spawn_reader(child.stderr.take(), eof_tx.clone(), Arc::clone(&stop));
    let mut open_pipes = 2_u8;

    let waited = wait_and_drain(
        &mut child,
        try_wait,
        &eof_rx,
        &mut open_pipes,
        started + timeout,
    );
    let result = match waited {
        Ok(Some(status)) if open_pipes == 0 => {
            let (stdout, stdout_truncated) = take_text(&stdout);
            let (stderr, stderr_truncated) = take_text(&stderr);
            Ok(ExecResult {
                stdout,
                stderr,
                stdout_truncated,
                stderr_truncated,
                exit_code: status.code().map(i64::from).unwrap_or(-1),
                duration: started.elapsed(),
            })
        }
        waited => {
            kill_tree(&child);
            let _ = child.kill();
            let _ = child.wait();
            drain_until(&eof_rx, &mut open_pipes, Instant::now() + KILL_GRACE);
            match waited {
                Err(err) => Err(anyhow::Error::new(err).context("wait on child")),
                Ok(_) => {
                    let (stdout, stdout_truncated) = take_text(&stdout);
                    let (stderr, stderr_truncated) = take_text(&stderr);
                    Err(ExecTimeout {
                        timeout,
                        stdout,
                        stderr,
                        stdout_truncated,
                        stderr_truncated,
                    }
                    .into())
                }
            }
        }
    };
    stop.store(true, Ordering::Relaxed);
    drop(eof_tx);
    result
}

/// Poll `child` and count pipe EOFs until the child has exited and both pipes
/// closed, or until `deadline`. Returns the exit status if the child exited,
/// or the error from waiting on it.
fn wait_and_drain(
    child: &mut Child,
    mut try_wait: impl FnMut(&mut Child) -> std::io::Result<Option<ExitStatus>>,
    eof_rx: &Receiver<()>,
    open_pipes: &mut u8,
    deadline: Instant,
) -> std::io::Result<Option<ExitStatus>> {
    let mut status = None;
    loop {
        if status.is_none() {
            status = try_wait(child)?;
        }
        if status.is_some() && *open_pipes == 0 {
            return Ok(status);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(status);
        }
        let tick = if status.is_none() {
            POLL.min(remaining)
        } else {
            remaining
        };
        if eof_rx.recv_timeout(tick).is_ok() {
            *open_pipes -= 1;
        }
    }
}

/// Count pipe EOFs until both pipes closed or `deadline` passes.
fn drain_until(eof_rx: &Receiver<()>, open_pipes: &mut u8, deadline: Instant) {
    while *open_pipes > 0 {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        if eof_rx.recv_timeout(remaining).is_ok() {
            *open_pipes -= 1;
        }
    }
}

/// Read `pipe` to EOF on its own thread into a capped buffer, then send on
/// `eof`. Once `stop` is set the thread exits after its next read and drops
/// the pipe.
///
/// The thread is detached, never joined: a descendant that escaped the kill
/// can hold the pipe open, and joining would block exec on that process. The
/// caller is bounded by the deadline in [`collect`], which stops waiting for
/// EOF; memory is bounded by the cap, since bytes past it are discarded. A
/// reader whose pipe is held open by a silent escaped process stays blocked in
/// `read`, using no CPU, until that process writes or closes it.
fn spawn_reader(
    pipe: Option<impl Read + Send + 'static>,
    eof: Sender<()>,
    stop: Arc<AtomicBool>,
) -> Buffer {
    let buffer = Buffer::default();
    let Some(mut pipe) = pipe else {
        let _ = eof.send(());
        return buffer;
    };
    let sink = Arc::clone(&buffer);
    std::thread::spawn(move || {
        let mut chunk = [0_u8; 8192];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(_) if stop.load(Ordering::Relaxed) => break,
                Ok(n) => sink
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(&chunk[..n]),
                Err(err) if err.kind() == ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let _ = eof.send(());
    });
    buffer
}

fn take_text(buffer: &Buffer) -> (String, bool) {
    std::mem::take(&mut *buffer.lock().unwrap_or_else(PoisonError::into_inner)).into_text()
}

/// SIGKILL the process group of `child`. The caller made `child` the group
/// leader, either with `process_group(0)` or with `setsid` in the child (that
/// sets the group id to the child pid). The kernel does not reuse the group
/// id while any member is alive, even after the leader is reaped.
#[cfg(unix)]
pub(crate) fn kill_process_group(child: &Child) {
    let Ok(pgid) = libc::pid_t::try_from(child.id()) else {
        return;
    };
    // SAFETY: kill takes no pointers and only sends a signal.
    #[expect(unsafe_code, reason = "libc call to signal the exec process group")]
    let _ = unsafe { libc::kill(-pgid, libc::SIGKILL) };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;
    #[cfg(unix)]
    use std::process::{Command, Stdio};

    #[test]
    fn push_never_buffers_more_than_the_limit() {
        let mut output = CappedOutput::default();
        output.push(&vec![b'a'; EXEC_OUTPUT_LIMIT - 1]);
        assert_eq!(
            (output.bytes.len(), output.truncated),
            (EXEC_OUTPUT_LIMIT - 1, false)
        );
        output.push(b"bc");
        assert_eq!(
            (output.bytes.len(), output.truncated),
            (EXEC_OUTPUT_LIMIT, true)
        );
        output.push(&vec![b'd'; EXEC_OUTPUT_LIMIT]);
        assert_eq!(
            (output.bytes.len(), output.bytes.last(), output.truncated),
            (EXEC_OUTPUT_LIMIT, Some(&b'b'), true)
        );
    }

    #[cfg(unix)]
    fn process_alive(pid: libc::pid_t) -> bool {
        // SAFETY: kill takes no pointers; signal 0 only checks for the process.
        #[expect(unsafe_code, reason = "libc call to probe a process")]
        let found = unsafe { libc::kill(pid, 0) };
        found == 0
    }

    #[cfg(unix)]
    #[test]
    fn reader_stops_and_closes_the_pipe_of_a_flooder_that_escaped_the_kill() {
        // The leader exits at once and nothing kills the background `yes`, as
        // if it had escaped the tree kill. It floods stdout until its reader
        // closes the pipe.
        let child = Command::new("sh")
            .args(["-c", "yes & echo $! >&2"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sh");
        let err = collect(child, Instant::now(), Duration::from_secs(1), |_| {})
            .expect_err("a held stdout pipe must end in a timeout");
        let timed_out = err.downcast::<ExecTimeout>().expect("ExecTimeout");
        let pid: libc::pid_t = timed_out.stderr.trim().parse().expect("yes pid on stderr");

        let deadline = Instant::now() + Duration::from_secs(5);
        while process_alive(pid) {
            if Instant::now() >= deadline {
                // SAFETY: kill takes no pointers and only sends a signal.
                #[expect(unsafe_code, reason = "libc call to clean up after a failed test")]
                let _ = unsafe { libc::kill(pid, libc::SIGKILL) };
                panic!(
                    "yes {pid} kept writing: its reader still drains the pipe after exec returned"
                );
            }
            #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[cfg(unix)]
    fn group_alive(pgid: libc::pid_t) -> bool {
        // SAFETY: kill takes no pointers; signal 0 only checks for the group.
        #[expect(unsafe_code, reason = "libc call to probe a process group")]
        let found = unsafe { libc::kill(-pgid, 0) };
        found == 0
    }

    #[cfg(unix)]
    #[test]
    fn failed_wait_kills_the_process_tree_before_returning() {
        // The leader exits at once; a background sleep in its process group
        // keeps stdout open.
        let child = Command::new("sh")
            .args(["-c", "sleep 600 &"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .expect("spawn sh");
        let pgid = libc::pid_t::try_from(child.id()).expect("pid fits pid_t");
        // Reap the leader behind `Child`'s back so `try_wait` fails with ECHILD.
        let mut wstatus = 0;
        // SAFETY: waitpid writes only to `wstatus`, which outlives the call.
        #[expect(unsafe_code, reason = "libc call to reap the child early")]
        let reaped = unsafe { libc::waitpid(pgid, &raw mut wstatus, 0) };
        assert_eq!(reaped, pgid);

        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let outcome = collect(
                child,
                Instant::now(),
                Duration::from_secs(600),
                kill_process_group,
            );
            let _ = tx.send(outcome.map(|_| ()).map_err(|err| format!("{err:#}")));
        });
        let outcome = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("a failed wait must return, not wait out the timeout");
        assert_eq!(
            outcome,
            Err("wait on child: No child processes (os error 10)".to_string())
        );

        let deadline = Instant::now() + Duration::from_secs(5);
        while group_alive(pgid) {
            assert!(
                Instant::now() < deadline,
                "background sleep in group {pgid} survived a failed wait"
            );
            #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}
