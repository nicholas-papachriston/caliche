//! Newline-delimited JSON-RPC frames.
//!
//! ACP stdout is protocol bytes only. Callers flush each frame as soon as it
//! exists so a turn cannot buffer `session/update` until `session/prompt` returns.

use serde_json::Value;
use std::io::{self, Write};
use std::sync::Mutex;

pub trait AcpFrameSink: Send + Sync {
    fn write_frame(&self, frame: &Value) -> io::Result<()>;
}

pub struct StdFrameSink<W> {
    inner: Mutex<W>,
}

impl<W> StdFrameSink<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

impl<W: Write + Send> AcpFrameSink for StdFrameSink<W> {
    fn write_frame(&self, frame: &Value) -> io::Result<()> {
        let mut payload = serde_json::to_vec(frame).map_err(io::Error::other)?;
        if payload.contains(&b'\n') {
            return Err(io::Error::other(
                "ACP frame contained a raw newline; stdout is one JSON object per line",
            ));
        }
        payload.push(b'\n');
        let mut guard = self.inner.lock().unwrap_or_else(|err| err.into_inner());
        guard.write_all(&payload)?;
        guard.flush()?;
        Ok(())
    }
}
