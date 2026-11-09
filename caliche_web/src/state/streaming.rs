//! Stream cancellation helpers shared by the native SSE and WebSocket code
//! paths.
//!
//! The core idea: a monotonically-increasing "epoch" counter is bumped any
//! time the operator invalidates a running stream (cancel button, session end,
//! history resume, new send). The stream consumer captures the epoch at start
//! and compares it every chunk; as soon as the live counter diverges, the
//! consumer exits and any further chunks are dropped.
//!
//! Keeping the logic in one pure helper lets the UI wiring stay thin and
//! makes it easy to unit-test without spinning up a Dioxus runtime.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Shared, thread-safe monotonic counter.
#[derive(Debug, Clone, Default)]
pub struct EpochCounter {
    inner: Arc<AtomicU64>,
}

impl EpochCounter {
    /// Observe the current value without modifying it.
    pub fn current(&self) -> u64 {
        self.inner.load(Ordering::Acquire)
    }

    /// Increment the counter and return the new value. Call this any time a
    /// UI action invalidates any in-flight streams (cancel, resume, end
    /// session, new send).
    pub fn bump(&self) -> u64 {
        self.inner.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Capture the current value for use by a single streaming loop.
    pub fn capture(&self) -> StreamTicket {
        StreamTicket {
            captured: self.current(),
            counter: self.inner.clone(),
        }
    }
}

/// One-shot gate checked at every streaming chunk.
#[derive(Debug, Clone)]
pub struct StreamTicket {
    captured: u64,
    counter: Arc<AtomicU64>,
}

impl StreamTicket {
    /// True if the epoch has been bumped since [`EpochCounter::capture`] was
    /// called. Callers should exit their read loop (and discard the chunk) as
    /// soon as this returns true.
    pub fn is_superseded(&self) -> bool {
        self.counter.load(Ordering::Acquire) != self.captured
    }

    /// Counterpart to [`Self::is_superseded`] that returns the captured value
    /// for log messages.
    pub fn captured(&self) -> u64 {
        self.captured
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_session_ignores_late_chunks() {
        let counter = EpochCounter::default();
        let ticket = counter.capture();
        assert!(!ticket.is_superseded());

        // Simulate the user switching sessions (or hitting Cancel).
        let new_epoch = counter.bump();
        assert_eq!(new_epoch, 1);
        assert!(ticket.is_superseded());
    }

    #[test]
    fn fresh_ticket_after_bump_is_valid() {
        let counter = EpochCounter::default();
        counter.bump();
        let ticket = counter.capture();
        assert!(!ticket.is_superseded());
    }

    #[test]
    fn multiple_bumps_still_invalidate_old_ticket() {
        let counter = EpochCounter::default();
        let ticket = counter.capture();
        for _ in 0..5 {
            counter.bump();
        }
        assert!(ticket.is_superseded());
        assert_eq!(counter.current(), 5);
    }
}
