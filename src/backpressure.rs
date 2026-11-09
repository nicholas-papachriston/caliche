//! Bounded queues whose full behavior is explicit.
//!
//! The file watcher uses [`CoalescedSignal`]: the notify callback is synchronous,
//! so it cannot wait on an async channel. The signal has one slot. A full slot
//! means the processor already has a wake queued. Each path keeps its newest
//! value in the map, so a full signal does not drop a path.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

use tokio::sync::mpsc;

/// Wakeups for a coalescing producer.
///
/// Capacity is 1 because the payload lives in the map. One queued wake is
/// enough for the consumer to drain every path recorded since the last drain.
pub const WATCH_SIGNAL_CAPACITY: usize = 1;

/// Newest value per path, plus a single wake signal.
pub struct CoalescedSignal<T> {
    latest: Mutex<HashMap<PathBuf, T>>,
    signal: mpsc::Sender<()>,
}

impl<T> CoalescedSignal<T> {
    /// Create the map and a 1-slot wake channel.
    #[must_use]
    pub fn channel() -> (Self, mpsc::Receiver<()>) {
        let (signal, rx) = mpsc::channel(WATCH_SIGNAL_CAPACITY);
        (
            Self {
                latest: Mutex::new(HashMap::new()),
                signal,
            },
            rx,
        )
    }

    /// Record `value` for `path` and wake the consumer if it is idle.
    ///
    /// A later value replaces the one already stored for `path`. A full signal
    /// is left as-is: the pending wake will drain this path. A closed signal
    /// means the consumer is gone; the path stays in the map until this value
    /// is dropped.
    ///
    /// The file watcher calls [`Self::push_keeping`] instead, so a non-test
    /// build that does not index files does not call this method.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the file watcher calls push_keeping; tests call this"
        )
    )]
    pub fn push_path(&self, path: PathBuf, value: T) {
        self.push_keeping(path, value, |_previous, incoming| incoming);
    }

    /// Record `incoming` for `path`, folded with the value already stored.
    ///
    /// `keep` sees the previous value, if any, and the new one. The wake rules
    /// are the same as [`Self::push_path`].
    pub fn push_keeping<F>(&self, path: PathBuf, incoming: T, keep: F)
    where
        F: FnOnce(Option<T>, T) -> T,
    {
        {
            let mut latest = self.latest.lock().unwrap_or_else(PoisonError::into_inner);
            let previous = latest.remove(&path);
            latest.insert(path, keep(previous, incoming));
        }
        match self.signal.try_send(()) {
            Ok(()) | Err(mpsc::error::TrySendError::Full(())) => {}
            Err(mpsc::error::TrySendError::Closed(())) => {
                tracing::warn!("coalesced signal closed; path retained until drop");
            }
        }
    }

    /// Take every path recorded since the last drain.
    #[must_use]
    pub fn drain(&self) -> Vec<(PathBuf, T)> {
        self.latest
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .drain()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::CoalescedSignal;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    #[tokio::test]
    async fn coalesced_signal_keeps_every_path_when_the_wake_is_full() {
        let (signal, mut wake) = CoalescedSignal::channel();
        signal.push_path(PathBuf::from("a.rs"), "modify");
        signal.push_path(PathBuf::from("a.rs"), "remove");
        signal.push_path(PathBuf::from("b.rs"), "create");

        assert!(wake.try_recv().is_ok(), "one wake is queued");
        assert!(
            wake.try_recv().is_err(),
            "later paths share the pending wake"
        );

        let drained: HashMap<_, _> = signal.drain().into_iter().collect();
        assert_eq!(drained.get(Path::new("a.rs")).copied(), Some("remove"));
        assert_eq!(drained.get(Path::new("b.rs")).copied(), Some("create"));
        assert_eq!(drained.len(), 2);
    }

    #[tokio::test]
    async fn push_keeping_can_retain_an_older_value_when_the_wake_is_full() {
        let (signal, mut wake) = CoalescedSignal::channel();
        let keep_write = |previous: Option<&'static str>, next: &'static str| match next {
            "access" => previous.unwrap_or(next),
            other => other,
        };
        signal.push_keeping(PathBuf::from("a.rs"), "modify", keep_write);
        signal.push_keeping(PathBuf::from("a.rs"), "access", keep_write);

        assert!(wake.try_recv().is_ok(), "one wake is queued");
        let drained: HashMap<_, _> = signal.drain().into_iter().collect();
        assert_eq!(drained.get(Path::new("a.rs")).copied(), Some("modify"));
    }
}
