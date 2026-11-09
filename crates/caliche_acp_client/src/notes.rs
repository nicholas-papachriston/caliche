//! Bounded queue of ACP session notes.
//!
//! One task owns the receiver for the life of the client. `prompt` only sets
//! the active session. Cancelling a prompt clears that session and leaves the
//! receiver in place, so the next prompt still receives.
//!
//! The stdout reader awaits [`deliver`] when the queue is full, which stops
//! the pipe read so the peer blocks on stdout. The owner task is the only
//! receiver, so a full queue waits until that task takes a note. A note is not
//! dropped by this queue. Idle notes and foreign replay notes each keep at
//! most [`MAX_PARKED_NOTES`] entries, in separate deques, so a burst after a
//! prompt ends cannot evict another session's replay. Passing either cap drops
//! the oldest note of that kind, and a foreign idle note that finds the replay
//! deque full is dropped. Every drop increments [`NotesHub::dropped`]. The
//! first drop of an episode warns once. The next begin or finish warns once
//! more with the count.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, PoisonError};

use caliche_contracts::acp::SessionUpdate;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::error::AcpClientError;
use crate::rpc::Incoming;

/// Session notes waiting to be received.
///
/// 64 is a burst of `session/update` frames between two receives. See the
/// module docs for what happens when the queue is full.
pub(crate) const NOTES_CHANNEL_CAPACITY: usize = 64;

/// Commands from `prompt` to the task that owns the receiver.
///
/// Two slots: `Finish` and a cancel `End` can both be queued. `prompt` waits
/// on `Begin` and `Finish`. `Drop` uses `try_send` and ignores a full queue
/// because a `Finish` or `End` is already queued, and either one clears the
/// active session.
const NOTES_COMMAND_CAPACITY: usize = 2;

/// Cap for the idle deque and, separately, for the foreign-replay deque.
///
/// Older notes of that kind are dropped once that deque reaches this length
/// so a session that is never prompted again cannot grow without bound.
/// Idle notes are dropped before a replay note is dropped.
pub(crate) const MAX_PARKED_NOTES: usize = 32;

/// Notes collected for the session that was active when the prompt finished.
#[derive(Debug, Default)]
pub(crate) struct PromptNotes {
    pub(crate) updates: Vec<SessionUpdate>,
    pub(crate) unrecognized: Vec<Value>,
}

pub(crate) fn notes_channel() -> (mpsc::Sender<Incoming>, mpsc::Receiver<Incoming>) {
    mpsc::channel(NOTES_CHANNEL_CAPACITY)
}

/// Wait until `incoming` is queued. A closed consumer returns without killing
/// the stdout reader, so in-flight responses can still complete.
pub(crate) async fn deliver(tx: &mpsc::Sender<Incoming>, incoming: Incoming) {
    let _ = tx.send(incoming).await;
}

fn push_capped<T>(queue: &mut VecDeque<T>, item: T) -> usize {
    let mut dropped = 0usize;
    while queue.len() >= MAX_PARKED_NOTES {
        queue.pop_front();
        dropped += 1;
    }
    queue.push_back(item);
    dropped
}

enum Cmd {
    Begin {
        session_id: String,
        ack: oneshot::Sender<Result<(), AcpClientError>>,
    },
    Finish {
        ack: oneshot::Sender<PromptNotes>,
    },
    End,
}

/// Live session for one `prompt` call.
///
/// Drop sends `End` so a cancelled prompt does not leave the owner collecting
/// into a future that is gone, and does not drop the notes receiver.
pub(crate) struct PromptLease {
    cmds: mpsc::Sender<Cmd>,
    armed: bool,
}

impl PromptLease {
    /// Drain notes queued before the prompt result and clear the active session.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::StdoutClosed`] when the owner task is gone.
    pub(crate) async fn finish(&mut self) -> Result<PromptNotes, AcpClientError> {
        let notes = finish_cmd(&self.cmds).await?;
        self.armed = false;
        Ok(notes)
    }
}

impl Drop for PromptLease {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.cmds.try_send(Cmd::End);
        }
    }
}

pub(crate) struct NotesHub {
    cmds: mpsc::Sender<Cmd>,
    prompt_gate: Mutex<()>,
    task: StdMutex<Option<JoinHandle<()>>>,
    dropped: Arc<AtomicU64>,
}

impl NotesHub {
    pub(crate) fn new(rx: mpsc::Receiver<Incoming>) -> Arc<Self> {
        let (cmds, cmd_rx) = mpsc::channel(NOTES_COMMAND_CAPACITY);
        let dropped = Arc::new(AtomicU64::new(0));
        let task = tokio::spawn(owner_loop(rx, cmd_rx, Arc::clone(&dropped)));
        Arc::new(Self {
            cmds,
            prompt_gate: Mutex::new(()),
            task: StdMutex::new(Some(task)),
            dropped,
        })
    }

    /// Replay notes discarded because the replay deque was already at [`MAX_PARKED_NOTES`].
    ///
    /// [`crate::AcpClient::parked_notes_dropped`] reads this. It counts idle and
    /// replay notes dropped at either cap.
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub(crate) fn try_prompt_gate(
        &self,
    ) -> Result<tokio::sync::MutexGuard<'_, ()>, AcpClientError> {
        self.prompt_gate
            .try_lock()
            .map_err(|_| AcpClientError::PromptInFlight)
    }

    /// Park notes already queued, then collect for `session_id` until [`PromptLease::finish`].
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::StdoutClosed`] when the notes receiver is closed
    /// or the owner task has stopped.
    pub(crate) async fn begin(&self, session_id: &str) -> Result<PromptLease, AcpClientError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.cmds
            .send(Cmd::Begin {
                session_id: session_id.to_owned(),
                ack: ack_tx,
            })
            .await
            .map_err(|_| AcpClientError::StdoutClosed)?;
        ack_rx.await.map_err(|_| AcpClientError::StdoutClosed)??;
        Ok(PromptLease {
            cmds: self.cmds.clone(),
            armed: true,
        })
    }

    pub(crate) fn shutdown(&self) {
        if let Some(task) = lock_task(&self.task).take() {
            task.abort();
        }
    }
}

async fn finish_cmd(cmds: &mpsc::Sender<Cmd>) -> Result<PromptNotes, AcpClientError> {
    let (ack_tx, ack_rx) = oneshot::channel();
    cmds.send(Cmd::Finish { ack: ack_tx })
        .await
        .map_err(|_| AcpClientError::StdoutClosed)?;
    ack_rx.await.map_err(|_| AcpClientError::StdoutClosed)
}

fn lock_task(
    task: &StdMutex<Option<JoinHandle<()>>>,
) -> std::sync::MutexGuard<'_, Option<JoinHandle<()>>> {
    task.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Owner {
    rx: mpsc::Receiver<Incoming>,
    /// Notes that arrived with no prompt active. The next prompt for that
    /// session discards them. They do not share slots with [`Self::replay`].
    idle: VecDeque<Incoming>,
    /// Foreign-session notes captured during another prompt.
    replay: VecDeque<Incoming>,
    collected: PromptNotes,
    active: Option<String>,
    closed: bool,
    dropped: Arc<AtomicU64>,
    /// Drops since the last begin or finish summary. Zero means the next drop warns.
    episode_drops: u64,
}

async fn owner_loop(
    rx: mpsc::Receiver<Incoming>,
    mut cmds: mpsc::Receiver<Cmd>,
    dropped: Arc<AtomicU64>,
) {
    let mut owner = Owner {
        rx,
        idle: VecDeque::new(),
        replay: VecDeque::new(),
        collected: PromptNotes::default(),
        active: None,
        closed: false,
        dropped,
        episode_drops: 0,
    };
    loop {
        if owner.closed {
            let Some(cmd) = cmds.recv().await else {
                return;
            };
            owner.handle(cmd);
            continue;
        }
        tokio::select! {
            biased;
            cmd = cmds.recv() => {
                let Some(cmd) = cmd else {
                    return;
                };
                owner.handle(cmd);
            }
            incoming = owner.rx.recv() => {
                match incoming {
                    Some(incoming) => owner.on_note(incoming),
                    None => owner.closed = true,
                }
            }
        }
    }
}

impl Owner {
    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Begin { session_id, ack } => self.handle_begin(session_id, ack),
            Cmd::Finish { ack } => {
                let _ = ack.send(self.finish());
            }
            Cmd::End => self.discard_active(),
        }
    }

    fn handle_begin(
        &mut self,
        session_id: String,
        ack: oneshot::Sender<Result<(), AcpClientError>>,
    ) {
        if self.closed {
            let _ = ack.send(Err(AcpClientError::StdoutClosed));
            return;
        }
        self.arm(session_id);
        if ack.send(Ok(())).is_err() {
            self.discard_active();
        }
    }

    fn arm(&mut self, session_id: String) {
        self.collected = PromptNotes::default();
        self.drain_as_idle();
        self.summarize_episode();
        self.shelve(&session_id);
        self.active = Some(session_id);
    }

    fn drain_as_idle(&mut self) {
        while let Ok(note) = self.rx.try_recv() {
            self.park(note, true);
        }
    }

    /// Drop idle notes for `session_id`. Replay notes stay.
    fn shelve(&mut self, session_id: &str) {
        let pending = std::mem::take(&mut self.idle);
        for incoming in pending {
            if note_matches(&incoming, session_id) {
                continue;
            }
            self.idle.push_back(incoming);
        }
    }

    fn finish(&mut self) -> PromptNotes {
        if self.active.is_none() {
            self.summarize_episode();
            return PromptNotes::default();
        }
        while let Ok(note) = self.rx.try_recv() {
            self.on_note(note);
        }
        let Some(session_id) = self.active.take() else {
            self.summarize_episode();
            return PromptNotes::default();
        };
        self.fold_parked(&session_id);
        self.summarize_episode();
        std::mem::take(&mut self.collected)
    }

    fn fold_parked(&mut self, session_id: &str) {
        let replay = std::mem::take(&mut self.replay);
        let idle = std::mem::take(&mut self.idle);
        for incoming in replay {
            self.route(incoming, session_id);
        }
        for incoming in idle {
            self.keep_idle_without_evicting_replay(incoming, session_id);
        }
    }

    /// An idle note must not take a replay slot that is already full.
    fn keep_idle_without_evicting_replay(&mut self, incoming: Incoming, session_id: &str) {
        if note_matches(&incoming, session_id) {
            self.collect(incoming);
            return;
        }
        if self.replay.len() >= MAX_PARKED_NOTES {
            self.record_drops(1);
            return;
        }
        self.replay.push_back(incoming);
    }

    fn discard_active(&mut self) {
        if self.active.is_none() {
            return;
        }
        while let Ok(note) = self.rx.try_recv() {
            self.on_note(note);
        }
        self.collected = PromptNotes::default();
        self.active = None;
    }

    fn on_note(&mut self, incoming: Incoming) {
        let Some(session_id) = self.active.take() else {
            self.park(incoming, true);
            return;
        };
        self.route(incoming, &session_id);
        self.active = Some(session_id);
    }

    fn route(&mut self, incoming: Incoming, session_id: &str) {
        if !note_matches(&incoming, session_id) {
            self.park(incoming, false);
            return;
        }
        self.collect(incoming);
    }

    fn collect(&mut self, incoming: Incoming) {
        match incoming {
            Incoming::Update(update) => self.collected.updates.push(update.update),
            Incoming::Unrecognized(raw) => self.collected.unrecognized.push(raw),
        }
    }

    fn park(&mut self, incoming: Incoming, before_prompt: bool) {
        let evicted = if before_prompt {
            push_capped(&mut self.idle, incoming)
        } else {
            push_capped(&mut self.replay, incoming)
        };
        self.record_drops(evicted);
    }

    fn record_drops(&mut self, evicted: usize) {
        if evicted == 0 {
            return;
        }
        let count = u64::try_from(evicted).unwrap_or(u64::MAX);
        self.dropped.fetch_add(count, Ordering::Relaxed);
        let first = self.episode_drops == 0;
        self.episode_drops = self.episode_drops.saturating_add(count);
        if first {
            tracing::warn!(
                target: "caliche_acp_client::parked_notes",
                dropped = count,
                cap = MAX_PARKED_NOTES,
                "parked ACP notes dropped the oldest"
            );
        }
    }

    fn summarize_episode(&mut self) {
        if self.episode_drops == 0 {
            return;
        }
        let dropped = self.episode_drops;
        self.episode_drops = 0;
        tracing::warn!(
            target: "caliche_acp_client::parked_notes",
            dropped,
            total_dropped = self.dropped.load(Ordering::Relaxed),
            "parked ACP note drop summary"
        );
    }
}

fn note_matches(incoming: &Incoming, session_id: &str) -> bool {
    match incoming {
        Incoming::Update(update) => update.session_id.0.as_ref() == session_id,
        Incoming::Unrecognized(raw) => raw
            .get("sessionId")
            .and_then(Value::as_str)
            .map_or(true, |id| id == session_id),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_PARKED_NOTES, NOTES_CHANNEL_CAPACITY, NotesHub, deliver, notes_channel, push_capped,
    };
    use crate::rpc::Incoming;
    use serde_json::json;
    use serial_test::serial;
    use std::collections::VecDeque;
    use std::sync::Once;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;

    #[tokio::test]
    async fn notes_channel_waits_when_full_and_delivers_every_note() {
        let (tx, mut rx) = notes_channel();
        for n in 0..NOTES_CHANNEL_CAPACITY {
            tx.try_send(Incoming::Unrecognized(json!({"n": n})))
                .expect("fill");
        }

        let extra = u64::try_from(NOTES_CHANNEL_CAPACITY).unwrap_or(0);
        let waiting = tx.clone();
        let mut pending = std::pin::pin!(deliver(
            &waiting,
            Incoming::Unrecognized(json!({"n": extra})),
        ));
        let blocked = tokio::time::timeout(Duration::from_millis(40), &mut pending).await;
        assert!(blocked.is_err(), "a note must wait while the queue is full");

        let mut notes = Vec::new();
        notes.push(rx.recv().await.expect("first"));
        pending.await;
        while let Ok(note) = rx.try_recv() {
            notes.push(note);
        }

        assert_eq!(notes.len(), NOTES_CHANNEL_CAPACITY + 1);
        let numbers: Vec<u64> = notes
            .into_iter()
            .map(|note| match note {
                Incoming::Unrecognized(value) => value["n"].as_u64().unwrap_or(u64::MAX),
                Incoming::Update(_) => panic!("expected an unrecognized note"),
            })
            .collect();
        let expected: Vec<u64> = (0..=extra).collect();
        assert_eq!(numbers, expected);
    }

    #[tokio::test]
    async fn cancelled_prompt_keeps_the_notes_receiver() {
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        let lease = hub.begin("sess-a").await.expect("begin");
        drop(lease);

        let mut lease = hub.begin("sess-b").await.expect("receiver still open");
        deliver(
            &tx,
            Incoming::Unrecognized(json!({"sessionId": "sess-b", "n": 1})),
        )
        .await;
        let notes = lease.finish().await.expect("finish");
        assert_eq!(notes.unrecognized.len(), 1);
        assert_eq!(notes.unrecognized[0]["n"], 1);
    }

    #[tokio::test]
    async fn idle_same_session_note_is_dropped_and_replay_is_kept() {
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        let mut lease = hub.begin("sess-a").await.expect("begin a");
        deliver(
            &tx,
            Incoming::Unrecognized(json!({"sessionId": "sess-b", "text": "replay"})),
        )
        .await;
        let during_a = lease.finish().await.expect("finish a");
        assert!(during_a.unrecognized.is_empty());

        deliver(
            &tx,
            Incoming::Unrecognized(json!({"sessionId": "sess-b", "text": "idle"})),
        )
        .await;
        let mut lease = hub.begin("sess-b").await.expect("begin b");
        deliver(
            &tx,
            Incoming::Unrecognized(json!({"sessionId": "sess-b", "text": "live"})),
        )
        .await;
        let notes = lease.finish().await.expect("finish b");
        let texts: Vec<&str> = notes
            .unrecognized
            .iter()
            .filter_map(|value| value["text"].as_str())
            .collect();
        assert!(texts.contains(&"replay"), "{texts:?}");
        assert!(texts.contains(&"live"), "{texts:?}");
        assert!(!texts.contains(&"idle"), "{texts:?}");
    }

    #[test]
    fn parked_notes_drop_the_oldest_past_the_cap() {
        let mut parked = VecDeque::new();
        let extra = 3;
        let mut dropped = 0;
        for n in 0..(MAX_PARKED_NOTES + extra) {
            dropped += push_capped(&mut parked, n);
        }
        assert_eq!(parked.len(), MAX_PARKED_NOTES);
        assert_eq!(parked.front().copied(), Some(extra));
        assert_eq!(dropped, extra);
    }

    #[tokio::test]
    #[serial(acp_note_warns)]
    async fn parked_notes_count_drops_past_the_cap() {
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        let extra = 3u64;
        let total = u64::try_from(MAX_PARKED_NOTES).unwrap_or(0) + extra;
        let mut lease = hub.begin("sess").await.expect("begin");
        for n in 0..total {
            deliver(
                &tx,
                Incoming::Unrecognized(json!({"sessionId": "other", "n": n})),
            )
            .await;
        }
        let notes = lease.finish().await.expect("finish");
        assert!(notes.unrecognized.is_empty());
        assert_eq!(hub.dropped(), extra);
    }

    #[tokio::test]
    #[serial(acp_note_warns)]
    async fn idle_notes_past_the_cap_are_counted() {
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        for n in 0..(MAX_PARKED_NOTES + 3) {
            deliver(
                &tx,
                Incoming::Unrecognized(json!({"sessionId": "other", "n": n})),
            )
            .await;
        }
        let mut lease = hub.begin("sess").await.expect("begin");
        lease.finish().await.expect("finish");
        assert_eq!(hub.dropped(), 3);
    }

    #[tokio::test]
    #[serial(acp_note_warns)]
    async fn a_foreign_idle_note_that_finds_replay_full_is_counted() {
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        let mut lease = hub.begin("sess-a").await.expect("begin a");
        for n in 0..MAX_PARKED_NOTES {
            deliver(
                &tx,
                Incoming::Unrecognized(json!({"sessionId": "sess-b", "n": n})),
            )
            .await;
        }
        lease.finish().await.expect("finish a");
        assert_eq!(hub.dropped(), 0);

        deliver(
            &tx,
            Incoming::Unrecognized(json!({"sessionId": "sess-b", "late": true})),
        )
        .await;
        let mut lease = hub.begin("sess-a").await.expect("begin a again");
        lease.finish().await.expect("finish a again");
        assert_eq!(hub.dropped(), 1);
    }

    #[tokio::test]
    #[serial(acp_note_warns)]
    async fn late_idle_notes_do_not_evict_foreign_replay() {
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        let mut lease = hub.begin("sess-a").await.expect("begin a");
        for n in 0..5 {
            deliver(
                &tx,
                Incoming::Unrecognized(json!({"sessionId": "sess-b", "n": n})),
            )
            .await;
        }
        let during_a = lease.finish().await.expect("finish a");
        assert!(during_a.unrecognized.is_empty());

        for n in 0..40 {
            deliver(
                &tx,
                Incoming::Unrecognized(json!({"sessionId": "sess-a", "late": n})),
            )
            .await;
        }
        let mut lease = hub.begin("sess-a").await.expect("begin a again");
        let again = lease.finish().await.expect("finish a again");
        assert!(
            again.unrecognized.is_empty(),
            "idle notes for A are discarded"
        );

        let mut lease = hub.begin("sess-b").await.expect("begin b");
        let notes = lease.finish().await.expect("finish b");
        let numbers: Vec<u64> = notes
            .unrecognized
            .iter()
            .filter_map(|value| value["n"].as_u64())
            .collect();
        assert_eq!(numbers, vec![0, 1, 2, 3, 4], "{numbers:?}");
        assert_eq!(hub.dropped(), 8, "40 late idle notes past a cap of 32");
    }

    #[tokio::test]
    #[serial(acp_note_warns)]
    async fn idle_flood_warns_once_then_summarizes_at_begin() {
        install_parked_note_warn_counter();
        WARN_COUNT.store(0, Ordering::SeqCst);
        let (tx, rx) = notes_channel();
        let hub = NotesHub::new(rx);
        for n in 0..10_000 {
            deliver(
                &tx,
                Incoming::Unrecognized(json!({"sessionId": "a", "n": n})),
            )
            .await;
        }
        let _lease = hub.begin("a").await.expect("begin");
        assert_eq!(
            WARN_COUNT.load(Ordering::SeqCst),
            2,
            "one warn at the first drop, one summary at begin"
        );
        assert_eq!(hub.dropped(), 9_968, "10,000 idle notes past a cap of 32");
    }

    static WARN_COUNT: AtomicU64 = AtomicU64::new(0);

    fn install_parked_note_warn_counter() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let subscriber = tracing_subscriber::registry().with(CountParkedWarns);
            let _ = tracing::subscriber::set_global_default(subscriber);
        });
    }

    struct CountParkedWarns;

    impl<S> Layer<S> for CountParkedWarns
    where
        S: tracing::Subscriber,
    {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if event.metadata().target() == "caliche_acp_client::parked_notes"
                && *event.metadata().level() == tracing::Level::WARN
            {
                WARN_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
