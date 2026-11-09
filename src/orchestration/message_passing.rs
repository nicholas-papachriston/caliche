use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;

/// Message priority for queue ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Message passed between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub message_id: String,
    pub from_agent_id: String,
    pub to_agent_id: Option<String>, // None for broadcast
    pub content: String,
    pub priority: MessagePriority,
    pub timestamp: DateTime<Utc>,
    pub metadata: serde_json::Value,
    pub reply_to: Option<String>, // For threading messages
}

impl AgentMessage {
    pub fn new(
        from_agent_id: impl Into<String>,
        to_agent_id: Option<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            message_id: Uuid::now_v7().to_string(),
            from_agent_id: from_agent_id.into(),
            to_agent_id,
            content: content.into(),
            priority: MessagePriority::Normal,
            timestamp: Utc::now(),
            metadata: serde_json::json!({}),
            reply_to: None,
        }
    }

    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn as_reply_to(mut self, message_id: impl Into<String>) -> Self {
        self.reply_to = Some(message_id.into());
        self
    }

    pub fn is_broadcast(&self) -> bool {
        self.to_agent_id.is_none()
    }

    pub fn is_for_agent(&self, agent_id: &str) -> bool {
        self.to_agent_id.as_ref().is_none_or(|to| to == agent_id)
    }
}

/// Event type for message bus subscriptions
#[derive(Debug, Clone)]
pub enum BusEvent {
    Message(AgentMessage),
    AgentJoined(String),
    AgentLeft(String),
}

/// Insert `message` into `queue` in descending-priority order (front = highest priority),
/// evicting low-priority (or, failing that, oldest) messages first when `max_queue_size` would
/// otherwise be exceeded.
fn enqueue_with_priority(
    queue: &mut VecDeque<AgentMessage>,
    message: AgentMessage,
    max_queue_size: usize,
) {
    while queue.len() >= max_queue_size {
        if let Some(pos) = queue
            .iter()
            .position(|m| m.priority == MessagePriority::Low)
        {
            queue.remove(pos);
            tracing::warn!("Dropped low-priority message due to queue overflow");
        } else {
            queue.pop_front();
            tracing::warn!("Dropped oldest message due to queue overflow");
        }
    }

    let insert_pos = queue
        .iter()
        .position(|m| m.priority < message.priority)
        .unwrap_or(queue.len());
    queue.insert(insert_pos, message);
}

/// Message bus for inter-agent communication.
///
/// Messages addressed to a specific agent are queued independently per recipient
/// (`per_agent_queues`) so that publishing to, or polling, one agent's mailbox never contends
/// with another agent's. Broadcast messages (`to_agent_id: None`) remain in one shared queue
/// since any agent may poll for them; consistent with the pre-existing semantics, a broadcast
/// message is delivered to whichever agent polls for it first (`poll_message`/
/// `poll_all_messages`), not fanned out to every agent — use `subscribe()` for true fan-out.
pub struct MessageBus {
    /// Broadcast channel for pub/sub messaging
    tx: broadcast::Sender<BusEvent>,
    /// Per-recipient message queues for reliable, point-to-point delivery.
    per_agent_queues: DashMap<String, RwLock<VecDeque<AgentMessage>>>,
    /// Queue for broadcast (`to_agent_id: None`) messages only.
    broadcast_queue: RwLock<VecDeque<AgentMessage>>,
    /// Maximum queue size (per recipient, and for the broadcast queue) before dropping
    /// low-priority messages.
    max_queue_size: usize,
}

impl MessageBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1000);

        Self {
            tx,
            per_agent_queues: DashMap::new(),
            broadcast_queue: RwLock::new(VecDeque::new()),
            max_queue_size: 10000,
        }
    }

    pub fn with_capacity(capacity: usize, max_queue_size: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);

        Self {
            tx,
            per_agent_queues: DashMap::new(),
            broadcast_queue: RwLock::new(VecDeque::new()),
            max_queue_size,
        }
    }

    /// Publish a message to the bus
    pub async fn publish(&self, message: AgentMessage) {
        match message.to_agent_id.clone() {
            Some(to_agent_id) => {
                let queue_lock = self
                    .per_agent_queues
                    .entry(to_agent_id)
                    .or_insert_with(|| RwLock::new(VecDeque::new()));
                let mut queue = queue_lock.write().await;
                enqueue_with_priority(&mut queue, message.clone(), self.max_queue_size);
            }
            None => {
                let mut queue = self.broadcast_queue.write().await;
                enqueue_with_priority(&mut queue, message.clone(), self.max_queue_size);
            }
        }

        // Broadcast to subscribers.  `SendError` is expected when there are no
        // active receivers — the message is still enqueued above and will be
        // delivered via `poll_message`.
        let _ = self.tx.send(BusEvent::Message(message));
    }

    /// Subscribe to messages on the bus
    pub fn subscribe(&self) -> broadcast::Receiver<BusEvent> {
        self.tx.subscribe()
    }

    /// Get next message from queue for a specific agent: whichever of this agent's own queue
    /// or the shared broadcast queue currently has the higher-priority head message.
    pub async fn poll_message(&self, agent_id: &str) -> Option<AgentMessage> {
        let agent_entry = self.per_agent_queues.get(agent_id);
        let mut agent_guard = match &agent_entry {
            Some(entry) => Some(entry.write().await),
            None => None,
        };
        let mut broadcast_guard = self.broadcast_queue.write().await;

        let agent_priority = agent_guard
            .as_ref()
            .and_then(|q| q.front())
            .map(|m| m.priority);
        let broadcast_priority = broadcast_guard.front().map(|m| m.priority);

        match (agent_priority, broadcast_priority) {
            (Some(ap), Some(bp)) if bp > ap => broadcast_guard.pop_front(),
            (Some(_), _) => agent_guard.as_mut().and_then(|q| q.pop_front()),
            (None, Some(_)) => broadcast_guard.pop_front(),
            (None, None) => None,
        }
    }

    /// Get all pending messages for a specific agent (their own queue plus any currently
    /// queued broadcasts, merged back into priority order).
    pub async fn poll_all_messages(&self, agent_id: &str) -> Vec<AgentMessage> {
        let mut messages = Vec::new();

        if let Some(entry) = self.per_agent_queues.get(agent_id) {
            let mut queue = entry.write().await;
            messages.extend(queue.drain(..));
        }
        {
            let mut broadcast = self.broadcast_queue.write().await;
            messages.extend(broadcast.drain(..));
        }

        messages.sort_by(|a, b| b.priority.cmp(&a.priority));
        messages
    }

    /// Notify that an agent has joined.
    ///
    /// The broadcast send error is intentionally ignored — it only occurs when
    /// there are no active subscribers, which is a normal state.
    pub fn notify_agent_joined(&self, agent_id: impl Into<String>) {
        let _ = self.tx.send(BusEvent::AgentJoined(agent_id.into()));
    }

    /// Notify that an agent has left.
    ///
    /// The broadcast send error is intentionally ignored — it only occurs when
    /// there are no active subscribers, which is a normal state.
    pub fn notify_agent_left(&self, agent_id: impl Into<String>) {
        let _ = self.tx.send(BusEvent::AgentLeft(agent_id.into()));
    }

    /// Get current queue size (across every per-agent queue plus the broadcast queue)
    pub async fn queue_size(&self) -> usize {
        let mut total = self.broadcast_queue.read().await.len();
        for entry in self.per_agent_queues.iter() {
            total += entry.value().read().await.len();
        }
        total
    }

    /// Clear all messages for a specific agent.
    ///
    /// Matches the pre-existing semantics where broadcast messages count as "for" every
    /// agent: clearing one agent's mailbox also clears the shared broadcast queue.
    pub async fn clear_messages_for_agent(&self, agent_id: &str) {
        if let Some(entry) = self.per_agent_queues.get(agent_id) {
            entry.write().await.clear();
        }
        self.broadcast_queue.write().await.clear();
    }

    /// Get message bus statistics
    pub async fn get_stats(&self) -> MessageBusStats {
        let mut priority_counts = [0usize; 4];
        let mut total = 0usize;

        {
            let broadcast = self.broadcast_queue.read().await;
            total += broadcast.len();
            for msg in broadcast.iter() {
                priority_counts[msg.priority as usize] += 1;
            }
        }

        for entry in self.per_agent_queues.iter() {
            let queue = entry.value().read().await;
            total += queue.len();
            for msg in queue.iter() {
                priority_counts[msg.priority as usize] += 1;
            }
        }

        MessageBusStats {
            queue_size: total,
            max_queue_size: self.max_queue_size,
            low_priority: priority_counts[0],
            normal_priority: priority_counts[1],
            high_priority: priority_counts[2],
            critical_priority: priority_counts[3],
            subscriber_count: self.tx.receiver_count(),
        }
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessageBusStats {
    pub queue_size: usize,
    pub max_queue_size: usize,
    pub low_priority: usize,
    pub normal_priority: usize,
    pub high_priority: usize,
    pub critical_priority: usize,
    pub subscriber_count: usize,
}
