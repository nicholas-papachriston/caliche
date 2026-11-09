use crate::gateway::types::{DeliveryPolicy, GatewayOutboundEnvelope};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use tokio::sync::RwLock;

/// Error classification for delivery failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryErrorKind {
    RateLimited,
    TemporaryNetwork,
    Permanent,
    Unauthorized,
    Unknown,
}

/// Attempt metadata for retries and observability.
#[derive(Debug, Clone)]
pub struct DeliveryAttempt {
    pub attempt: u32,
    pub timestamp: DateTime<Utc>,
    pub success: bool,
    pub error: Option<String>,
    pub error_kind: Option<DeliveryErrorKind>,
}

/// Delivery result for one outbound envelope.
#[derive(Debug, Clone)]
pub struct DeliveryResult {
    pub envelope_id: String,
    pub delivered: bool,
    pub attempts: Vec<DeliveryAttempt>,
}

/// Dead-letter record for repeated failures.
#[derive(Debug, Clone)]
pub struct DeliveryOutcome {
    pub envelope: GatewayOutboundEnvelope,
    pub result: DeliveryResult,
}

#[async_trait]
pub trait GatewayDeadLetterStore: Send + Sync {
    async fn push(&self, outcome: DeliveryOutcome) -> Result<()>;
}

/// Simple in-memory DLQ for development/testing and diagnostics.
#[derive(Debug)]
pub struct MemoryDeadLetterStore {
    cap: usize,
    rows: RwLock<VecDeque<DeliveryOutcome>>,
}

impl MemoryDeadLetterStore {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            rows: RwLock::new(VecDeque::new()),
        }
    }

    pub async fn list(&self) -> Vec<DeliveryOutcome> {
        self.rows.read().await.iter().cloned().collect()
    }
}

#[async_trait]
impl GatewayDeadLetterStore for MemoryDeadLetterStore {
    async fn push(&self, outcome: DeliveryOutcome) -> Result<()> {
        let mut rows = self.rows.write().await;
        rows.push_back(outcome);
        while rows.len() > self.cap {
            rows.pop_front();
        }
        Ok(())
    }
}

#[async_trait]
pub trait GatewayDelivery: Send + Sync {
    async fn deliver(
        &self,
        envelope: &GatewayOutboundEnvelope,
        policy: &DeliveryPolicy,
    ) -> Result<DeliveryResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::types::{ChannelKind, OutboundPayload};
    use std::collections::HashMap;

    fn outcome_with_id(outbound_id: &str) -> DeliveryOutcome {
        DeliveryOutcome {
            envelope: GatewayOutboundEnvelope {
                outbound_id: outbound_id.to_string(),
                channel_kind: ChannelKind::Api,
                channel_id: "ch".to_string(),
                adapter_id: "ad".to_string(),
                tenant_id: None,
                user_id: "u".to_string(),
                created_at: Utc::now(),
                payload: OutboundPayload {
                    text: String::new(),
                    attachments: Vec::new(),
                    reply_to_message_id: None,
                },
                metadata: HashMap::new(),
            },
            result: DeliveryResult {
                envelope_id: outbound_id.to_string(),
                delivered: false,
                attempts: Vec::new(),
            },
        }
    }

    #[tokio::test]
    async fn dead_letter_store_cap_zero_becomes_one() {
        let store = MemoryDeadLetterStore::new(0);
        store.push(outcome_with_id("a")).await.expect("push");
        let rows = store.list().await;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].envelope.outbound_id, "a");
    }

    #[tokio::test]
    async fn dead_letter_store_drops_oldest_when_over_cap() {
        let store = MemoryDeadLetterStore::new(2);
        store.push(outcome_with_id("first")).await.expect("p1");
        store.push(outcome_with_id("second")).await.expect("p2");
        store.push(outcome_with_id("third")).await.expect("p3");
        let rows = store.list().await;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].envelope.outbound_id, "second");
        assert_eq!(rows[1].envelope.outbound_id, "third");
    }
}
