//! rkyv v1 record encoding for mmap durable streams.
//!
//! Each on-disk record is a length-prefixed blob of [`DurableEventStoredV1`]. The stream file
//! begins with an 8-byte magic (`STRMV001`). Payload and optional metadata round-trip through
//! JSON at the serde layer; the envelope is a stable rkyv layout.

use crate::streams::types::{DurableEvent, DurableEventMetadata, DurableEventPayload};
use anyhow::{Context, Result};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Magic at the start of every mmap stream file (8 bytes, ASCII).
pub const STREAM_FILE_MAGIC: [u8; 8] = *b"STRMV001";

/// Length of [`STREAM_FILE_MAGIC`] in bytes.
pub const STREAM_FILE_HEADER_LEN: u64 = 8;

/// Archived form of [`DurableEvent`] for rkyv-backed mmap records.
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub(crate) struct DurableEventStoredV1 {
    /// Unix timestamp in milliseconds (UTC).
    pub timestamp_unix_ms: i64,
    /// JSON for [`DurableEventMetadata`] or the literal `null`.
    pub metadata_json: String,
    /// JSON for [`DurableEventPayload`] (serde externally tagged enum).
    pub payload_json: String,
}

impl TryFrom<&DurableEvent> for DurableEventStoredV1 {
    type Error = anyhow::Error;

    fn try_from(ev: &DurableEvent) -> Result<Self, Self::Error> {
        let metadata_json = match &ev.metadata {
            Some(m) => serde_json::to_string(m).context("serialize durable event metadata")?,
            None => "null".to_string(),
        };
        let payload_json =
            serde_json::to_string(&ev.payload).context("serialize durable event payload")?;
        Ok(Self {
            timestamp_unix_ms: ev.timestamp.timestamp_millis(),
            metadata_json,
            payload_json,
        })
    }
}

impl TryFrom<DurableEventStoredV1> for DurableEvent {
    type Error = anyhow::Error;

    fn try_from(stored: DurableEventStoredV1) -> Result<Self, Self::Error> {
        let metadata = if stored.metadata_json == "null" {
            None
        } else {
            Some(
                serde_json::from_str::<DurableEventMetadata>(&stored.metadata_json)
                    .context("deserialize durable event metadata")?,
            )
        };
        let payload: DurableEventPayload = serde_json::from_str(&stored.payload_json)
            .context("deserialize durable event payload")?;
        let timestamp = chrono::DateTime::from_timestamp_millis(stored.timestamp_unix_ms)
            .ok_or_else(|| anyhow::anyhow!("invalid durable event timestamp_ms"))?;
        Ok(Self {
            timestamp,
            metadata,
            payload,
        })
    }
}

/// Serialize a [`DurableEvent`] to rkyv bytes (no length prefix).
pub(crate) fn encode_durable_event_record(ev: &DurableEvent) -> Result<Vec<u8>> {
    let stored = DurableEventStoredV1::try_from(ev)?;
    rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
        .map(|b| b.into_vec())
        .map_err(|e| anyhow::anyhow!("rkyv serialize durable event: {e}"))
}

/// Deserialize rkyv bytes into a [`DurableEvent`].
pub(crate) fn decode_durable_event_record(bytes: &[u8]) -> Result<DurableEvent> {
    let stored: DurableEventStoredV1 = rkyv::from_bytes::<_, rkyv::rancor::Error>(bytes)
        .map_err(|e| anyhow::anyhow!("rkyv deserialize durable event: {e}"))?;
    DurableEvent::try_from(stored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::AgentEvent;
    use crate::messages::StreamEvent;
    use crate::streams::types::{DurableEvent, DurableEventMetadata, DurableEventPayload};
    use chrono::{TimeZone, Utc};

    #[test]
    fn durable_event_record_roundtrip_stream_payload() {
        let ts = Utc.with_ymd_and_hms(2026, 3, 24, 12, 0, 0).unwrap();
        let ev = DurableEvent {
            timestamp: ts,
            metadata: Some(DurableEventMetadata {
                session_id: Some("s1".to_string()),
                turn_id: Some(7),
                extra: Some(serde_json::json!({"x": 1})),
            }),
            payload: DurableEventPayload::Stream(StreamEvent::TextChunk {
                content: "hello".to_string(),
            }),
        };
        let bytes = encode_durable_event_record(&ev).expect("encode");
        let got = decode_durable_event_record(&bytes).expect("decode");
        assert_eq!(got.timestamp, ev.timestamp);
        let meta = got.metadata.as_ref().expect("metadata");
        assert_eq!(meta.session_id.as_deref(), Some("s1"));
        assert_eq!(meta.turn_id, Some(7));
        assert_eq!(meta.extra, Some(serde_json::json!({"x": 1})));
        match (&got.payload, &ev.payload) {
            (
                DurableEventPayload::Stream(StreamEvent::TextChunk { content: a }),
                DurableEventPayload::Stream(StreamEvent::TextChunk { content: b }),
            ) => assert_eq!(a, b),
            _ => panic!("payload mismatch"),
        }
    }

    #[test]
    fn durable_event_record_roundtrip_agent_payload() {
        let ts = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let ev = DurableEvent {
            timestamp: ts,
            metadata: None,
            payload: DurableEventPayload::Agent(AgentEvent::Thinking {
                content: "step".to_string(),
                action_id: String::new(),
            }),
        };
        let bytes = encode_durable_event_record(&ev).expect("encode");
        let got = decode_durable_event_record(&bytes).expect("decode");
        assert_eq!(got.timestamp, ev.timestamp);
        assert!(got.metadata.is_none());
        match (&got.payload, &ev.payload) {
            (
                DurableEventPayload::Agent(AgentEvent::Thinking { content: a, .. }),
                DurableEventPayload::Agent(AgentEvent::Thinking { content: b, .. }),
            ) => assert_eq!(a, b),
            _ => panic!("payload mismatch"),
        }
    }

    #[test]
    fn decode_rejects_truncated_bytes() {
        let err = decode_durable_event_record(&[1, 2, 3]).expect_err("truncated");
        assert!(
            err.to_string().contains("rkyv deserialize") || err.to_string().contains("deserialize"),
            "{err}"
        );
    }
}
