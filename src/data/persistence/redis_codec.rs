//! Versioned rkyv payloads for Redis-backed session blobs (`SessionData`, `TrackedSession`).
//!
//! Each value is a fixed magic prefix plus an rkyv body (v1 only).

use crate::data::persistence::SessionData;
use crate::data::persistence::rkyv_envelope::{
    REDIS_SESSION_DATA_MAGIC_V1, REDIS_TRACKED_SESSION_MAGIC_V1, prepend_magic, strip_magic_prefix,
};
use crate::data::persistence::session_registry::{SessionStatus, TrackedSession};
use anyhow::Context;
use chrono::{DateTime, Utc};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use uuid::Uuid;

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
struct SessionDataStoredV1 {
    session_id: String,
    agent_id: String,
    conversation_id: String,
    user_id: Option<String>,
    metadata_json: String,
    created_at_rfc3339: String,
    last_accessed_rfc3339: String,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
struct TrackedSessionStoredV1 {
    id: String,
    name_slug: Option<String>,
    agent_id: String,
    user_id: Option<String>,
    conversation_id: Option<String>,
    created_at_rfc3339: String,
    updated_at_rfc3339: String,
    last_accessed_rfc3339: String,
    metadata_json: String,
    message_count: u32,
    status_tag: u8,
}

const fn session_status_to_tag(s: SessionStatus) -> u8 {
    match s {
        SessionStatus::Active => 0,
        SessionStatus::Archived => 1,
        SessionStatus::Expired => 2,
    }
}

fn session_status_from_tag(tag: u8) -> anyhow::Result<SessionStatus> {
    match tag {
        0 => Ok(SessionStatus::Active),
        1 => Ok(SessionStatus::Archived),
        2 => Ok(SessionStatus::Expired),
        _ => Err(anyhow::anyhow!("invalid session status tag: {tag}")),
    }
}

impl TryFrom<&SessionData> for SessionDataStoredV1 {
    type Error = anyhow::Error;

    fn try_from(s: &SessionData) -> Result<Self, Self::Error> {
        let metadata_json =
            serde_json::to_string(&s.metadata).context("serialize session metadata")?;
        Ok(Self {
            session_id: s.session_id.clone(),
            agent_id: s.agent_id.clone(),
            conversation_id: s.conversation_id.clone(),
            user_id: s.user_id.clone(),
            metadata_json,
            created_at_rfc3339: s.created_at.to_rfc3339(),
            last_accessed_rfc3339: s.last_accessed.to_rfc3339(),
        })
    }
}

impl TryFrom<SessionDataStoredV1> for SessionData {
    type Error = anyhow::Error;

    fn try_from(s: SessionDataStoredV1) -> Result<Self, Self::Error> {
        let metadata =
            serde_json::from_str(&s.metadata_json).context("deserialize session metadata")?;
        let created_at = DateTime::parse_from_rfc3339(&s.created_at_rfc3339)
            .context("parse created_at")?
            .with_timezone(&Utc);
        let last_accessed = DateTime::parse_from_rfc3339(&s.last_accessed_rfc3339)
            .context("parse last_accessed")?
            .with_timezone(&Utc);
        Ok(Self {
            session_id: s.session_id,
            agent_id: s.agent_id,
            conversation_id: s.conversation_id,
            user_id: s.user_id,
            metadata,
            created_at,
            last_accessed,
        })
    }
}

impl TryFrom<&TrackedSession> for TrackedSessionStoredV1 {
    type Error = anyhow::Error;

    fn try_from(s: &TrackedSession) -> Result<Self, Self::Error> {
        let metadata_json =
            serde_json::to_string(&s.metadata).context("serialize tracked metadata")?;
        Ok(Self {
            id: s.id.to_string(),
            name_slug: s.name_slug.clone(),
            agent_id: s.agent_id.clone(),
            user_id: s.user_id.clone(),
            conversation_id: s.conversation_id.clone(),
            created_at_rfc3339: s.created_at.to_rfc3339(),
            updated_at_rfc3339: s.updated_at.to_rfc3339(),
            last_accessed_rfc3339: s.last_accessed.to_rfc3339(),
            metadata_json,
            message_count: s.message_count,
            status_tag: session_status_to_tag(s.status),
        })
    }
}

impl TryFrom<TrackedSessionStoredV1> for TrackedSession {
    type Error = anyhow::Error;

    fn try_from(s: TrackedSessionStoredV1) -> Result<Self, Self::Error> {
        let metadata =
            serde_json::from_str(&s.metadata_json).context("deserialize tracked metadata")?;
        let id = Uuid::parse_str(&s.id).context("parse session uuid")?;
        let created_at = DateTime::parse_from_rfc3339(&s.created_at_rfc3339)
            .context("parse created_at")?
            .with_timezone(&Utc);
        let updated_at = DateTime::parse_from_rfc3339(&s.updated_at_rfc3339)
            .context("parse updated_at")?
            .with_timezone(&Utc);
        let last_accessed = DateTime::parse_from_rfc3339(&s.last_accessed_rfc3339)
            .context("parse last_accessed")?
            .with_timezone(&Utc);
        Ok(Self {
            id,
            name_slug: s.name_slug,
            agent_id: s.agent_id,
            user_id: s.user_id,
            conversation_id: s.conversation_id,
            created_at,
            updated_at,
            last_accessed,
            metadata,
            message_count: s.message_count,
            status: session_status_from_tag(s.status_tag)?,
        })
    }
}

pub fn encode_session_data(session: &SessionData) -> anyhow::Result<Vec<u8>> {
    let stored = SessionDataStoredV1::try_from(session)?;
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
        .map(rkyv::util::AlignedVec::into_vec)
        .map_err(|e| anyhow::anyhow!("rkyv serialize SessionData: {e}"))?;
    Ok(prepend_magic(REDIS_SESSION_DATA_MAGIC_V1, &body))
}

pub fn decode_session_data(bytes: &[u8]) -> anyhow::Result<SessionData> {
    let body = strip_magic_prefix(REDIS_SESSION_DATA_MAGIC_V1, bytes, "Redis SessionData")?;
    let stored: SessionDataStoredV1 = rkyv::from_bytes::<_, rkyv::rancor::Error>(body)
        .map_err(|e| anyhow::anyhow!("rkyv deserialize SessionData: {e}"))?;
    SessionData::try_from(stored)
}

pub fn encode_tracked_session(session: &TrackedSession) -> anyhow::Result<Vec<u8>> {
    let stored = TrackedSessionStoredV1::try_from(session)?;
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
        .map(rkyv::util::AlignedVec::into_vec)
        .map_err(|e| anyhow::anyhow!("rkyv serialize TrackedSession: {e}"))?;
    Ok(prepend_magic(REDIS_TRACKED_SESSION_MAGIC_V1, &body))
}

pub fn decode_tracked_session(bytes: &[u8]) -> anyhow::Result<TrackedSession> {
    let body = strip_magic_prefix(
        REDIS_TRACKED_SESSION_MAGIC_V1,
        bytes,
        "Redis TrackedSession",
    )?;
    let stored: TrackedSessionStoredV1 = rkyv::from_bytes::<_, rkyv::rancor::Error>(body)
        .map_err(|e| anyhow::anyhow!("rkyv deserialize TrackedSession: {e}"))?;
    TrackedSession::try_from(stored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::persistence::rkyv_envelope::{
        REDIS_SESSION_DATA_MAGIC_V1, REDIS_TRACKED_SESSION_MAGIC_V1,
    };
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn session_data_roundtrip() {
        let now = Utc::now();
        let s = SessionData {
            session_id: "sid-1".into(),
            agent_id: "agent-1".into(),
            conversation_id: "conv-1".into(),
            user_id: Some("user-1".into()),
            metadata: serde_json::json!({"k": 1}),
            created_at: now,
            last_accessed: now,
        };
        let bytes = encode_session_data(&s).expect("encode");
        assert!(bytes.starts_with(REDIS_SESSION_DATA_MAGIC_V1));
        let got = decode_session_data(&bytes).expect("decode");
        assert_eq!(got.session_id, s.session_id);
        assert_eq!(got.metadata, s.metadata);
    }

    #[test]
    fn tracked_session_roundtrip() {
        let now = Utc::now();
        let id = Uuid::now_v7();
        let s = TrackedSession {
            id,
            name_slug: Some("my-slug".into()),
            agent_id: "a".into(),
            user_id: None,
            conversation_id: Some("c".into()),
            created_at: now,
            updated_at: now,
            last_accessed: now,
            metadata: serde_json::json!({}),
            message_count: 3,
            status: SessionStatus::Active,
        };
        let bytes = encode_tracked_session(&s).expect("encode");
        assert!(bytes.starts_with(REDIS_TRACKED_SESSION_MAGIC_V1));
        let got = decode_tracked_session(&bytes).expect("decode");
        assert_eq!(got.id, id);
        assert_eq!(got.message_count, 3);
        assert_eq!(got.status, SessionStatus::Active);
    }

    #[test]
    fn tracked_session_roundtrips_boundary_message_count() {
        let now = Utc::now();
        let session = TrackedSession {
            id: Uuid::now_v7(),
            name_slug: None,
            agent_id: String::new(),
            user_id: None,
            conversation_id: None,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            metadata: serde_json::json!({"max": u64::MAX}),
            message_count: u32::MAX,
            status: SessionStatus::Archived,
        };

        let decoded = decode_tracked_session(&encode_tracked_session(&session).expect("encode"))
            .expect("decode");
        assert_eq!(decoded.message_count, u32::MAX);
        assert_eq!(decoded.metadata, session.metadata);
        assert_eq!(decoded.status, SessionStatus::Archived);
    }

    #[test]
    fn decode_session_data_rejects_bad_magic() {
        let err = decode_session_data(b"notmagic").expect_err("bad magic");
        assert!(
            err.to_string().contains("SessionData") || err.to_string().contains("magic"),
            "{err}"
        );
    }

    #[test]
    fn decode_session_data_rejects_truncated() {
        let mut v = REDIS_SESSION_DATA_MAGIC_V1.to_vec();
        v.push(1);
        let err = decode_session_data(&v).expect_err("truncated");
        assert!(
            err.to_string().contains("rkyv") || err.to_string().contains("deserialize"),
            "{err}"
        );
    }
}
