//! rkyv encoding for [`SessionSnapshot`] files (canonical on-disk persistence for `BinaryFileSessionStore`).
//!
//! On-disk layout: 8-byte magic followed by a single rkyv blob.
//! Large fields are JSON strings inside the rkyv struct for stable evolution without re-deriving rkyv for every message type.

use std::path::Path;

use anyhow::{Context, Result};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use super::{SessionMetadata, SessionSnapshot};
use crate::harness::SessionDiffState;

/// Magic prefix for binary session snapshot files.
///
/// Bumped to `CLXSSN03` when durable per-message diff state was added.
pub const SESSION_SNAPSHOT_FILE_MAGIC: [u8; 8] = *b"CLXSSN03";

const SESSION_SNAPSHOT_FILE_MAGIC_V2: [u8; 8] = *b"CLXSSN02";

/// Previous rkyv row for [`SessionSnapshot`] (CLXSSN02, no diff state).
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct SessionSnapshotStoredV2 {
    pub session_id: String,
    pub conversation_history_json: String,
    pub memory_snapshot_json: String,
    pub knowledge_snapshot_json: String,
    pub plan_document_json: String,
    pub shared_state_json: String,
    pub metadata_json: String,
}

/// Current rkyv row for [`SessionSnapshot`] (CLXSSN03).
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct SessionSnapshotStoredV3 {
    pub session_id: String,
    pub conversation_history_json: String,
    pub memory_snapshot_json: String,
    pub knowledge_snapshot_json: String,
    pub plan_document_json: String,
    pub diff_state_json: String,
    pub shared_state_json: String,
    pub metadata_json: String,
}

impl TryFrom<&SessionSnapshot> for SessionSnapshotStoredV3 {
    type Error = anyhow::Error;

    fn try_from(s: &SessionSnapshot) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: s.session_id.clone(),
            conversation_history_json: serde_json::to_string(&s.conversation_history)
                .context("serialize conversation_history")?,
            memory_snapshot_json: serde_json::to_string(&s.memory_snapshot)
                .context("serialize memory_snapshot")?,
            knowledge_snapshot_json: serde_json::to_string(&s.knowledge_snapshot)
                .context("serialize knowledge_snapshot")?,
            plan_document_json: serde_json::to_string(&s.plan_document)
                .context("serialize plan_document")?,
            diff_state_json: serde_json::to_string(&s.diff_state)
                .context("serialize diff_state")?,
            shared_state_json: serde_json::to_string(&s.shared_state)
                .context("serialize shared_state")?,
            metadata_json: serde_json::to_string(&s.metadata).context("serialize metadata")?,
        })
    }
}

impl TryFrom<SessionSnapshotStoredV2> for SessionSnapshot {
    type Error = anyhow::Error;

    fn try_from(s: SessionSnapshotStoredV2) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: s.session_id,
            conversation_history: serde_json::from_str(&s.conversation_history_json)
                .context("deserialize conversation_history")?,
            memory_snapshot: serde_json::from_str(&s.memory_snapshot_json)
                .context("deserialize memory_snapshot")?,
            knowledge_snapshot: serde_json::from_str(&s.knowledge_snapshot_json)
                .context("deserialize knowledge_snapshot")?,
            plan_document: serde_json::from_str(&s.plan_document_json)
                .context("deserialize plan_document")?,
            diff_state: SessionDiffState::default(),
            shared_state: serde_json::from_str(&s.shared_state_json)
                .context("deserialize shared_state")?,
            metadata: serde_json::from_str::<SessionMetadata>(&s.metadata_json)
                .context("deserialize metadata")?,
        })
    }
}

impl TryFrom<SessionSnapshotStoredV3> for SessionSnapshot {
    type Error = anyhow::Error;

    fn try_from(s: SessionSnapshotStoredV3) -> Result<Self, Self::Error> {
        Ok(Self {
            session_id: s.session_id,
            conversation_history: serde_json::from_str(&s.conversation_history_json)
                .context("deserialize conversation_history")?,
            memory_snapshot: serde_json::from_str(&s.memory_snapshot_json)
                .context("deserialize memory_snapshot")?,
            knowledge_snapshot: serde_json::from_str(&s.knowledge_snapshot_json)
                .context("deserialize knowledge_snapshot")?,
            plan_document: serde_json::from_str(&s.plan_document_json)
                .context("deserialize plan_document")?,
            diff_state: serde_json::from_str(&s.diff_state_json)
                .context("deserialize diff_state")?,
            shared_state: serde_json::from_str(&s.shared_state_json)
                .context("deserialize shared_state")?,
            metadata: serde_json::from_str::<SessionMetadata>(&s.metadata_json)
                .context("deserialize metadata")?,
        })
    }
}

/// Encode a snapshot to bytes (magic + rkyv).
pub fn encode_session_snapshot_v1(snap: &SessionSnapshot) -> Result<Vec<u8>> {
    let stored = SessionSnapshotStoredV3::try_from(snap)?;
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
        .map(rkyv::util::AlignedVec::into_vec)
        .map_err(|e| anyhow::anyhow!("rkyv serialize session snapshot: {e}"))?;
    let mut out = Vec::with_capacity(SESSION_SNAPSHOT_FILE_MAGIC.len() + body.len());
    out.extend_from_slice(&SESSION_SNAPSHOT_FILE_MAGIC);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode bytes into a snapshot. CLXSSN03 is current; CLXSSN02 decodes with empty diff state.
pub fn decode_session_snapshot_v1(bytes: &[u8]) -> Result<SessionSnapshot> {
    if bytes.len() < SESSION_SNAPSHOT_FILE_MAGIC.len() {
        anyhow::bail!("session snapshot too short for CLXSSN03 header");
    }
    if bytes[..SESSION_SNAPSHOT_FILE_MAGIC.len()] == SESSION_SNAPSHOT_FILE_MAGIC {
        let stored: SessionSnapshotStoredV3 =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes[SESSION_SNAPSHOT_FILE_MAGIC.len()..])
                .map_err(|e| anyhow::anyhow!("rkyv deserialize session snapshot: {e}"))?;
        return SessionSnapshot::try_from(stored);
    }
    if bytes[..SESSION_SNAPSHOT_FILE_MAGIC.len()] == SESSION_SNAPSHOT_FILE_MAGIC_V2 {
        let stored: SessionSnapshotStoredV2 =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(&bytes[SESSION_SNAPSHOT_FILE_MAGIC.len()..])
                .map_err(|e| anyhow::anyhow!("rkyv deserialize session snapshot: {e}"))?;
        return SessionSnapshot::try_from(stored);
    }
    anyhow::bail!("expected binary session magic CLXSSN03 or CLXSSN02 at offset 0");
}

/// Load a binary rkyv session snapshot from disk.
pub fn load_session_snapshot_from_path(path: &Path) -> Result<SessionSnapshot> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read session file {}", path.display()))?;
    decode_session_snapshot_v1(&bytes).with_context(|| {
        format!(
            "parse binary session snapshot {} (expected CLXSSN03/CLXSSN02 rkyv)",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::MemorySnapshot;
    use crate::harness::{FileSnapshot, MessageDiff};
    use crate::messages::Message;
    use chrono::{DateTime, TimeZone, Utc};
    use std::io::Write;
    use uuid::Uuid;

    fn fixed_time(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
    ) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("invalid fixed test time"))
    }

    fn sample_snapshot() -> Result<SessionSnapshot> {
        Ok(SessionSnapshot {
            session_id: "sess-test-1".to_string(),
            conversation_history: vec![Message::user("hello"), Message::assistant("hi there")],
            memory_snapshot: Some(MemorySnapshot {
                messages: vec![Message::system("ctx")],
                metadata: serde_json::json!({"m": true}),
            }),
            knowledge_snapshot: None,
            plan_document: None,
            diff_state: SessionDiffState::default(),
            shared_state: Some(serde_json::json!({"flag": 1})),
            metadata: SessionMetadata {
                created_at: fixed_time(2025, 6, 1, 10, 0)?,
                saved_at: fixed_time(2025, 6, 1, 10, 30)?,
                memory_type: "buffer".to_string(),
                total_messages: 2,
                total_tokens_estimated: 42,
            },
        })
    }

    #[test]
    fn session_snapshot_v1_roundtrip() -> Result<()> {
        let mut snap = sample_snapshot()?;
        let diff_id = Uuid::now_v7();
        snap.diff_state.active.push(MessageDiff {
            message_id: diff_id,
            snapshots: vec![FileSnapshot {
                path: "src/lib.rs".into(),
                before: Some(b"old".to_vec()),
                after: Some(b"new".to_vec()),
            }],
        });
        let bytes = encode_session_snapshot_v1(&snap)?;
        assert_eq!(&bytes[..8], &SESSION_SNAPSHOT_FILE_MAGIC);
        let got = decode_session_snapshot_v1(&bytes)?;
        assert_eq!(got.session_id, snap.session_id);
        assert_eq!(
            got.conversation_history.len(),
            snap.conversation_history.len()
        );
        assert_eq!(
            got.conversation_history[0].as_text(),
            snap.conversation_history[0].as_text()
        );
        assert!(got.memory_snapshot.is_some());
        assert!(got.knowledge_snapshot.is_none());
        assert!(got.plan_document.is_none());
        assert_eq!(got.diff_state.active.len(), 1);
        assert_eq!(got.diff_state.active[0].message_id, diff_id);
        assert_eq!(got.metadata.total_tokens_estimated, 42);
        Ok(())
    }

    #[test]
    fn session_snapshot_v2_fallback_decodes_empty_diff_state() -> Result<()> {
        let snap = sample_snapshot()?;
        let stored = SessionSnapshotStoredV2 {
            session_id: snap.session_id.clone(),
            conversation_history_json: serde_json::to_string(&snap.conversation_history)?,
            memory_snapshot_json: serde_json::to_string(&snap.memory_snapshot)?,
            knowledge_snapshot_json: serde_json::to_string(&snap.knowledge_snapshot)?,
            plan_document_json: serde_json::to_string(&snap.plan_document)?,
            shared_state_json: serde_json::to_string(&snap.shared_state)?,
            metadata_json: serde_json::to_string(&snap.metadata)?,
        };
        let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
            .map_err(|e| anyhow::anyhow!("encode v2: {e}"))?
            .into_vec();
        let mut bytes = Vec::with_capacity(SESSION_SNAPSHOT_FILE_MAGIC_V2.len() + body.len());
        bytes.extend_from_slice(&SESSION_SNAPSHOT_FILE_MAGIC_V2);
        bytes.extend_from_slice(&body);

        let got = decode_session_snapshot_v1(&bytes)?;

        assert_eq!(got.session_id, snap.session_id);
        assert!(got.diff_state.active.is_empty());
        assert!(got.diff_state.reverted.is_empty());
        Ok(())
    }

    #[test]
    fn load_path_binary_only() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let bin_path = dir.path().join("snap.cls1");
        let snap = sample_snapshot()?;
        std::fs::write(&bin_path, encode_session_snapshot_v1(&snap)?)?;
        let from_bin = load_session_snapshot_from_path(&bin_path)?;
        assert_eq!(from_bin.session_id, snap.session_id);
        let json_path = dir.path().join("snap.json");
        let mut jf = std::fs::File::create(&json_path)?;
        serde_json::to_writer(&mut jf, &snap)?;
        jf.flush()?;
        assert!(load_session_snapshot_from_path(&json_path).is_err());
        Ok(())
    }

    #[test]
    fn decode_rejects_non_magic() {
        let err = decode_session_snapshot_v1(b"notmagic").expect_err("bad magic");
        assert!(err.to_string().contains("CLXSSN03") || err.to_string().contains("short"));
    }
}
