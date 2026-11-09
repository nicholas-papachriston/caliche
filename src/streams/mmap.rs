//! Memory-mapped append log backend for durable event streams.
//!
//! Format: 8-byte magic `STRMV001`, then length-prefixed rkyv records ([`super::codec::DurableEventStoredV1`]).
//! Event IDs are file byte offsets (end of each record) for resume, matching other append-only backends.

use super::codec::{
    STREAM_FILE_HEADER_LEN, STREAM_FILE_MAGIC, decode_durable_event_record,
    encode_durable_event_record,
};
use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::types::{DurableEvent, STREAM_START, TrimPolicy};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{Mutex as TokioMutex, RwLock};

const TRIM_READ_BATCH: usize = 256;

/// Shard count for open-handle maps; appends on different keys rarely share a shard lock.
const STREAM_SHARD_COUNT: usize = 64;

type StreamSlot = (File, AtomicU64);
type StreamShard = RwLock<HashMap<String, StreamSlot>>;

fn stream_shard_index(stream_key: &str) -> usize {
    let mut hasher = DefaultHasher::new();
    stream_key.hash(&mut hasher);
    (hasher.finish() as usize) % STREAM_SHARD_COUNT
}

/// Upper bound on a single record payload size (after the 4-byte length prefix) to avoid OOM on corrupt lengths.
const MAX_RECORD_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Memory-mapped (append-only file) backend for durable event streams.
///
/// Uses an `STRMV001` file header and length-prefixed rkyv records. IDs are file byte offsets.
/// Trim rewrites the file (see [`DurableStreamBackend::trim`]); event IDs change after trim.
#[derive(Debug)]
pub struct MmapDurableStreamBackend {
    /// Base directory; stream_key becomes a file name under it.
    base_path: std::path::PathBuf,
    /// Open file handles sharded by `stream_key` hash so unrelated keys append in parallel.
    stream_shards: [StreamShard; STREAM_SHARD_COUNT],
    /// Serialize append / trim / delete per `stream_key` so trim can drop the shard lock
    /// during rewrite without racing another append on the same file.
    stream_ops: RwLock<HashMap<String, std::sync::Arc<TokioMutex<()>>>>,
}

impl MmapDurableStreamBackend {
    /// Create a backend that stores streams as files under `base_path`.
    pub fn new(base_path: impl AsRef<Path>) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path).context("Create mmap stream base dir")?;
        Ok(Self {
            base_path,
            stream_shards: std::array::from_fn(|_| RwLock::new(HashMap::new())),
            stream_ops: RwLock::new(HashMap::new()),
        })
    }

    fn shard_for(&self, stream_key: &str) -> &StreamShard {
        &self.stream_shards[stream_shard_index(stream_key)]
    }

    async fn per_stream_ops_lock(&self, stream_key: &str) -> std::sync::Arc<TokioMutex<()>> {
        {
            let locks = self.stream_ops.read().await;
            if let Some(m) = locks.get(stream_key) {
                return std::sync::Arc::clone(m);
            }
        }
        let mut locks = self.stream_ops.write().await;
        locks
            .entry(stream_key.to_string())
            .or_insert_with(|| std::sync::Arc::new(TokioMutex::new(())))
            .clone()
    }

    fn path_for(&self, stream_key: &str) -> std::path::PathBuf {
        let safe = stream_key.replace(':', "_");
        self.base_path.join(format!("{}.log", safe))
    }

    fn ensure_stream_header(file: &mut File, path: &Path) -> Result<u64> {
        let len = file
            .metadata()
            .with_context(|| format!("stream file metadata {:?}", path))?
            .len();
        if len == 0 {
            file.seek(SeekFrom::Start(0))
                .with_context(|| format!("seek start {:?}", path))?;
            file.write_all(&STREAM_FILE_MAGIC)
                .with_context(|| format!("write STRMV001 header {:?}", path))?;
            file.sync_all()
                .with_context(|| format!("sync stream header {:?}", path))?;
            return Ok(STREAM_FILE_HEADER_LEN);
        }
        if len < STREAM_FILE_HEADER_LEN {
            anyhow::bail!(
                "mmap stream file {:?} is too short for STRMV001 header (got {} bytes)",
                path,
                len
            );
        }
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("seek for magic {:?}", path))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .with_context(|| format!("read stream magic {:?}", path))?;
        if magic != STREAM_FILE_MAGIC {
            anyhow::bail!(
                "mmap stream file {:?} is not rkyv v1 (expected STRMV001 at offset 0; remove legacy JSON logs or use a new path)",
                path
            );
        }
        file.seek(SeekFrom::End(0))
            .with_context(|| format!("seek end {:?}", path))?;
        Ok(len)
    }

    async fn write_offset(&self, stream_key: &str) -> Result<(File, u64)> {
        let path = self.path_for(stream_key);
        let shard = self.shard_for(stream_key);
        let mut streams = shard.write().await;
        if let Some(entry) = streams.get_mut(stream_key) {
            let pos = entry.1.load(Ordering::Relaxed);
            return Ok((entry.0.try_clone().context("Clone file handle")?, pos));
        }
        let existed_before_open = path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .append(!existed_before_open)
            .open(&path)
            .context("Open mmap stream file")?;

        let offset = Self::ensure_stream_header(&mut file, &path)?;

        streams.insert(
            stream_key.to_string(),
            (
                file.try_clone().context("Clone file")?,
                AtomicU64::new(offset),
            ),
        );
        Ok((file, offset))
    }

    fn validate_stream_file_header(file: &mut File, path: &Path, meta_len: u64) -> Result<()> {
        if meta_len > 0 && meta_len < STREAM_FILE_HEADER_LEN {
            anyhow::bail!(
                "mmap stream file {:?} is too short for STRMV001 header (got {} bytes)",
                path,
                meta_len
            );
        }
        if meta_len < STREAM_FILE_HEADER_LEN {
            return Ok(());
        }
        file.seek(SeekFrom::Start(0))
            .with_context(|| format!("seek magic {:?}", path))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .with_context(|| format!("read stream magic {:?}", path))?;
        if magic != STREAM_FILE_MAGIC {
            anyhow::bail!(
                "mmap stream file {:?} is not rkyv v1 (expected STRMV001 at offset 0)",
                path
            );
        }
        Ok(())
    }

    fn read_one_record(file: &mut File, pos: &mut u64) -> Result<Option<(String, DurableEvent)>> {
        let mut len_buf = [0u8; 4];
        if file.read_exact(&mut len_buf).is_err() {
            return Ok(None);
        }
        let len = u32::from_le_bytes(len_buf) as u64;
        if len > MAX_RECORD_PAYLOAD_BYTES {
            anyhow::bail!(
                "stream record length {} exceeds maximum {} bytes",
                len,
                MAX_RECORD_PAYLOAD_BYTES
            );
        }
        let len_usize = usize::try_from(len).context("stream record length does not fit usize")?;
        *pos += 4;
        let mut buf = vec![0u8; len_usize];
        if file.read_exact(&mut buf).is_err() {
            return Ok(None);
        }
        *pos += len;
        let ev = decode_durable_event_record(&buf).context("Decode mmap durable stream record")?;
        Ok(Some((pos.to_string(), ev)))
    }

    fn read_records(
        path: &Path,
        after_offset: u64,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut file = File::open(path).context("Open stream file for read")?;
        let meta_len = file
            .metadata()
            .with_context(|| format!("stream file metadata {:?}", path))?
            .len();
        Self::validate_stream_file_header(&mut file, path, meta_len)?;

        let mut pos = if after_offset == 0 {
            STREAM_FILE_HEADER_LEN
        } else {
            after_offset
        };
        file.seek(SeekFrom::Start(pos)).context("Seek")?;
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            match Self::read_one_record(&mut file, &mut pos)? {
                Some(record) => out.push(record),
                None => break,
            }
        }
        Ok(out)
    }

    /// Load all events with their end-offset IDs (same IDs as [`Self::read_after`]).
    fn scan_all_events(path: &Path) -> Result<Vec<(u64, DurableEvent)>> {
        let mut out = Vec::new();
        let mut offset = 0u64;
        loop {
            let batch = Self::read_records(path, offset, TRIM_READ_BATCH)?;
            if batch.is_empty() {
                break;
            }
            let mut last_end = offset;
            for (id_s, ev) in batch {
                let end = id_s
                    .parse::<u64>()
                    .context("Parse stream record id as u64")?;
                out.push((end, ev));
                last_end = end;
            }
            if last_end == offset {
                break;
            }
            offset = last_end;
        }
        Ok(out)
    }

    /// Rewrite the stream file from scratch (used after trim). Drops old byte offsets.
    fn rewrite_stream_file(path: &Path, events: &[DurableEvent]) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Create stream directory for trim")?;
        }
        let tmp = path.with_extension("log.tmptrim");
        let write_result = (|| -> Result<()> {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)
                .context("Open trim temp file")?;
            f.write_all(&STREAM_FILE_MAGIC)
                .context("Write STRMV001 header")?;
            for ev in events {
                let body =
                    encode_durable_event_record(ev).context("Encode event for trim rewrite")?;
                let len_u32 =
                    u32::try_from(body.len()).context("trimmed record exceeds u32::MAX")?;
                f.write_all(&len_u32.to_le_bytes())
                    .context("Write record length")?;
                f.write_all(&body).context("Write record body")?;
            }
            f.sync_all().context("Sync trim temp file")?;
            Ok(())
        })();

        if write_result.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return write_result;
        }

        #[cfg(unix)]
        {
            if let Err(e) = std::fs::rename(&tmp, path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(e).context("Atomic replace stream file after trim");
            }
        }
        #[cfg(not(unix))]
        {
            if path.exists() {
                std::fs::remove_file(path).context("Remove stream file before trim replace")?;
            }
            if let Err(e) = std::fs::rename(&tmp, path) {
                let _ = std::fs::remove_file(&tmp);
                return Err(e).context("Replace stream file after trim");
            }
        }
        Ok(())
    }
}

#[async_trait]
impl DurableStreamBackend for MmapDurableStreamBackend {
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String> {
        let op = self.per_stream_ops_lock(stream_key).await;
        let _op_guard = op.lock().await;
        let body = encode_durable_event_record(event).context("Encode DurableEvent for mmap")?;
        let (mut file, start) = self.write_offset(stream_key).await?;
        let len = u32::try_from(body.len()).context("DurableEvent record exceeds u32::MAX")?;
        file.write_all(&len.to_le_bytes()).context("Write length")?;
        file.write_all(&body).context("Write payload")?;
        file.sync_all().context("Sync")?;
        let end = start + 4 + body.len() as u64;
        let shard = self.shard_for(stream_key);
        if let Some(entry) = shard.write().await.get_mut(stream_key) {
            entry.1.store(end, Ordering::Relaxed);
        }
        Ok(end.to_string())
    }

    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let start = if after_id == STREAM_START {
            0u64
        } else {
            after_id.parse().context("Parse after_id as offset")?
        };
        let path = self.path_for(stream_key);
        Self::read_records(&path, start, count)
    }

    async fn read_blocking(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let start = if after_id == STREAM_START {
            0u64
        } else {
            after_id.parse().context("Parse after_id as offset")?
        };
        let path = self.path_for(stream_key);
        let deadline = timeout.map(|d| std::time::Instant::now() + d);
        loop {
            let out = Self::read_records(&path, start, count)?;
            if !out.is_empty() {
                return Ok(out);
            }
            if let Some(dead) = deadline
                && std::time::Instant::now() >= dead
            {
                return Ok(Vec::new());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn trim(&self, stream_key: &str, policy: TrimPolicy) -> Result<u64> {
        let path = self.path_for(stream_key);
        let op = self.per_stream_ops_lock(stream_key).await;
        let _op_guard = op.lock().await;

        if !path.exists() {
            return Ok(0);
        }

        let scanned = Self::scan_all_events(&path)?;
        let total = scanned.len();
        if total == 0 {
            return Ok(0);
        }

        let kept: Vec<DurableEvent> = match &policy {
            TrimPolicy::MaxLen(max) => {
                let keep = *max;
                if total <= keep {
                    return Ok(0);
                }
                scanned
                    .into_iter()
                    .rev()
                    .take(keep)
                    .map(|(_, e)| e)
                    .rev()
                    .collect()
            }
            TrimPolicy::MinId(min_id_s) => {
                let threshold = min_id_s.parse::<u64>().with_context(|| {
                    format!(
                        "mmap TrimPolicy::MinId expects numeric byte-offset event id, got {min_id_s:?}"
                    )
                })?;
                scanned
                    .into_iter()
                    .filter(|(id, _)| *id >= threshold)
                    .map(|(_, e)| e)
                    .collect()
            }
            TrimPolicy::MaxAgeSecs(secs) => {
                let secs_i64 = (*secs).min(i64::MAX as u64) as i64;
                let cutoff = chrono::Utc::now() - chrono::Duration::seconds(secs_i64);
                scanned
                    .into_iter()
                    .filter(|(_, e)| e.timestamp >= cutoff)
                    .map(|(_, e)| e)
                    .collect()
            }
        };

        let removed = (total - kept.len()) as u64;
        if removed == 0 {
            return Ok(0);
        }

        {
            let shard = self.shard_for(stream_key);
            let mut streams = shard.write().await;
            streams.remove(stream_key);
        }

        let rewrite = Self::rewrite_stream_file(&path, &kept);
        if rewrite.is_err() {
            let tmp = path.with_extension("log.tmptrim");
            let _ = std::fs::remove_file(&tmp);
        }
        rewrite?;
        Ok(removed)
    }

    async fn last_id(&self, stream_key: &str) -> Result<Option<String>> {
        {
            let shard = self.shard_for(stream_key);
            let streams = shard.read().await;
            if let Some((_, offset)) = streams.get(stream_key) {
                let pos = offset.load(Ordering::Relaxed);
                return Ok(if pos > STREAM_FILE_HEADER_LEN {
                    Some(pos.to_string())
                } else {
                    None
                });
            }
        }
        let path = self.path_for(stream_key);
        if !path.exists() {
            return Ok(None);
        }
        let mut f = File::open(&path).context("Open for last_id")?;
        let len = f.seek(SeekFrom::End(0)).context("Seek end")?;
        Ok(if len > STREAM_FILE_HEADER_LEN {
            Some(len.to_string())
        } else {
            None
        })
    }

    async fn delete_stream(&self, stream_key: &str) -> Result<()> {
        let op = self.per_stream_ops_lock(stream_key).await;
        let _op_guard = op.lock().await;
        let path = self.path_for(stream_key);
        let shard = self.shard_for(stream_key);
        let mut streams = shard.write().await;
        streams.remove(stream_key);
        drop(streams);
        if path.exists() {
            std::fs::remove_file(&path).context("Remove stream file")?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "durable-streams-mmap"))]
mod mmap_trim_tests {
    use super::*;
    use crate::messages::StreamEvent;
    use crate::streams::DurableStreamBackend;
    use crate::streams::types::{DurableEvent, DurableEventPayload, TrimPolicy};
    use chrono::{Duration, Utc};

    fn ev_at(ts: chrono::DateTime<Utc>, tag: &str) -> DurableEvent {
        DurableEvent {
            timestamp: ts,
            metadata: None,
            payload: DurableEventPayload::Stream(StreamEvent::TextChunk {
                content: tag.to_string(),
            }),
        }
    }

    #[tokio::test]
    async fn trim_max_len_drops_oldest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = MmapDurableStreamBackend::new(dir.path()).expect("backend");
        let sk = "stream_a";
        for i in 0..5 {
            let ev = ev_at(Utc::now(), &format!("m{i}"));
            backend.append(sk, &ev).await.expect("append");
        }
        let removed = backend.trim(sk, TrimPolicy::MaxLen(2)).await.expect("trim");
        assert_eq!(removed, 3);
        let rows = backend
            .read_after(sk, STREAM_START, 100)
            .await
            .expect("read");
        assert_eq!(rows.len(), 2);
        assert!(rows[0].1.as_stream_event().is_some());
    }

    #[tokio::test]
    async fn trim_max_age_drops_old() {
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = MmapDurableStreamBackend::new(dir.path()).expect("backend");
        let sk = "stream_b";
        let old = ev_at(Utc::now() - Duration::hours(48), "old");
        let new = ev_at(Utc::now(), "new");
        backend.append(sk, &old).await.expect("a1");
        backend.append(sk, &new).await.expect("a2");
        let removed = backend
            .trim(sk, TrimPolicy::MaxAgeSecs(3600))
            .await
            .expect("trim");
        assert_eq!(removed, 1);
        let rows = backend
            .read_after(sk, STREAM_START, 10)
            .await
            .expect("read");
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].1.as_stream_event().and_then(|s| match s {
                StreamEvent::TextChunk { content } => Some(content.as_ref()),
                _ => None,
            }),
            Some("new")
        );
    }
}
