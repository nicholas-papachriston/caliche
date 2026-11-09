//! Redis/Valkey Streams backend for durable event streams.
//!
//! Uses Redis Streams (XADD, XREAD, XRANGE, XTRIM) with deadpool-redis.
//! Compatible with both Redis and Valkey.
//! Uses low-level cmd() API for compatibility with deadpool-redis 0.23 (redis 1.0).

use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::types::{DurableEvent, STREAM_START, TrimPolicy};
use anyhow::{Context, Result};
use async_trait::async_trait;
use deadpool_redis::redis::Value;
use deadpool_redis::{Config, Pool, Runtime};
use std::time::Duration;

const EVENT_FIELD: &str = "e";

fn redis_value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::BulkString(b) => Some(String::from_utf8_lossy(b).into_owned()),
        Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

fn parse_xrange_entry(entry: &[Value]) -> Option<(String, String)> {
    let id = redis_value_to_string(entry.first()?)?;
    let pairs = match entry.get(1)? {
        Value::Array(arr) => arr,
        _ => return None,
    };
    for chunk in pairs.chunks(2) {
        if chunk.len() >= 2 {
            let k = redis_value_to_string(&chunk[0])?;
            if k == EVENT_FIELD {
                let v = redis_value_to_string(&chunk[1])?;
                return Some((id, v));
            }
        }
    }
    None
}

/// Parse XREAD response into (id, event) pairs.
fn parse_xread_response(val: Value) -> Vec<(String, DurableEvent)> {
    let mut out = Vec::new();
    let streams = match val {
        Value::Array(s) => s,
        Value::Nil => return out,
        _ => return out,
    };
    for stream in streams {
        let entries_val = match stream {
            Value::Array(mut arr) if arr.len() >= 2 => {
                arr.remove(0); // stream key
                arr.remove(0) // entries array
            }
            _ => continue,
        };
        let entries_arr = match entries_val {
            Value::Array(e) => e,
            _ => continue,
        };
        for entry in entries_arr {
            if let Value::Array(ref row) = entry
                && let Some((id, json)) = parse_xrange_entry(row)
                && let Ok(ev) = serde_json::from_str::<DurableEvent>(&json)
            {
                out.push((id, ev));
            }
        }
    }
    out
}

/// Redis/Valkey Streams backend for durable event streams.
#[derive(Clone, Debug)]
pub struct RedisDurableStreamBackend {
    pool: Pool,
}

impl RedisDurableStreamBackend {
    /// Create a backend from a Redis URL.
    pub fn new(redis_url: &str) -> Result<Self> {
        let cfg = Config::from_url(redis_url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("Failed to create Redis pool for durable streams")?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl DurableStreamBackend for RedisDurableStreamBackend {
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String> {
        let json = serde_json::to_string(event).context("Serialize DurableEvent")?;
        let mut conn = self.pool.get().await?;
        let id: String = deadpool_redis::redis::cmd("XADD")
            .arg(stream_key)
            .arg("*")
            .arg(EVENT_FIELD)
            .arg(&json)
            .query_async(&mut conn)
            .await
            .context("XADD")?;
        Ok(id)
    }

    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let start = if after_id == STREAM_START {
            "-".to_string()
        } else {
            format!("({}", after_id)
        };
        let mut conn = self.pool.get().await?;
        let raw: Value = deadpool_redis::redis::cmd("XRANGE")
            .arg(stream_key)
            .arg(&start)
            .arg("+")
            .arg("COUNT")
            .arg(count)
            .query_async(&mut conn)
            .await
            .context("XRANGE")?;
        let mut out = Vec::new();
        if let Value::Array(entries) = raw {
            for entry in entries {
                if let Value::Array(ref row) = entry
                    && let Some((id, json)) = parse_xrange_entry(row)
                {
                    let ev: DurableEvent =
                        serde_json::from_str(&json).context("Deserialize DurableEvent")?;
                    out.push((id, ev));
                }
            }
        }
        Ok(out)
    }

    async fn read_blocking(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let start_id = if after_id == STREAM_START {
            "0"
        } else {
            after_id
        };
        let block_ms = timeout.map(|d| d.as_millis() as isize).unwrap_or(0);
        let mut conn = self.pool.get().await?;
        let raw: Value = if block_ms > 0 {
            deadpool_redis::redis::cmd("XREAD")
                .arg("BLOCK")
                .arg(block_ms)
                .arg("COUNT")
                .arg(count)
                .arg("STREAMS")
                .arg(stream_key)
                .arg(start_id)
                .query_async(&mut conn)
                .await
                .context("XREAD BLOCK")?
        } else {
            deadpool_redis::redis::cmd("XREAD")
                .arg("COUNT")
                .arg(count)
                .arg("STREAMS")
                .arg(stream_key)
                .arg(start_id)
                .query_async(&mut conn)
                .await
                .context("XREAD")?
        };
        Ok(parse_xread_response(raw))
    }

    async fn trim(&self, stream_key: &str, policy: TrimPolicy) -> Result<u64> {
        let mut conn = self.pool.get().await?;
        let removed: i64 = match policy {
            TrimPolicy::MaxLen(n) => deadpool_redis::redis::cmd("XTRIM")
                .arg(stream_key)
                .arg("MAXLEN")
                .arg("~")
                .arg(n as i64)
                .query_async(&mut conn)
                .await
                .context("XTRIM MAXLEN")?,
            TrimPolicy::MinId(min_id) => deadpool_redis::redis::cmd("XTRIM")
                .arg(stream_key)
                .arg("MINID")
                .arg(min_id)
                .query_async(&mut conn)
                .await
                .context("XTRIM MINID")?,
            TrimPolicy::MaxAgeSecs(secs) => {
                let ts = chrono::Utc::now().timestamp_millis() - (secs as i64 * 1000);
                let min_id = format!("{}-0", ts);
                deadpool_redis::redis::cmd("XTRIM")
                    .arg(stream_key)
                    .arg("MINID")
                    .arg(min_id)
                    .query_async(&mut conn)
                    .await
                    .context("XTRIM MINID (MaxAge)")?
            }
        };
        Ok(removed.max(0) as u64)
    }

    async fn last_id(&self, stream_key: &str) -> Result<Option<String>> {
        let mut conn = self.pool.get().await?;
        let raw: Value = deadpool_redis::redis::cmd("XREVRANGE")
            .arg(stream_key)
            .arg("+")
            .arg("-")
            .arg("COUNT")
            .arg(1)
            .query_async(&mut conn)
            .await
            .context("XREVRANGE")?;
        Ok(match raw {
            Value::Array(mut entries) => entries
                .pop()
                .and_then(|v| {
                    if let Value::Array(row) = v {
                        row.first().cloned()
                    } else {
                        None
                    }
                })
                .and_then(|v| redis_value_to_string(&v)),
            _ => None,
        })
    }

    async fn delete_stream(&self, stream_key: &str) -> Result<()> {
        let mut conn = self.pool.get().await?;
        deadpool_redis::redis::cmd("DEL")
            .arg(stream_key)
            .query_async::<()>(&mut conn)
            .await
            .context("DEL")?;
        Ok(())
    }
}
