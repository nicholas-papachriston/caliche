use super::{PersistenceBackend, SessionData};
use anyhow::{Context, Result};
use async_trait::async_trait;
use deadpool_redis::redis::AsyncCommands;
use deadpool_redis::{Config, Pool, Runtime};
use serde::{Serialize, de::DeserializeOwned};
use tracing::instrument;

/// Default cap on the number of session IDs returned by
/// [`RedisSessionStore::list_user_sessions`] when the caller doesn't provide
/// an explicit limit.
pub const DEFAULT_USER_SESSIONS_LIMIT: usize = 1_000;

/// Batch size used when SCAN-ing the session keyspace. Larger values reduce
/// round-trips but increase the size of each intermediate `MGET`.
const USER_SESSIONS_SCAN_COUNT: usize = 200;

/// Redis-backed session store with fast access and TTL support
pub struct RedisSessionStore {
    pool: Pool,
    prefix: String,
}

impl std::fmt::Debug for RedisSessionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisSessionStore")
            .field("pool", &"Pool")
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl RedisSessionStore {
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub fn new(redis_url: &str, prefix: Option<String>) -> Result<Self> {
        let cfg = Config::from_url(redis_url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("Failed to create Redis pool")?;

        Ok(Self {
            pool,
            prefix: prefix.unwrap_or_else(|| "session:".to_owned()),
        })
    }

    fn get_key(&self, key: &str) -> String {
        format!("{}{key}", self.prefix)
    }

    /// Save session data
    #[instrument(skip(self, session), fields(session_id = %session.session_id, ttl_seconds), err)]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn save_session(&self, session: &SessionData, ttl_seconds: u64) -> Result<()> {
        let key = self.get_key(&session.session_id);
        let value = super::redis_codec::encode_session_data(session)?;

        self.set(&key, &value, Some(ttl_seconds)).await
    }

    /// Get session data
    #[instrument(skip(self), fields(session_id = %session_id), err)]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get_session(&self, session_id: &str) -> Result<Option<SessionData>> {
        let key = self.get_key(session_id);

        if let Some(data) = self.get(&key).await? {
            let session = super::redis_codec::decode_session_data(&data)?;
            Ok(Some(session))
        } else {
            Ok(None)
        }
    }

    /// Update session last accessed time
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn touch_session(&self, session_id: &str, ttl_seconds: u64) -> Result<()> {
        let key = self.get_key(session_id);
        self.expire(&key, ttl_seconds).await
    }

    /// Delete session
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let key = self.get_key(session_id);
        self.delete(&key).await
    }

    /// List session IDs for a user, stopping once `limit` matches have been
    /// found.
    ///
    /// `limit = None` falls back to [`DEFAULT_USER_SESSIONS_LIMIT`] so callers
    /// don't accidentally `MGET` an unbounded fleet of sessions. If you truly
    /// need every session for a user, pass `Some(usize::MAX)` explicitly.
    ///
    /// Uses `SCAN` + batched `MGET` so memory usage stays bounded regardless
    /// of the keyspace size.
    #[instrument(skip(self), fields(user_id = %user_id, limit = ?limit), err)]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn list_user_sessions(
        &self,
        user_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<String>> {
        let cap = limit.unwrap_or(DEFAULT_USER_SESSIONS_LIMIT);
        if cap == 0 {
            return Ok(Vec::new());
        }
        let mut conn = self.pool.get().await?;
        let pattern = format!("{}*", self.prefix);
        let mut cursor: u64 = 0;
        let mut user_sessions = Vec::new();

        'scan: loop {
            let (next_cursor, batch): (u64, Vec<String>) = deadpool_redis::redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(USER_SESSIONS_SCAN_COUNT)
                .query_async(&mut conn)
                .await?;

            if !batch.is_empty() {
                let values: Vec<Option<Vec<u8>>> = deadpool_redis::redis::cmd("MGET")
                    .arg(&batch)
                    .query_async(&mut conn)
                    .await?;

                for maybe_data in values {
                    if let Some(data) = maybe_data
                        && let Ok(session) =
                            super::redis_codec::decode_session_data(data.as_slice())
                        && session.user_id.as_ref().is_some_and(|id| id == user_id)
                    {
                        user_sessions.push(session.session_id);
                        if user_sessions.len() >= cap {
                            break 'scan;
                        }
                    }
                }
            }

            if next_cursor == 0 {
                break;
            }
            cursor = next_cursor;
        }

        Ok(user_sessions)
    }

    /// Store generic JSON data with TTL
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn set_json<T: Serialize + Sync>(
        &self,
        key: &str,
        value: &T,
        ttl: Option<u64>,
    ) -> Result<()> {
        let data = serde_json::to_vec(value)?;
        self.set(key, &data, ttl).await
    }

    /// Get generic JSON data
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        if let Some(data) = self.get(key).await? {
            let value = serde_json::from_slice(&data)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Atomic increment
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn increment(&self, key: &str) -> Result<i64> {
        let mut conn = self.pool.get().await?;
        let key = self.get_key(key);
        let value: i64 = conn.incr(key, 1).await?;
        Ok(value)
    }

    /// Atomic decrement
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn decrement(&self, key: &str) -> Result<i64> {
        let mut conn = self.pool.get().await?;
        let key = self.get_key(key);
        let value: i64 = conn.decr(key, 1).await?;
        Ok(value)
    }

    /// Get Redis connection statistics
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get_stats(&self) -> Result<RedisStats> {
        let mut conn = self.pool.get().await?;

        let info: String = deadpool_redis::redis::cmd("INFO")
            .arg("stats")
            .query_async(&mut conn)
            .await?;

        // Parse basic stats from INFO response
        let mut total_commands = 0;
        let mut connected_clients = 0;

        for line in info.lines() {
            if line.starts_with("total_commands_processed:") {
                total_commands = line.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
            } else if line.starts_with("connected_clients:") {
                connected_clients = line.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
            }
        }

        Ok(RedisStats {
            pool_size: self.pool.status().size,
            available_connections: self.pool.status().available,
            total_commands,
            connected_clients,
        })
    }
}

#[async_trait]
impl PersistenceBackend for RedisSessionStore {
    async fn set(&self, key: &str, value: &[u8], ttl: Option<u64>) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let full_key = self.get_key(key);

        if let Some(ttl_secs) = ttl {
            conn.set_ex::<_, _, ()>(&full_key, value, ttl_secs).await?;
        } else {
            conn.set::<_, _, ()>(&full_key, value).await?;
        }

        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let mut conn = self.pool.get().await?;
        let full_key = self.get_key(key);

        let value: Option<Vec<u8>> = conn.get(&full_key).await?;
        Ok(value)
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let full_key = self.get_key(key);

        conn.del::<_, ()>(&full_key).await?;
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let mut conn = self.pool.get().await?;
        let full_key = self.get_key(key);

        let exists: bool = conn.exists(&full_key).await?;
        Ok(exists)
    }

    async fn expire(&self, key: &str, ttl: u64) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let full_key = self.get_key(key);
        let ttl = i64::try_from(ttl).context("Redis TTL exceeds i64 range")?;

        conn.expire::<_, ()>(&full_key, ttl).await?;
        Ok(())
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RedisStats {
    pub pool_size: usize,
    pub available_connections: usize,
    pub total_commands: u64,
    pub connected_clients: u64,
}
