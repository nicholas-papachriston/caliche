use crate::data::persistence::session_registry::{
    SessionQuery, SessionRegistry, SessionStatus, TrackedSession, generate_session_id,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use deadpool_redis::redis::AsyncCommands;
use deadpool_redis::{Config, Pool, Runtime};
use tracing::instrument;
use uuid::Uuid;

fn i64_to_f64(value: i64) -> f64 {
    const TWO_TO_32: f64 = 4_294_967_296.0;
    let high = i32::try_from(value >> 32).unwrap_or_else(|_| {
        if value.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }
    });
    let low_bits = value & i64::from(u32::MAX);
    let low = u32::try_from(low_bits).unwrap_or(u32::MAX);
    f64::from(high).mul_add(TWO_TO_32, f64::from(low))
}

/// Redis-backed session registry with fast access and time-ordered listing.
#[derive(Debug)]
pub struct RedisSessionRegistry {
    pool: Pool,
    prefix: String,
}

impl RedisSessionRegistry {
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
            prefix: prefix.unwrap_or_default(),
        })
    }

    fn session_key(&self, id: Uuid) -> String {
        format!("{}session:{}", self.prefix, id)
    }

    fn slug_key(&self, slug: &str) -> String {
        format!("{}session:slug:{}", self.prefix, slug)
    }

    fn by_time_key(&self) -> String {
        format!("{}sessions:by_time", self.prefix)
    }

    fn by_access_key(&self) -> String {
        format!("{}sessions:by_access", self.prefix)
    }

    fn score_from_dt(dt: chrono::DateTime<chrono::Utc>) -> f64 {
        i64_to_f64(dt.timestamp_millis())
    }

    /// Persist a session record plus its time/access zset positions and
    /// slug index in a single `MULTI/EXEC` transaction so partial failures
    /// cannot leave the keyspace inconsistent.
    async fn save_session(&self, session: &TrackedSession) -> Result<()> {
        let key = self.session_key(session.id);
        let value = super::redis_codec::encode_tracked_session(session)
            .context("Failed to serialize session")?;

        let by_time = self.by_time_key();
        let by_access = self.by_access_key();
        let time_score = Self::score_from_dt(session.created_at);
        let access_score = Self::score_from_dt(session.last_accessed);
        let id_str = session.id.to_string();

        let mut conn = self.pool.get().await?;

        let mut pipe = deadpool_redis::redis::pipe();
        pipe.atomic()
            .cmd("SET")
            .arg(&key)
            .arg(value)
            .ignore()
            .cmd("ZADD")
            .arg(&by_time)
            .arg(time_score)
            .arg(&id_str)
            .ignore()
            .cmd("ZADD")
            .arg(&by_access)
            .arg(access_score)
            .arg(&id_str)
            .ignore();

        if let Some(slug) = session.name_slug.as_deref() {
            let slug_key = self.slug_key(slug);
            pipe.cmd("SET").arg(slug_key).arg(&id_str).ignore();
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .context("Failed to persist session atomically")?;

        Ok(())
    }

    async fn load_session(&self, id: Uuid) -> Result<Option<TrackedSession>> {
        let key = self.session_key(id);
        let mut conn = self.pool.get().await?;
        let data: Option<Vec<u8>> = conn.get(key).await?;
        let Some(data) = data else {
            return Ok(None);
        };
        let session = super::redis_codec::decode_tracked_session(&data)
            .context("Failed to deserialize session")?;
        Ok(Some(session))
    }

    async fn delete_session_keys(&self, session: &TrackedSession) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let id_str = session.id.to_string();

        let mut pipe = deadpool_redis::redis::pipe();
        pipe.atomic()
            .cmd("DEL")
            .arg(self.session_key(session.id))
            .ignore()
            .cmd("ZREM")
            .arg(self.by_time_key())
            .arg(&id_str)
            .ignore()
            .cmd("ZREM")
            .arg(self.by_access_key())
            .arg(&id_str)
            .ignore();

        if let Some(slug) = session.name_slug.as_deref() {
            pipe.cmd("DEL").arg(self.slug_key(slug)).ignore();
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .context("Failed to delete session atomically")?;

        Ok(())
    }
}

#[async_trait]
impl SessionRegistry for RedisSessionRegistry {
    async fn init(&self) -> Result<()> {
        // No schema to initialize for Redis.
        Ok(())
    }

    #[instrument(skip(self), fields(agent_id = %agent_id, user_id = user_id.unwrap_or("")), err)]
    async fn create(&self, agent_id: &str, user_id: Option<&str>) -> Result<TrackedSession> {
        let id = generate_session_id();
        let now = Utc::now();

        let session = TrackedSession {
            id,
            name_slug: None,
            agent_id: agent_id.to_string(),
            user_id: user_id.map(str::to_string),
            conversation_id: None,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            metadata: serde_json::json!({}),
            message_count: 0,
            status: SessionStatus::Active,
        };

        self.save_session(&session).await?;
        Ok(session)
    }

    async fn create_with_id(
        &self,
        id: Uuid,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<TrackedSession> {
        let now = Utc::now();

        let session = TrackedSession {
            id,
            name_slug: None,
            agent_id: agent_id.to_string(),
            user_id: user_id.map(str::to_string),
            conversation_id: None,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            metadata: serde_json::json!({}),
            message_count: 0,
            status: SessionStatus::Active,
        };

        self.save_session(&session).await?;
        Ok(session)
    }

    #[instrument(skip(self), fields(session_id = %id), err)]
    async fn get(&self, id: Uuid) -> Result<Option<TrackedSession>> {
        self.load_session(id).await
    }

    async fn get_by_slug(&self, slug: &str) -> Result<Option<TrackedSession>> {
        let mut conn = self.pool.get().await?;
        let id_str: Option<String> = conn.get(self.slug_key(slug)).await?;
        let Some(id_str) = id_str else {
            return Ok(None);
        };
        let id = Uuid::parse_str(&id_str).context("Failed to parse session id as UUID")?;
        self.load_session(id).await
    }

    async fn update(&self, session: &TrackedSession) -> Result<()> {
        self.save_session(session).await
    }

    async fn update_slug(&self, id: Uuid, slug: &str) -> Result<()> {
        let mut session = self.load_session(id).await?.context("Session not found")?;

        let old_slug = session.name_slug.clone();
        session.name_slug = Some(slug.to_string());
        session.updated_at = Utc::now();

        // Atomic swap: persist the session (which registers the new slug key)
        // and delete the old slug pointer in a single MULTI/EXEC so observers
        // never see two slugs pointing at the same session.
        let key = self.session_key(session.id);
        let value = super::redis_codec::encode_tracked_session(&session)
            .context("Failed to serialize session")?;

        let by_time = self.by_time_key();
        let by_access = self.by_access_key();
        let time_score = Self::score_from_dt(session.created_at);
        let access_score = Self::score_from_dt(session.last_accessed);
        let id_str = session.id.to_string();
        let new_slug_key = self.slug_key(slug);

        let mut conn = self.pool.get().await?;
        let mut pipe = deadpool_redis::redis::pipe();
        pipe.atomic()
            .cmd("SET")
            .arg(&key)
            .arg(value)
            .ignore()
            .cmd("ZADD")
            .arg(&by_time)
            .arg(time_score)
            .arg(&id_str)
            .ignore()
            .cmd("ZADD")
            .arg(&by_access)
            .arg(access_score)
            .arg(&id_str)
            .ignore()
            .cmd("SET")
            .arg(new_slug_key)
            .arg(&id_str)
            .ignore();

        if let Some(old_slug) = old_slug.as_deref()
            && old_slug != slug
        {
            pipe.cmd("DEL").arg(self.slug_key(old_slug)).ignore();
        }

        pipe.query_async::<()>(&mut conn)
            .await
            .context("Failed to update slug atomically")?;

        Ok(())
    }

    async fn touch(&self, id: Uuid) -> Result<()> {
        let mut session = self.load_session(id).await?.context("Session not found")?;

        let now = Utc::now();
        session.last_accessed = now;
        session.updated_at = now;

        self.save_session(&session).await?;
        Ok(())
    }

    #[instrument(skip(self), fields(session_id = %id), err)]
    async fn delete(&self, id: Uuid) -> Result<()> {
        if let Some(session) = self.load_session(id).await? {
            self.delete_session_keys(&session).await?;
        }
        Ok(())
    }

    /// Scan the time-ordered zset from newest to oldest, applying filters
    /// in-memory. `offset` and `limit` are applied to the **filtered** stream
    /// so pagination lines up with SQL-equivalent behavior.
    #[instrument(skip(self, query), fields(limit = query.limit, offset = query.offset), err)]
    async fn list(&self, query: &SessionQuery) -> Result<Vec<TrackedSession>> {
        let mut conn = self.pool.get().await?;
        let by_time = self.by_time_key();

        let offset = usize::try_from(query.offset.max(0)).unwrap_or(0);
        let limit = usize::try_from(query.limit.max(0)).unwrap_or(0);
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::with_capacity(limit);
        let mut skipped = 0usize;
        let mut start: isize = 0;
        // Conservative page size: large enough to amortize round-trips even
        // under heavy filtering, but bounded so a narrow filter does not blow
        // up memory on large zsets.
        let requested_page_size = ((limit + offset).saturating_mul(4)).clamp(128, 1024);
        let page_size = isize::try_from(requested_page_size).unwrap_or(1024);

        loop {
            let end = start + page_size - 1;
            let ids: Vec<String> = conn.zrevrange(by_time.as_str(), start, end).await?;
            if ids.is_empty() {
                break;
            }

            let parsed_ids: Vec<Uuid> = ids
                .into_iter()
                .filter_map(|id_str| Uuid::parse_str(&id_str).ok())
                .collect();
            if parsed_ids.is_empty() {
                start += page_size;
                continue;
            }

            let session_keys: Vec<String> =
                parsed_ids.iter().map(|id| self.session_key(*id)).collect();
            let payloads: Vec<Option<Vec<u8>>> = deadpool_redis::redis::cmd("MGET")
                .arg(session_keys)
                .query_async(&mut conn)
                .await?;

            for maybe_payload in payloads {
                let Some(payload) = maybe_payload else {
                    continue;
                };
                let session = match super::redis_codec::decode_tracked_session(payload.as_slice()) {
                    Ok(s) => s,
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "skipping session with undecodable payload during list"
                        );
                        continue;
                    }
                };

                if let Some(agent_id) = &query.agent_id
                    && session.agent_id != *agent_id
                {
                    continue;
                }
                if let Some(user_id) = &query.user_id
                    && session.user_id.as_deref() != Some(user_id.as_str())
                {
                    continue;
                }
                if let Some(status) = query.status
                    && session.status != status
                {
                    continue;
                }

                if skipped < offset {
                    skipped += 1;
                    continue;
                }

                sessions.push(session);
                if sessions.len() >= limit {
                    return Ok(sessions);
                }
            }

            start += page_size;
        }

        Ok(sessions)
    }

    async fn archive(&self, id: Uuid) -> Result<()> {
        let mut session = self.load_session(id).await?.context("Session not found")?;

        session.status = SessionStatus::Archived;
        session.updated_at = Utc::now();
        self.save_session(&session).await?;
        Ok(())
    }

    async fn cleanup_expired(&self, older_than: Duration) -> Result<u64> {
        let threshold = Utc::now() - older_than;
        let threshold_score = Self::score_from_dt(threshold);

        let mut conn = self.pool.get().await?;
        let by_access = self.by_access_key();

        let expired_ids: Vec<String> = conn
            .zrangebyscore(by_access.as_str(), 0.0, threshold_score)
            .await?;

        let mut removed = 0u64;
        for id_str in expired_ids {
            let Ok(id) = Uuid::parse_str(&id_str) else {
                continue;
            };
            if let Some(session) = self.load_session(id).await? {
                self.delete_session_keys(&session).await?;
                removed += 1;
            }
        }

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::i64_to_f64;

    #[test]
    #[expect(
        clippy::lossy_float_literal,
        reason = "the literals spell the integer bounds under test"
    )]
    fn i64_to_f64_handles_signed_precision_boundaries() {
        assert_eq!(i64_to_f64(0), 0.0);
        assert_eq!(i64_to_f64(-(1_i64 << 53)), -9_007_199_254_740_992.0);
        assert_eq!(i64_to_f64(i64::MAX), 9_223_372_036_854_775_807.0);
        assert_eq!(i64_to_f64(i64::MIN), -9_223_372_036_854_775_808.0);
    }
}
