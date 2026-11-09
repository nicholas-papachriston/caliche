use super::{AgentBuilder, Arc, Cow};

impl AgentBuilder {
    pub fn durable_stream_backend(
        mut self,
        backend: Arc<dyn crate::streams::DurableStreamBackend>,
    ) -> Self {
        self.durable_stream_backend = Some(backend);
        self
    }

    /// Use in-memory durable stream with the given capacity (no external deps).
    #[must_use]
    pub fn durable_stream_capacity(mut self, capacity: usize) -> Self {
        self.durable_stream_capacity = Some(capacity);
        self.durable_stream_backend = Some(Arc::new(
            crate::streams::MemoryDurableStreamBackend::new(capacity),
        ));
        self
    }

    /// Use Redis or Valkey Streams for durable event persistence.
    #[cfg(feature = "durable-streams-redis")]
    pub fn durable_stream_redis(mut self, redis_url: &str) -> anyhow::Result<Self> {
        let backend = Arc::new(crate::streams::RedisDurableStreamBackend::new(redis_url)?);
        self.durable_stream_backend = Some(backend);
        Ok(self)
    }

    /// Use SQLite for durable event persistence (single file, no external server).
    /// The backend is created when you call `build()`.
    #[cfg(feature = "durable-streams-sqlite")]
    pub fn durable_stream_sqlite(mut self, db_path: impl Into<String>) -> Self {
        self.durable_stream_sqlite_path = Some(db_path.into());
        self
    }

    /// Use Postgres for durable event persistence. The backend is created when you call `build()`.
    #[cfg(feature = "durable-streams-postgres")]
    pub fn durable_stream_postgres(mut self, connection_string: impl Into<String>) -> Self {
        self.durable_stream_postgres_url = Some(connection_string.into());
        self
    }

    /// Use S3 for durable event persistence (bucket, key prefix, optional region). The backend is created when you call `build()`.
    #[cfg(feature = "durable-streams-s3")]
    pub fn durable_stream_s3(
        mut self,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: Option<impl Into<String>>,
    ) -> Self {
        self.durable_stream_s3_config =
            Some((bucket.into(), prefix.into(), region.map(Into::into)));
        self
    }

    /// Set key prefix for durable stream keys (default "events:").
    pub fn durable_stream_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.durable_stream_key_prefix = Cow::owned(prefix.into());
        self
    }

    /// Set TTL for auto-trim of old events (best-effort).
    #[must_use]
    pub const fn durable_stream_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.durable_stream_ttl = Some(ttl);
        self
    }
}
