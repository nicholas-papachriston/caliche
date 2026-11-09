use crate::data::persistence::schema::DatabaseSchema;
use moka::future::Cache;
use std::time::{Duration, Instant};

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CachedSchema {
    schema: DatabaseSchema,
    cached_at: Instant,
    ttl_override: Option<Duration>,
}

/// Schema cache with TTL support
#[derive(Debug, Clone)]
pub struct SchemaCache {
    cache: Cache<String, CachedSchema>,
    default_ttl: Duration,
}

impl SchemaCache {
    /// Create a new schema cache with default TTL
    #[must_use]
    pub fn new(default_ttl_secs: u64) -> Self {
        let default_ttl = Duration::from_secs(default_ttl_secs);
        let cache = Cache::builder()
            .max_capacity(100)
            .time_to_live(default_ttl)
            .build();

        Self { cache, default_ttl }
    }

    /// Get a cached schema if it exists and is not expired
    pub async fn get(&self, connection_name: &str) -> Option<DatabaseSchema> {
        self.get_with_ttl_check(connection_name, Some(self.default_ttl))
            .await
    }

    /// Get a cached schema if it exists and not expired, with optional TTL override.
    /// When `ttl_override` is `Some(d)`, the entry is considered expired if
    /// `cached_at + d` is in the past, and is invalidated. When `None`, the
    /// entry is returned if present (moka's global TTL still applies).
    pub async fn get_with_ttl_check(
        &self,
        connection_name: &str,
        ttl_override: Option<Duration>,
    ) -> Option<DatabaseSchema> {
        let cached = self.cache.get(connection_name).await?;
        let effective_ttl = cached.ttl_override.or(ttl_override);
        if let Some(ttl) = effective_ttl
            && cached.cached_at.elapsed() > ttl
        {
            self.cache.invalidate(connection_name).await;
            return None;
        }
        Some(cached.schema)
    }

    /// Store a schema in the cache
    pub async fn put(&self, connection_name: impl Into<String>, schema: DatabaseSchema) {
        self.put_with_ttl(connection_name.into(), schema, None)
            .await;
    }

    /// Store a schema with an optional custom TTL (seconds). When `Some(secs)`,
    /// the entry can be validated via `get_with_ttl_check` with that TTL.
    /// The cache's global TTL is unchanged; use `get_with_ttl_check` with the
    /// same `ttl_secs` to enforce per-request TTL.
    pub async fn put_with_ttl(
        &self,
        connection_name: impl Into<String>,
        schema: DatabaseSchema,
        ttl_secs: Option<u64>,
    ) {
        let cached = CachedSchema {
            schema,
            cached_at: Instant::now(),
            ttl_override: ttl_secs.map(Duration::from_secs),
        };
        self.cache.insert(connection_name.into(), cached).await;
    }

    /// Invalidate a specific connection's schema cache
    pub async fn invalidate(&self, connection_name: &str) {
        self.cache.invalidate(connection_name).await;
    }

    /// Invalidate all cached schemas
    pub async fn invalidate_all(&self) {
        std::future::ready(()).await;
        self.cache.invalidate_all();
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let metrics = self.cache.weighted_size();
        CacheStats {
            entry_count: usize::try_from(self.cache.entry_count()).unwrap_or(usize::MAX),
            weighted_size: metrics,
        }
    }

    /// Check if a schema is cached and not expired.
    ///
    /// Unlike `Cache::contains_key` this applies the same TTL / per-entry
    /// override check as [`Self::get`] so callers never see stale entries
    /// reported as present. Expired entries are invalidated as a side
    /// effect so a subsequent `get` path does not have to repeat the work.
    pub async fn contains(&self, connection_name: &str) -> bool {
        self.get_with_ttl_check(connection_name, Some(self.default_ttl))
            .await
            .is_some()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entry_count: usize,
    pub weighted_size: u64,
}

impl Default for SchemaCache {
    fn default() -> Self {
        Self::new(300) // 5 minutes default TTL
    }
}
