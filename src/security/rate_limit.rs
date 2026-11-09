use dashmap::DashMap;
#[cfg(feature = "redis")]
use deadpool_redis::Pool as RedisPool;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub requests_per_window: u32,
    pub window_duration: Duration,
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 100,
            window_duration: Duration::from_secs(60),
            burst_size: 10,
        }
    }
}

impl RateLimitConfig {
    /// Create a builder for RateLimitConfig
    pub fn builder() -> RateLimitConfigBuilder {
        RateLimitConfigBuilder::default()
    }

    /// Create configuration from environment variables.
    ///
    /// Currently uses default values as there are no specific env vars for RateLimitConfig.
    /// Use TenantRateLimitConfig::from_env() for tenant-specific rate limiting.
    pub fn from_env() -> Self {
        Self::default()
    }
}

/// Builder for RateLimitConfig
#[derive(Debug, Default)]
pub struct RateLimitConfigBuilder {
    requests_per_window: Option<u32>,
    window_duration: Option<Duration>,
    burst_size: Option<u32>,
}

impl RateLimitConfigBuilder {
    /// Set requests per window
    pub fn requests_per_window(mut self, requests: u32) -> Self {
        self.requests_per_window = Some(requests);
        self
    }

    /// Set window duration
    pub fn window_duration(mut self, duration: Duration) -> Self {
        self.window_duration = Some(duration);
        self
    }

    /// Set burst size
    pub fn burst_size(mut self, size: u32) -> Self {
        self.burst_size = Some(size);
        self
    }

    /// Build the configuration
    pub fn build(self) -> RateLimitConfig {
        let defaults = RateLimitConfig::default();
        RateLimitConfig {
            requests_per_window: self
                .requests_per_window
                .unwrap_or(defaults.requests_per_window),
            window_duration: self.window_duration.unwrap_or(defaults.window_duration),
            burst_size: self.burst_size.unwrap_or(defaults.burst_size),
        }
    }
}

struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    config: RateLimitConfig,
}

impl TokenBucket {
    fn new(config: RateLimitConfig) -> Self {
        Self {
            tokens: config.requests_per_window as f64,
            last_refill: Instant::now(),
            config,
        }
    }

    fn try_consume(&mut self) -> bool {
        self.refill();

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let refill_rate =
            self.config.requests_per_window as f64 / self.config.window_duration.as_secs_f64();

        self.tokens += elapsed.as_secs_f64() * refill_rate;
        self.tokens = self.tokens.min(self.config.requests_per_window as f64);
        self.last_refill = now;
    }
}

pub struct RateLimiter {
    buckets: Arc<DashMap<String, TokenBucket>>,
    default_config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            buckets: Arc::new(DashMap::new()),
            default_config: config,
        }
    }

    pub fn check_rate_limit(&self, key: &str) -> bool {
        let mut entry = self
            .buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(self.default_config.clone()));

        entry.try_consume()
    }

    pub fn reset(&self, key: &str) {
        self.buckets.remove(key);
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

/// Per-tenant rate limiter configuration
#[derive(Debug, Clone)]
pub struct TenantRateLimitConfig {
    pub requests_per_minute: u32,
    pub tokens_per_day: u64,
    pub concurrent_requests: u32,
}

impl Default for TenantRateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 100,
            tokens_per_day: 1_000_000,
            concurrent_requests: 10,
        }
    }
}

impl TenantRateLimitConfig {
    pub fn from_env() -> Self {
        let requests_per_min = std::env::var("RATE_LIMIT_TENANT_REQUESTS_PER_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100u32);

        let tokens_per_day = std::env::var("RATE_LIMIT_TENANT_TOKENS_PER_DAY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1_000_000u64);

        let concurrent = std::env::var("RATE_LIMIT_TENANT_CONCURRENT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10u32);

        Self {
            requests_per_minute: requests_per_min,
            tokens_per_day,
            concurrent_requests: concurrent,
        }
    }
}

/// Tenant-aware rate limiter using Redis for distributed rate limiting
pub struct TenantRateLimiter {
    #[cfg(feature = "redis")]
    redis_pool: Option<RedisPool>,
    #[cfg(feature = "redis")]
    config: TenantRateLimitConfig,
    fallback: RateLimiter, // Fallback to in-memory if Redis unavailable
}

impl TenantRateLimiter {
    #[cfg(feature = "redis")]
    pub fn new(config: TenantRateLimitConfig, redis_pool: Option<RedisPool>) -> Self {
        let fallback_config = RateLimitConfig {
            requests_per_window: config.requests_per_minute,
            window_duration: Duration::from_secs(60),
            burst_size: config.concurrent_requests,
        };

        Self {
            redis_pool,
            config,
            fallback: RateLimiter::new(fallback_config),
        }
    }

    #[cfg(not(feature = "redis"))]
    pub fn new(config: TenantRateLimitConfig) -> Self {
        let fallback_config = RateLimitConfig {
            requests_per_window: config.requests_per_minute,
            window_duration: Duration::from_secs(60),
            burst_size: config.concurrent_requests,
        };

        Self {
            fallback: RateLimiter::new(fallback_config),
        }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let enabled = std::env::var("RATE_LIMIT_PER_TENANT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        if !enabled {
            #[cfg(feature = "redis")]
            return Ok(Self::new(TenantRateLimitConfig::default(), None));
            #[cfg(not(feature = "redis"))]
            return Ok(Self::new(TenantRateLimitConfig::default()));
        }

        let config = TenantRateLimitConfig::from_env();
        #[cfg(feature = "redis")]
        let redis_pool = if let Ok(redis_url) = std::env::var("REDIS_URL") {
            Some(
                deadpool_redis::Config::from_url(redis_url)
                    .create_pool(Some(deadpool_redis::Runtime::Tokio1))
                    .map_err(|e| anyhow::anyhow!("Failed to create Redis pool: {}", e))?,
            )
        } else {
            None
        };

        #[cfg(feature = "redis")]
        {
            Ok(Self::new(config, redis_pool))
        }
        #[cfg(not(feature = "redis"))]
        {
            if std::env::var("REDIS_URL").is_ok() {
                tracing::warn!(
                    "REDIS_URL is set but Redis rate limiting support is disabled; using in-memory rate limits"
                );
            }
            Ok(Self::new(config))
        }
    }

    /// Check if tenant can make a request
    #[cfg_attr(
        not(feature = "redis"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    pub async fn check_request_limit(&self, tenant_id: &str) -> bool {
        #[cfg(feature = "redis")]
        if let Some(ref pool) = self.redis_pool {
            // Use Redis for distributed rate limiting
            if let Ok(mut conn) = pool.get().await {
                use deadpool_redis::redis::AsyncCommands;
                let key = format!("rate_limit:tenant:{}:requests", tenant_id);

                // Simple counter-based rate limiting with TTL
                let count: Result<u32, _> = conn.get(&key).await;
                let current_count = count.unwrap_or(0);

                if current_count >= self.config.requests_per_minute {
                    return false;
                }

                // Increment counter
                if let Err(e) = conn.incr::<_, _, ()>(&key, 1).await {
                    tracing::warn!(tenant_id, "Redis rate-limit incr failed: {e}");
                }
                // Set expiry to 60 seconds (sliding window)
                if let Err(e) = conn.expire::<_, ()>(&key, 60).await {
                    tracing::warn!(tenant_id, "Redis rate-limit expire failed: {e}");
                }

                return true;
            }
        }

        // Fallback to in-memory rate limiting
        self.fallback
            .check_rate_limit(&format!("tenant:{}", tenant_id))
    }

    /// Check if tenant has token quota remaining
    #[cfg_attr(
        not(feature = "redis"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    pub async fn check_token_limit(&self, tenant_id: &str, tokens: u64) -> bool {
        #[cfg(not(feature = "redis"))]
        let _ = (tenant_id, tokens);
        #[cfg(feature = "redis")]
        if let Some(ref pool) = self.redis_pool {
            use deadpool_redis::redis::AsyncCommands;
            if let Ok(mut conn) = pool.get().await {
                let key = format!("rate_limit:tenant:{}:tokens", tenant_id);
                let day_key = format!("rate_limit:tenant:{}:day", tenant_id);

                // Get current day token count
                let current_day = chrono::Utc::now().format("%Y-%m-%d").to_string();
                let stored_day: Result<Option<String>, _> = conn.get(&day_key).await;

                let count = if let Ok(Some(stored)) = stored_day {
                    if stored == current_day {
                        conn.get::<_, u64>(&key).await.unwrap_or(0)
                    } else {
                        // New day, reset counter
                        if let Err(e) = conn.set::<_, _, ()>(&day_key, &current_day).await {
                            tracing::warn!(tenant_id, "Redis token day-key set failed: {e}");
                        }
                        if let Err(e) = conn.set::<_, _, ()>(&key, 0u64).await {
                            tracing::warn!(tenant_id, "Redis token counter reset failed: {e}");
                        }
                        0
                    }
                } else {
                    // First request today
                    if let Err(e) = conn.set::<_, _, ()>(&day_key, &current_day).await {
                        tracing::warn!(tenant_id, "Redis token day-key init failed: {e}");
                    }
                    if let Err(e) = conn.set::<_, _, ()>(&key, 0u64).await {
                        tracing::warn!(tenant_id, "Redis token counter init failed: {e}");
                    }
                    0
                };

                if count + tokens > self.config.tokens_per_day {
                    return false;
                }

                // Increment token count
                if let Err(e) = conn.incr::<_, _, ()>(&key, tokens).await {
                    tracing::warn!(tenant_id, tokens, "Redis token-count incr failed: {e}");
                }
                return true;
            }
        }

        // Fallback: allow if no Redis
        true
    }

    /// Check concurrent request limit
    #[cfg_attr(
        not(feature = "redis"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    pub async fn check_concurrent_limit(&self, tenant_id: &str) -> bool {
        #[cfg(not(feature = "redis"))]
        let _ = tenant_id;
        #[cfg(feature = "redis")]
        if let Some(ref pool) = self.redis_pool {
            use deadpool_redis::redis::AsyncCommands;
            if let Ok(mut conn) = pool.get().await {
                let key = format!("rate_limit:tenant:{}:concurrent", tenant_id);
                let count: Result<u32, _> = conn.get(&key).await;

                if let Ok(count) = count
                    && count >= self.config.concurrent_requests
                {
                    return false;
                }

                // Increment concurrent count
                if let Err(e) = conn.incr::<_, _, ()>(&key, 1).await {
                    tracing::warn!(tenant_id, "Redis concurrent-limit incr failed: {e}");
                }
                if let Err(e) = conn.expire::<_, ()>(&key, 300).await {
                    tracing::warn!(tenant_id, "Redis concurrent-limit expire failed: {e}");
                }
                return true;
            }
        }

        // Fallback: allow if no Redis
        true
    }

    /// Decrement concurrent request count
    #[cfg_attr(
        not(feature = "redis"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    pub async fn release_concurrent(&self, tenant_id: &str) {
        #[cfg(not(feature = "redis"))]
        let _ = tenant_id;
        #[cfg(feature = "redis")]
        if let Some(ref pool) = self.redis_pool {
            use deadpool_redis::redis::AsyncCommands;
            if let Ok(mut conn) = pool.get().await {
                let key = format!("rate_limit:tenant:{}:concurrent", tenant_id);
                if let Err(e) = conn.decr::<_, _, ()>(&key, 1).await {
                    tracing::warn!(tenant_id, "Redis concurrent-count release failed: {e}");
                }
            }
        }
    }
}
