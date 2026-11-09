//! NAPI bindings for durable event stream backends.
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]
//!
//! Exposes Memory and Redis backends so JavaScript/TypeScript can configure
//! durable streams for replay and resume (e.g. when using the API or a harness).

#[cfg(feature = "durable-streams-redis")]
use caliche::streams::RedisDurableStreamBackend;
use caliche::streams::{DurableStreamBackend, MemoryDurableStreamBackend};
use napi_derive::napi;
use std::sync::Arc;

/// Durable stream backend for event persistence and replay (e.g. when using the API).
///
/// Create via `DurableStreamBackend.memory(capacity)` or `DurableStreamBackend.redis(redisUrl)`,
/// then pass to `AgentBuilder.setDurableStreamBackend(backend)`.
#[napi(js_name = "DurableStreamBackend")]
pub struct DurableStreamBackendJs {
    pub(crate) inner: Arc<dyn DurableStreamBackend>,
}

#[napi]
impl DurableStreamBackendJs {
    /// Create an in-memory durable stream backend with a fixed capacity.
    /// No external dependencies. Events are stored in a ring buffer.
    #[napi(factory, catch_unwind)]
    pub fn memory(capacity: u32) -> Self {
        Self {
            inner: Arc::new(MemoryDurableStreamBackend::new(capacity as usize)),
        }
    }

    /// Create a Redis/Valkey Streams backend for durable event storage.
    ///
    /// Requires building with `--features durable-streams-redis` (or `full`).
    #[napi(factory, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn redis(redis_url: String) -> napi::Result<Self> {
        #[cfg(feature = "durable-streams-redis")]
        {
            let backend = RedisDurableStreamBackend::new(redis_url.as_str())
                .map_err(|e| napi::Error::from_reason(format!("{}", e)))?;
            Ok(Self {
                inner: Arc::new(backend),
            })
        }
        #[cfg(not(feature = "durable-streams-redis"))]
        {
            let _ = redis_url;
            Err(napi::Error::from_reason(
                "Redis durable streams require building with --features durable-streams-redis (or full)",
            ))
        }
    }

    /// Return a clone of the inner backend for use by the builder.
    pub(crate) fn get_inner(&self) -> Arc<dyn DurableStreamBackend> {
        Arc::clone(&self.inner)
    }
}
