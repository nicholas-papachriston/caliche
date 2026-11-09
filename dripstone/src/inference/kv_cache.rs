//! KV cache support for incremental decoding.
//!
//! Backends that support a key-value cache can avoid re-feeding the full sequence
//! each step by following the prefill-then-decode protocol below. Cache state is
//! backend-specific (e.g. ORT holds ort tensors; Candle uses its own `Cache`);
//! this module only defines the capability trait and the protocol.

/// Protocol for backends that support incremental decoding with a KV cache:
///
/// - **Prefill**: First step runs with the full prompt (no past KV). The backend
///   may capture present key/value outputs and store them as the initial cache.
/// - **Decode**: Subsequent steps run with a single new token plus past KV as
///   input; present KV outputs update the cache for the next step.
///
/// Backends implement this with their own cache representation; no shared tensor
/// type is required.
pub trait KvCacheSupport: Send + Sync {
    /// Returns whether this backend supports incremental decode with a KV cache.
    /// When true, the backend uses prefill then decode (single-token steps with
    /// past/present KV) when the model supports it; when false, generation may
    /// re-feed the full sequence each step.
    fn supports_kv_cache(&self) -> bool;
}
