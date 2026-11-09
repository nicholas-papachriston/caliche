//! Shared JWKS fetch/cache for JWT and Bot Framework verification.
//!
//! Re-exports the core implementation from `caliche::security` so API
//! middleware and webhook verification stay on one code path.

pub use caliche::security::JwksCache;

// Re-export helpers for any external callers that imported them from this module.
#[allow(unused_imports)]
pub use caliche::security::{jwk_to_decoding_key, jwk_x5c_to_decoding_key, pem_wrap_certificate};
