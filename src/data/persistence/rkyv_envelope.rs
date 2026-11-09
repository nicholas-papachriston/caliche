//! Shared **magic + rkyv v1** wire layout for Redis values, LLM cache entries, and distributed
//! session affinity.
//!
//! # Reserved 4-byte magics (`CL` + domain + version)
//!
//! | Bytes (ASCII) | Consumer |
//! |---------------|----------|
//! | `CL\x01\x01` | LLM response cache (`CachedResponseStoredV1` in `llm::cache::codec`) |
//! | `CL\x02\x01` | Redis `SessionData` |
//! | `CL\x03\x01` | Redis `TrackedSession` (session registry) |
//! | `CL\x04\x01` | Distributed `SessionAffinity` |

/// LLM response cache: `CL` + version bytes + rkyv body.
pub const LLM_CACHE_MAGIC_V1: &[u8; 4] = b"CL\x01\x01";
/// Redis fast session blob (`SessionData`).
#[cfg(feature = "redis")]
pub const REDIS_SESSION_DATA_MAGIC_V1: &[u8; 4] = b"CL\x02\x01";
/// Redis session registry row (`TrackedSession`).
#[cfg(feature = "redis")]
pub const REDIS_TRACKED_SESSION_MAGIC_V1: &[u8; 4] = b"CL\x03\x01";
/// Distributed session affinity value.
#[cfg(feature = "redis")]
pub const SESSION_AFFINITY_MAGIC_V1: &[u8; 4] = b"CL\x04\x01";

/// Prefix `body` with `magic`.
pub fn prepend_magic(magic: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(magic.len() + body.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(body);
    out
}

/// Strip a fixed `magic` prefix or fail with `kind` in the error message.
pub fn strip_magic_prefix<'a>(
    magic: &[u8],
    bytes: &'a [u8],
    kind: &str,
) -> anyhow::Result<&'a [u8]> {
    let n = magic.len();
    if bytes.len() < n || &bytes[..n] != magic {
        anyhow::bail!("{kind} missing or invalid v1 magic (len={})", bytes.len());
    }
    Ok(&bytes[n..])
}
