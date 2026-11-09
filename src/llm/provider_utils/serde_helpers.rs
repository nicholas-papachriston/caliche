//! Shared serde helpers for efficient string handling in provider wire types.

use serde::{Deserialize, Deserializer, Serializer};
use std::sync::Arc;

// Re-export Cow serde from beef_helpers so providers use a single implementation.
pub use crate::utils::beef_helpers::{
    deserialize_cow_str as deserialize_cow_static_str,
    serialize_cow_str as serialize_cow_static_str,
};

/// Serialize Arc<String> as a plain string.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn serialize_arc_str<S>(v: &Arc<String>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(v.as_str())
}

/// Deserialize into Arc<String>.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn deserialize_arc_str<'de, D>(d: D) -> Result<Arc<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Ok(Arc::new(s))
}

/// Serialize Option<Arc<String>> as optional string.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn serialize_option_arc_str<S>(v: &Option<Arc<String>>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match v {
        None => s.serialize_none(),
        Some(arc) => s.serialize_some(arc.as_str()),
    }
}

/// Deserialize into Option<Arc<String>>.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn deserialize_option_arc_str<'de, D>(d: D) -> Result<Option<Arc<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(d)?;
    Ok(opt.map(Arc::new))
}

// Re-export optional Cow serde from beef_helpers.
pub use crate::utils::beef_helpers::{
    deserialize_option_cow_str as deserialize_option_cow_static_str,
    serialize_option_cow_str as serialize_option_cow_static_str,
};
