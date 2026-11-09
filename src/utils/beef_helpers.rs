//! Serialization helpers for beef::lean::Cow types.
//!
//! Provides standardized serde functions for `beef::lean::Cow<'static, str>`
//! used throughout `messages/mod.rs` and other Cow-based structs.
//!
//! ## When to use which string type
//!
//! - **`Cow<'static, str>`**: Use for fields that are often literals or shared
//!   (roles, tool names, schema types, IDs). Clone is cheap; values can be
//!   borrowed from `'static` strings or owned when deserialized or built at
//!   runtime. Use with `serialize_cow_str` / `deserialize_cow_str` (or the
//!   option variants) in serde attributes.
//!
//! - **`String`**: Use when the value is always owned and not shared across
//!   many clones.
//!
//! - **`Arc<String>`**: Use when the same string is shared across many clones
//!   (e.g. long message content) and you want to avoid copying the buffer;
//!   beef's Cow does not cover this case.

use beef::lean::Cow;
use serde::{Deserialize, Deserializer, Serializer};

/// Serialize a beef::lean::Cow<'static, str> as a plain string.
///
/// # Example
/// ```text
/// #[derive(Serialize)]
/// struct MyType {
///     #[serde(serialize_with = "serialize_cow_str")]
///     name: Cow<'static, str>,
/// }
/// ```
pub fn serialize_cow_str<S>(v: &Cow<'static, str>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(v.as_ref())
}

/// Deserialize a string into a beef::lean::Cow<'static, str>.
///
/// # Example
/// ```text
/// #[derive(Deserialize)]
/// struct MyType {
///     #[serde(deserialize_with = "deserialize_cow_str")]
///     name: Cow<'static, str>,
/// }
/// ```
pub fn deserialize_cow_str<'de, D>(d: D) -> Result<Cow<'static, str>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Ok(Cow::owned(s))
}

/// Serialize an optional beef::lean::Cow<'static, str>.
pub fn serialize_option_cow_str<S>(v: &Option<Cow<'static, str>>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match v {
        None => s.serialize_none(),
        Some(cow) => s.serialize_str(cow.as_ref()),
    }
}

/// Deserialize an optional string into a beef::lean::Cow<'static, str>.
pub fn deserialize_option_cow_str<'de, D>(d: D) -> Result<Option<Cow<'static, str>>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(d)?;
    Ok(opt.map(Cow::owned))
}

/// Serialize a slice of beef::lean::Cow<'static, str> as a JSON array of strings.
pub fn serialize_vec_cow_str<S>(v: &[Cow<'static, str>], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for cow in v {
        seq.serialize_element(cow.as_ref())?;
    }
    seq.end()
}

/// Deserialize a JSON array of strings into Vec<Cow<'static, str>>.
pub fn deserialize_vec_cow_str<'de, D>(d: D) -> Result<Vec<Cow<'static, str>>, D::Error>
where
    D: Deserializer<'de>,
{
    let vec = Vec::<String>::deserialize(d)?;
    Ok(vec.into_iter().map(Cow::owned).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[test]
    fn test_serialization_roundtrip() {
        #[derive(Serialize, Deserialize)]
        struct TestType {
            #[serde(
                serialize_with = "serialize_cow_str",
                deserialize_with = "deserialize_cow_str"
            )]
            name: Cow<'static, str>,
        }

        let original = TestType {
            name: Cow::owned("test".to_string()),
        };

        let json = serde_json::to_value(&original).expect("serialize");
        assert_eq!(json["name"], "test");

        let deserialized: TestType = serde_json::from_value(json).expect("deserialize");
        assert_eq!(deserialized.name.as_ref(), "test");
    }

    #[test]
    fn test_option_serialization_roundtrip() {
        #[derive(Serialize, Deserialize)]
        struct TestType {
            #[serde(
                serialize_with = "serialize_option_cow_str",
                deserialize_with = "deserialize_option_cow_str"
            )]
            description: Option<Cow<'static, str>>,
        }

        let with_value = TestType {
            description: Some(Cow::owned("test".to_string())),
        };

        let json = serde_json::to_value(&with_value).expect("serialize");
        assert_eq!(json["description"], "test");

        let deserialized: TestType = serde_json::from_value(json).expect("deserialize");
        assert_eq!(
            deserialized.description.as_ref().map(|s| s.as_ref()),
            Some("test")
        );

        let none_case = TestType { description: None };
        let json = serde_json::to_value(&none_case).expect("serialize none");
        assert_eq!(json["description"], serde_json::Value::Null);
    }
}
