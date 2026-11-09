//! Convert dripstone [`Filter`] DSL to Milvus boolean expression strings.
//!
//! Chunk user metadata is stored under the JSON key `original_metadata` inside the Milvus
//! `metadata` column (same envelope as other dripstone HTTP backends).
//!
//! # Limitations
//!
//! - Field paths must be non-empty, ASCII alphanumeric / `_` / `.`, with no empty segments
//!   (so `a..b` is rejected).
//! - String bounds may not contain ASCII control characters other than tab/newline/CR
//!   (which are escaped). Use [`Filter`] semantics on the client for Unicode text otherwise.
//! - [`Filter::GreaterThan`] and related variants require a JSON **number** bound (unlike
//!   [`Filter::matches`], which has special cases for string comparisons).
//! - [`Filter::In`] with an empty value list is rejected here; [`Filter::NotIn`] with an
//!   empty list compiles to a tautology (`id != ""`) matching [`Filter::matches`].

use crate::errors::DripstoneError;
use crate::search::Filter;
use serde_json::Value;

/// Root path into the stored metadata JSON object for user-defined fields.
const METADATA_USER_PREFIX: &str = r#"metadata["original_metadata"]"#;

/// Convert a filter to a Milvus `expr` string for use in search/query APIs.
///
/// Returns an error for unsupported constructs rather than silently ignoring filters.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn filter_to_milvus(filter: &Filter) -> Result<String, DripstoneError> {
    convert_filter(filter)
}

fn is_safe_field(field: &str) -> bool {
    !field.is_empty()
        && field
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// Build `metadata["original_metadata"]["a"]["b"]` from dot-separated field path.
fn json_path_expr(field: &str) -> Result<String, DripstoneError> {
    if !is_safe_field(field) {
        return Err(DripstoneError::storage(format!(
            "milvus filter: unsupported field name (use ASCII alphanumeric, '_', '.', no empty segments): {field}"
        )));
    }
    let mut out = String::from(METADATA_USER_PREFIX);
    for part in field.split('.') {
        if part.is_empty() {
            return Err(DripstoneError::storage(
                "milvus filter: field path must not contain empty segments (e.g. avoid 'a..b')"
                    .to_string(),
            ));
        }
        out.reserve(part.len() + 5);
        out.push_str("[\"");
        out.push_str(part);
        out.push_str("\"]");
    }
    Ok(out)
}

fn value_literal(v: &Value) -> Result<String, DripstoneError> {
    match v {
        Value::Null => Ok("null".to_string()),
        Value::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),
        Value::Number(n) => Ok(n.to_string()),
        Value::String(s) => Ok(format!("\"{}\"", escape_milvus_string(s)?)),
        Value::Array(_) | Value::Object(_) => Err(DripstoneError::storage(
            "milvus filter: equality on JSON object/array is not supported in Milvus expr"
                .to_string(),
        )),
    }
}

fn escape_milvus_string(s: &str) -> Result<String, DripstoneError> {
    let extra = s
        .chars()
        .filter(|c| matches!(c, '\\' | '"' | '\n' | '\r' | '\t'))
        .count();
    let mut out = String::with_capacity(s.len().saturating_add(extra.saturating_mul(2)));
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                return Err(DripstoneError::storage(format!(
                    "milvus filter: string values must not contain control characters (found U+{:04X})",
                    c as u32
                )));
            }
            c => out.push(c),
        }
    }
    Ok(out)
}

fn numeric_bound(value: &Value) -> Result<String, DripstoneError> {
    if !value.is_number() {
        return Err(DripstoneError::storage(
            "milvus filter: numeric comparison requires a JSON number bound".to_string(),
        ));
    }
    value_literal(value)
}

fn convert_filter(filter: &Filter) -> Result<String, DripstoneError> {
    match filter {
        Filter::Equal { field, value } => compare_path(field, "==", value, false),
        Filter::NotEqual { field, value } => compare_path(field, "!=", value, false),
        Filter::In { field, values } => in_list(field, values, false),
        Filter::NotIn { field, values } => in_list(field, values, true),
        Filter::GreaterThan { field, value } => compare_path(field, ">", value, true),
        Filter::LessThan { field, value } => compare_path(field, "<", value, true),
        Filter::GreaterThanOrEqual { field, value } => compare_path(field, ">=", value, true),
        Filter::LessThanOrEqual { field, value } => compare_path(field, "<=", value, true),
        Filter::And { left, right } => {
            let l = convert_filter(left)?;
            let r = convert_filter(right)?;
            Ok(format!("({l} and {r})"))
        }
        Filter::Or { left, right } => {
            let l = convert_filter(left)?;
            let r = convert_filter(right)?;
            Ok(format!("({l} or {r})"))
        }
        Filter::Not { filter: inner } => {
            let i = convert_filter(inner)?;
            Ok(format!("not ({i})"))
        }
    }
}

fn compare_path(
    field: &str,
    op: &str,
    value: &Value,
    numeric: bool,
) -> Result<String, DripstoneError> {
    let path = json_path_expr(field)?;
    let lit = if numeric {
        numeric_bound(value)?
    } else {
        value_literal(value)?
    };
    Ok(format!("{path} {op} {lit}"))
}

fn in_list(field: &str, values: &[Value], negate: bool) -> Result<String, DripstoneError> {
    let path = json_path_expr(field)?;
    if values.is_empty() {
        if negate {
            return Ok(r#"id != """#.to_string());
        }
        return Err(DripstoneError::storage(
            "milvus filter: IN with empty value list is invalid".to_string(),
        ));
    }
    let mut parts = Vec::with_capacity(values.len());
    for v in values {
        let lit = value_literal(v)?;
        parts.push(format!("{path} == {lit}"));
    }
    let joined = parts.join(" or ");
    if negate {
        Ok(format!("not ({joined})"))
    } else {
        Ok(format!("({joined})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn equal_and_and() {
        let f = Filter::And {
            left: Box::new(Filter::Equal {
                field: "category".into(),
                value: json!("A"),
            }),
            right: Box::new(Filter::GreaterThan {
                field: "price".into(),
                value: json!(100),
            }),
        };
        let expr = filter_to_milvus(&f).expect("expr");
        assert_eq!(
            expr,
            r#"(metadata["original_metadata"]["category"] == "A" and metadata["original_metadata"]["price"] > 100)"#
        );
    }

    #[test]
    fn nested_field() {
        let f = Filter::Equal {
            field: "user.name".into(),
            value: json!("John"),
        };
        let expr = filter_to_milvus(&f).expect("expr");
        assert_eq!(
            expr,
            r#"metadata["original_metadata"]["user"]["name"] == "John""#
        );
    }

    #[test]
    fn not_in_empty_is_tautology() {
        let f = Filter::NotIn {
            field: "x".into(),
            values: vec![],
        };
        let expr = filter_to_milvus(&f).expect("expr");
        assert_eq!(expr, r#"id != """#);
    }

    #[test]
    fn rejects_object_bound() {
        let f = Filter::Equal {
            field: "k".into(),
            value: json!({ "a": 1 }),
        };
        assert!(filter_to_milvus(&f).is_err());
    }

    #[test]
    fn rejects_empty_field() {
        let f = Filter::Equal {
            field: String::new(),
            value: json!("x"),
        };
        assert!(filter_to_milvus(&f).is_err());
    }

    #[test]
    fn rejects_double_dot_path() {
        let f = Filter::Equal {
            field: "a..b".into(),
            value: json!(1),
        };
        assert!(filter_to_milvus(&f).is_err());
    }

    #[test]
    fn in_multiple_values_uses_or() {
        let f = Filter::In {
            field: "status".into(),
            values: vec![json!("a"), json!("b")],
        };
        let expr = filter_to_milvus(&f).expect("expr");
        assert_eq!(
            expr,
            r#"(metadata["original_metadata"]["status"] == "a" or metadata["original_metadata"]["status"] == "b")"#
        );
    }

    #[test]
    fn in_empty_list_errors() {
        let f = Filter::In {
            field: "x".into(),
            values: vec![],
        };
        assert!(filter_to_milvus(&f).is_err());
    }

    #[test]
    fn or_not_roundtrip() {
        let f = Filter::Or {
            left: Box::new(Filter::Equal {
                field: "a".into(),
                value: json!(1),
            }),
            right: Box::new(Filter::Not {
                filter: Box::new(Filter::Equal {
                    field: "b".into(),
                    value: json!(false),
                }),
            }),
        };
        let expr = filter_to_milvus(&f).expect("expr");
        assert_eq!(
            expr,
            r#"(metadata["original_metadata"]["a"] == 1 or not (metadata["original_metadata"]["b"] == false))"#
        );
    }

    #[test]
    fn string_escapes_quote() {
        let f = Filter::Equal {
            field: "msg".into(),
            value: json!("say \"hi\""),
        };
        let expr = filter_to_milvus(&f).expect("expr");
        assert_eq!(
            expr,
            r#"metadata["original_metadata"]["msg"] == "say \"hi\"""#
        );
    }
}
