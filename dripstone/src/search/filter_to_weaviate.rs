//! Convert dripstone [`Filter`] DSL to Weaviate GraphQL `where` clause fragments.
//!
//! User metadata is logically stored under `metadata.original_metadata` in the dripstone
//! envelope (same as Milvus/Qdrant translators).
//!
//! # API gaps (current dripstone Weaviate backend)
//!
//! - The Weaviate class schema stores `metadata` as **`text`** (a JSON string), not as an
//!   `object`. Weaviate cannot apply XPath-style `path` filters into nested fields inside
//!   `text` properties, so [`filter_to_weaviate`] currently always returns `None` and the
//!   backend must fall back to client-side [`Filter::matches`].
//! - When `metadata` is migrated to `object` with nested `original_metadata`, enable
//!   [`METADATA_OBJECT_FILTER_PUSHDOWN`] and the translator will emit GraphQL such as:
//!   `path: ["metadata", "original_metadata", "category"], operator: Equal, valueText: "A"`.
//! - [`Filter::In`] / [`Filter::NotIn`] compile to `Or` / `Not`+`Or` of leaf predicates;
//!   empty `In` returns `None` (client-side match-nothing semantics).
//! - Empty `NotIn` is not pushed down (`None`; client-side match-all).
//! - Range comparisons require numeric JSON bounds (like Milvus).

use crate::search::Filter;
use serde_json::Value;

/// When true, [`filter_to_weaviate`] delegates to [`convert_filter_impl`].
/// Set to true once the Weaviate schema stores `metadata` as `object`.
pub const METADATA_OBJECT_FILTER_PUSHDOWN: bool = false;

/// GraphQL `where` inner clause (contents inside `where: { ... }`), or `None` when push-down
/// is not possible and callers should use [`Filter::matches`] client-side.
#[must_use]
pub fn filter_to_weaviate(filter: &Filter) -> Option<String> {
    if METADATA_OBJECT_FILTER_PUSHDOWN {
        convert_filter_impl(filter)
    } else {
        let _ = filter;
        None
    }
}

fn is_safe_field(field: &str) -> bool {
    !field.is_empty()
        && field
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        && !field.split('.').any(str::is_empty)
}

fn format_path(field: &str) -> Option<String> {
    if !is_safe_field(field) {
        return None;
    }
    let mut parts = vec![
        "\"metadata\"".to_string(),
        "\"original_metadata\"".to_string(),
    ];
    for segment in field.split('.') {
        parts.push(format!("\"{}\"", escape_graphql_string(segment)));
    }
    Some(format!("[{}]", parts.join(", ")))
}

fn escape_graphql_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn value_binding(value: &Value) -> Option<(&'static str, String)> {
    match value {
        Value::Null | Value::Array(_) | Value::Object(_) => None,
        Value::Bool(b) => Some(("valueBoolean", b.to_string())),
        Value::Number(n) => Some(("valueNumber", n.to_string())),
        Value::String(s) => Some(("valueText", format!("\"{}\"", escape_graphql_string(s)))),
    }
}

fn leaf_predicate(path: &str, operator: &str, value: &Value) -> Option<String> {
    let (value_key, value_lit) = value_binding(value)?;
    Some(format!(
        "path: {path}, operator: {operator}, {value_key}: {value_lit}"
    ))
}

fn convert_filter_impl(filter: &Filter) -> Option<String> {
    match filter {
        Filter::Equal { field, value } => comparison_leaf(field, "Equal", value, false),
        Filter::NotEqual { field, value } => comparison_leaf(field, "NotEqual", value, false),
        Filter::In { field, values } => in_operands(field, values, false),
        Filter::NotIn { field, values } => in_operands(field, values, true),
        Filter::GreaterThan { field, value } => comparison_leaf(field, "GreaterThan", value, true),
        Filter::LessThan { field, value } => comparison_leaf(field, "LessThan", value, true),
        Filter::GreaterThanOrEqual { field, value } => {
            comparison_leaf(field, "GreaterThanEqual", value, true)
        }
        Filter::LessThanOrEqual { field, value } => {
            comparison_leaf(field, "LessThanEqual", value, true)
        }
        Filter::And { left, right } => binary_operands("And", left, right),
        Filter::Or { left, right } => binary_operands("Or", left, right),
        Filter::Not { filter: inner } => {
            let i = convert_filter_impl(inner)?;
            Some(format!("operator: Not, operands: [{{ {i} }}]"))
        }
    }
}

fn comparison_leaf(
    field: &str,
    operator: &str,
    value: &Value,
    require_number: bool,
) -> Option<String> {
    if require_number && !value.is_number() {
        return None;
    }
    let path = format_path(field)?;
    leaf_predicate(&path, operator, value)
}

fn or_join_equals(path: &str, values: &[Value]) -> Option<String> {
    let mut operands = Vec::with_capacity(values.len());
    for v in values {
        operands.push(leaf_predicate(path, "Equal", v)?);
    }
    if operands.len() == 1 {
        return operands.pop();
    }
    Some(format!(
        "operator: Or, operands: [{{ {} }}]",
        operands
            .iter()
            .map(|op| format!("{{ {op} }}"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

fn in_operands(field: &str, values: &[Value], negate: bool) -> Option<String> {
    if values.is_empty() {
        return None;
    }
    let path = format_path(field)?;
    let inner = or_join_equals(&path, values)?;
    if negate {
        Some(format!("operator: Not, operands: [{{ {inner} }}]"))
    } else {
        Some(inner)
    }
}

fn binary_operands(operator: &str, left: &Filter, right: &Filter) -> Option<String> {
    let l = convert_filter_impl(left)?;
    let r = convert_filter_impl(right)?;
    Some(format!(
        "operator: {operator}, operands: [{{ {l} }}, {{ {r} }}]"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn impl_for_tests(filter: &Filter) -> Option<String> {
        convert_filter_impl(filter)
    }

    #[test]
    fn equal_graphql_object_metadata() {
        let filter = Filter::Equal {
            field: "category".into(),
            value: json!("A"),
        };
        let clause = impl_for_tests(&filter).expect("clause");
        assert!(clause.contains(r#"path: ["metadata", "original_metadata", "category"]"#));
        assert!(clause.contains("operator: Equal"));
        assert!(clause.contains(r#"valueText: "A""#));
    }

    #[test]
    fn nested_field_path() {
        let filter = Filter::Equal {
            field: "user.name".into(),
            value: json!("John"),
        };
        let clause = impl_for_tests(&filter).expect("clause");
        assert!(clause.contains(r#""user""#));
        assert!(clause.contains(r#""name""#));
    }

    #[test]
    fn and_graphql() {
        let filter = Filter::And {
            left: Box::new(Filter::Equal {
                field: "a".into(),
                value: json!(1),
            }),
            right: Box::new(Filter::GreaterThan {
                field: "b".into(),
                value: json!(2),
            }),
        };
        let clause = impl_for_tests(&filter).expect("clause");
        assert!(clause.starts_with("operator: And"));
    }

    #[test]
    fn in_empty_returns_none() {
        let filter = Filter::In {
            field: "x".into(),
            values: vec![],
        };
        assert!(impl_for_tests(&filter).is_none());
    }

    #[test]
    fn public_api_none_until_object_schema() {
        let filter = Filter::Equal {
            field: "category".into(),
            value: json!("A"),
        };
        assert!(filter_to_weaviate(&filter).is_none());
    }

    #[test]
    fn rejects_unsafe_field() {
        let filter = Filter::Equal {
            field: "a..b".into(),
            value: json!(1),
        };
        assert!(impl_for_tests(&filter).is_none());
    }
}
