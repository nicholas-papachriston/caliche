//! Convert dripstone Filter DSL to `PostgreSQL` JSONB WHERE clause fragments.
//!
//! `PostgreSQL` stores chunk metadata in the `metadata` JSONB column.
//! Filters are converted to parameterized WHERE conditions to avoid injection.

use crate::search::Filter;
use serde_json::Value;

/// Validate that a field name is safe for use in SQL (no injection).
///
/// Rejects empty strings and fields with leading/trailing dots or
/// empty path components (e.g. `foo..bar`, `.foo`, `foo.`).
fn is_safe_field(field: &str) -> bool {
    if field.is_empty() {
        return false;
    }
    if !field
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return false;
    }
    !field.split('.').any(str::is_empty)
}

/// Result of converting a filter to SQL: the WHERE fragment and bound values.
/// Placeholders in the fragment use 1-based indices starting from `param_start`.
#[derive(Debug)]
pub struct FilterSql {
    /// WHERE clause fragment, e.g. " AND metadata @> $4"
    pub clause: String,
    /// Values to bind to the placeholders (in order)
    pub params: Vec<Value>,
}

/// Convert a dripstone Filter to a `PostgreSQL` WHERE clause and params.
/// Returns `Some(FilterSql)` when the filter can be pushed down, or `None` to
/// fall back to client-side filtering.
#[must_use]
pub fn filter_to_sql(filter: &Filter, param_start: usize) -> Option<FilterSql> {
    let mut params = Vec::new();
    let clause = convert_filter(filter, param_start, &mut params)?;
    Some(FilterSql { clause, params })
}

fn next_param(idx: &mut usize) -> String {
    let s = (*idx).to_string();
    *idx += 1;
    format!("${s}")
}

fn convert_filter(filter: &Filter, param_start: usize, params: &mut Vec<Value>) -> Option<String> {
    let mut idx = param_start;
    convert_filter_inner(filter, &mut idx, params)
}

fn convert_filter_inner(
    filter: &Filter,
    idx: &mut usize,
    params: &mut Vec<Value>,
) -> Option<String> {
    match filter {
        Filter::Equal { field, value } => containment(field, value, idx, params, false),
        Filter::NotEqual { field, value } => containment(field, value, idx, params, true),
        Filter::In { field, values } => containment_any_or_none(field, values, idx, params, false),
        Filter::NotIn { field, values } => {
            containment_any_or_none(field, values, idx, params, true)
        }
        Filter::GreaterThan { field, value } => numeric_comparison(field, value, idx, params, ">"),
        Filter::LessThan { field, value } => numeric_comparison(field, value, idx, params, "<"),
        Filter::GreaterThanOrEqual { field, value } => {
            numeric_comparison(field, value, idx, params, ">=")
        }
        Filter::LessThanOrEqual { field, value } => {
            numeric_comparison(field, value, idx, params, "<=")
        }
        Filter::And { left, right } => {
            let l = convert_filter_inner(left, idx, params)?;
            let r = convert_filter_inner(right, idx, params)?;
            Some(format!("{l}{r}"))
        }
        Filter::Or { left, right } => {
            let l = convert_filter_inner(left, idx, params)?;
            let r = convert_filter_inner(right, idx, params)?;
            let l_trimmed = l.strip_prefix(" AND ").unwrap_or(&l);
            let r_trimmed = r.strip_prefix(" AND ").unwrap_or(&r);
            Some(format!(" AND ({l_trimmed} OR {r_trimmed})"))
        }
        Filter::Not { filter: inner } => {
            let i = convert_filter_inner(inner, idx, params)?;
            let i_trimmed = i.strip_prefix(" AND ").unwrap_or(&i);
            Some(format!(" AND NOT ({i_trimmed})"))
        }
    }
}

/// `metadata @> {field: value}`, or its negation, as one AND clause.
fn containment(
    field: &str,
    value: &Value,
    idx: &mut usize,
    params: &mut Vec<Value>,
    negate: bool,
) -> Option<String> {
    if !is_safe_field(field) {
        return None;
    }
    let clause = containment_param(field, value, idx, params, negate);
    Some(format!(" AND {clause}"))
}

/// Any of `values` (`In`), or none of them (`NotIn` with `negate`), as one AND clause.
fn containment_any_or_none(
    field: &str,
    values: &[Value],
    idx: &mut usize,
    params: &mut Vec<Value>,
    negate: bool,
) -> Option<String> {
    if !is_safe_field(field) || values.is_empty() {
        return None;
    }
    let clauses: Vec<String> = values
        .iter()
        .map(|value| containment_param(field, value, idx, params, negate))
        .collect();
    let joiner = if negate { " AND " } else { " OR " };
    Some(format!(" AND ({})", clauses.join(joiner)))
}

/// Push `{field: value}` as the next parameter and return its containment test.
fn containment_param(
    field: &str,
    value: &Value,
    idx: &mut usize,
    params: &mut Vec<Value>,
    negate: bool,
) -> String {
    let ph = next_param(idx);
    params.push(json_object_for_field(field, value.clone()));
    if negate {
        format!("NOT (metadata @> {ph}::jsonb)")
    } else {
        format!("metadata @> {ph}::jsonb")
    }
}

fn json_object_for_field(field: &str, value: Value) -> Value {
    if field.contains('.') {
        let parts: Vec<&str> = field.split('.').collect();
        build_nested_json(parts.as_slice(), value)
    } else {
        serde_json::json!({ field: value })
    }
}

fn build_nested_json(parts: &[&str], value: Value) -> Value {
    if parts.is_empty() {
        value
    } else {
        let first = parts[0];
        let rest = &parts[1..];
        serde_json::json!({ first: build_nested_json(rest, value) })
    }
}

fn numeric_comparison(
    field: &str,
    value: &Value,
    idx: &mut usize,
    params: &mut Vec<Value>,
    op: &str,
) -> Option<String> {
    if !is_safe_field(field) {
        return None;
    }
    let f = value.as_number()?.as_f64()?;
    let ph = next_param(idx);
    params.push(serde_json::json!(f));
    let path = metadata_path_expr(field);
    Some(format!(" AND ({path}) {op} {ph}::numeric"))
}

fn metadata_path_expr(field: &str) -> String {
    if field.contains('.') {
        let path = field.split('.').collect::<Vec<_>>().join(",");
        format!("(metadata #> '{{{path}}}')::text::numeric")
    } else {
        format!("(metadata ->> '{field}')::numeric")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_equal_to_sql() {
        let filter = Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        };
        let result = filter_to_sql(&filter, 4).unwrap();
        assert!(result.clause.contains("metadata @>"));
        assert_eq!(result.params.len(), 1);
        assert_eq!(result.params[0], json!({"category": "A"}));
    }

    #[test]
    fn test_and_to_sql() {
        let filter = Filter::And {
            left: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("A"),
            }),
            right: Box::new(Filter::Equal {
                field: "price".to_string(),
                value: json!(100),
            }),
        };
        let result = filter_to_sql(&filter, 4).unwrap();
        assert!(result.clause.contains("metadata @>"));
        assert_eq!(result.params.len(), 2);
    }

    #[test]
    fn test_nested_field() {
        let filter = Filter::Equal {
            field: "user.name".to_string(),
            value: json!("John"),
        };
        let result = filter_to_sql(&filter, 4).unwrap();
        assert_eq!(result.params[0], json!({"user": {"name": "John"}}));
    }

    #[test]
    fn test_greater_than() {
        let filter = Filter::GreaterThan {
            field: "price".to_string(),
            value: json!(100),
        };
        let result = filter_to_sql(&filter, 4).unwrap();
        assert!(result.clause.contains('>'));
        assert_eq!(result.params[0].as_number().unwrap().as_f64(), Some(100.0));
    }

    #[test]
    fn test_or_to_sql() {
        let filter = Filter::Or {
            left: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("A"),
            }),
            right: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("B"),
            }),
        };
        let result = filter_to_sql(&filter, 4).unwrap();
        assert!(result.clause.contains("OR"));
        assert_eq!(result.params.len(), 2);
    }

    #[test]
    fn containment_clauses_are_exact() {
        let not_equal = Filter::NotEqual {
            field: "category".to_string(),
            value: json!("A"),
        };
        let result = filter_to_sql(&not_equal, 3).expect("not equal");
        assert_eq!(result.clause, " AND NOT (metadata @> $3::jsonb)");
        assert_eq!(result.params, vec![json!({"category": "A"})]);

        let any = Filter::In {
            field: "tag".to_string(),
            values: vec![json!("x"), json!("y")],
        };
        let result = filter_to_sql(&any, 1).expect("in");
        assert_eq!(
            result.clause,
            " AND (metadata @> $1::jsonb OR metadata @> $2::jsonb)"
        );
        assert_eq!(
            result.params,
            vec![json!({"tag": "x"}), json!({"tag": "y"})]
        );

        let none = Filter::NotIn {
            field: "tag".to_string(),
            values: vec![json!("x"), json!("y")],
        };
        let result = filter_to_sql(&none, 5).expect("not in");
        assert_eq!(
            result.clause,
            " AND (NOT (metadata @> $5::jsonb) AND NOT (metadata @> $6::jsonb))"
        );
        assert_eq!(
            result.params,
            vec![json!({"tag": "x"}), json!({"tag": "y"})]
        );

        let equal_then_in = Filter::And {
            left: Box::new(Filter::Equal {
                field: "a".to_string(),
                value: json!(1),
            }),
            right: Box::new(Filter::In {
                field: "b".to_string(),
                values: vec![json!(2)],
            }),
        };
        let result = filter_to_sql(&equal_then_in, 1).expect("and");
        assert_eq!(
            result.clause,
            " AND metadata @> $1::jsonb AND (metadata @> $2::jsonb)"
        );
    }

    #[test]
    fn empty_or_unsafe_sets_are_rejected() {
        for filter in [
            Filter::In {
                field: "tag".to_string(),
                values: vec![],
            },
            Filter::NotIn {
                field: "tag".to_string(),
                values: vec![],
            },
            Filter::NotEqual {
                field: String::new(),
                value: json!(1),
            },
            Filter::In {
                field: String::new(),
                values: vec![json!(1)],
            },
        ] {
            assert!(filter_to_sql(&filter, 1).is_none(), "{filter:?}");
        }
    }

    #[test]
    fn test_empty_field_rejected() {
        let filter = Filter::Equal {
            field: String::new(),
            value: json!("x"),
        };
        assert!(filter_to_sql(&filter, 1).is_none());

        let filter = Filter::Equal {
            field: "foo..bar".to_string(),
            value: json!("x"),
        };
        assert!(filter_to_sql(&filter, 1).is_none());

        let filter = Filter::Equal {
            field: ".foo".to_string(),
            value: json!("x"),
        };
        assert!(filter_to_sql(&filter, 1).is_none());

        let filter = Filter::Equal {
            field: "foo;--".to_string(),
            value: json!("x"),
        };
        assert!(filter_to_sql(&filter, 1).is_none());
    }

    #[test]
    fn test_not_to_sql() {
        let filter = Filter::Not {
            filter: Box::new(Filter::Equal {
                field: "active".to_string(),
                value: json!(false),
            }),
        };
        let result = filter_to_sql(&filter, 4).unwrap();
        assert!(result.clause.contains("NOT"));
        assert_eq!(result.params.len(), 1);
    }
}
