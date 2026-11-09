//! Convert dripstone Filter DSL to Qdrant filter JSON format.
//!
//! Qdrant stores chunk metadata at `metadata.original_metadata` in the payload.
//! Filter fields map to keys under that path (e.g. "category" -> "`metadata.original_metadata.category`").

use crate::search::Filter;
use serde_json::{Value, json};

const METADATA_PREFIX: &str = "metadata.original_metadata.";

/// Convert a dripstone Filter to Qdrant filter JSON.
///
/// Returns `Some(filter_value)` when the filter can be pushed down to Qdrant,
/// or `None` when the caller must fall back to client-side filtering via
/// [`crate::search::Filter::matches`]. In particular, empty `In([])` and
/// empty `NotIn([])` both return `None`; the two cases have different
/// client-side semantics (match-nothing vs. match-all), so callers MUST
/// apply the client-side fallback rather than dropping the filter silently.
#[must_use]
pub fn filter_to_qdrant(filter: &Filter) -> Option<Value> {
    convert_filter(filter)
}

fn qdrant_key(field: &str) -> String {
    format!("{METADATA_PREFIX}{field}")
}

fn value_to_qdrant_compatible(value: &Value) -> Option<Value> {
    match value {
        Value::String(s) => Some(Value::String(s.clone())),
        Value::Number(n) => Some(Value::Number(n.clone())),
        Value::Bool(b) => Some(Value::Bool(*b)),
        Value::Null => Some(Value::Null),
        Value::Array(arr) => {
            let converted: Vec<Value> = arr.iter().filter_map(value_to_qdrant_compatible).collect();
            if converted.len() == arr.len() {
                Some(Value::Array(converted))
            } else {
                None
            }
        }
        Value::Object(_) => None,
    }
}

fn convert_filter(filter: &Filter) -> Option<Value> {
    match filter {
        Filter::Equal { field, value } => match_value(field, value, true),
        Filter::NotEqual { field, value } => match_value(field, value, false),
        Filter::In { field, values } => match_any(field, values, true),
        Filter::NotIn { field, values } => match_any(field, values, false),
        Filter::GreaterThan { field, value } => range_condition_greater_than(field, value),
        Filter::LessThan { field, value } => range_condition_less_than(field, value),
        Filter::GreaterThanOrEqual { field, value } => range_condition_gte(field, value),
        Filter::LessThanOrEqual { field, value } => range_condition_lte(field, value),
        Filter::And { left, right } => {
            let left_v = convert_filter(left)?;
            let right_v = convert_filter(right)?;
            Some(json!({ "must": [left_v, right_v] }))
        }
        Filter::Or { left, right } => {
            let left_v = convert_filter(left)?;
            let right_v = convert_filter(right)?;
            Some(json!({
                "should": [left_v, right_v]
            }))
        }
        Filter::Not { filter: inner } => {
            let inner_v = convert_filter(inner)?;
            Some(json!({
                "must_not": [inner_v]
            }))
        }
    }
}

fn match_value(field: &str, value: &Value, include: bool) -> Option<Value> {
    let key = qdrant_key(field);
    let qv = value_to_qdrant_compatible(value)?;
    let clause = json!({
        "key": key,
        "match": { "value": qv }
    });
    if include {
        Some(json!({ "must": [clause] }))
    } else {
        Some(json!({ "must_not": [clause] }))
    }
}

fn match_any(field: &str, values: &[Value], include: bool) -> Option<Value> {
    if values.is_empty() {
        return None;
    }
    let any_values: Vec<Value> = values
        .iter()
        .filter_map(value_to_qdrant_compatible)
        .collect();
    if any_values.len() != values.len() {
        return None;
    }
    let key = qdrant_key(field);
    let clause = json!({
        "key": key,
        "match": { "any": any_values }
    });
    if include {
        Some(json!({ "must": [clause] }))
    } else {
        Some(json!({ "must_not": [clause] }))
    }
}

fn range_condition_greater_than(field: &str, value: &Value) -> Option<Value> {
    let key = qdrant_key(field);
    let f = value.as_number()?.as_f64()?;
    Some(json!({
        "must": [{
            "key": key,
            "range": { "gt": f }
        }]
    }))
}

fn range_condition_less_than(field: &str, value: &Value) -> Option<Value> {
    let key = qdrant_key(field);
    let f = value.as_number()?.as_f64()?;
    Some(json!({
        "must": [{
            "key": key,
            "range": { "lt": f }
        }]
    }))
}

fn range_condition_gte(field: &str, value: &Value) -> Option<Value> {
    let key = qdrant_key(field);
    let f = value.as_number()?.as_f64()?;
    Some(json!({
        "must": [{
            "key": key,
            "range": { "gte": f }
        }]
    }))
}

fn range_condition_lte(field: &str, value: &Value) -> Option<Value> {
    let key = qdrant_key(field);
    let f = value.as_number()?.as_f64()?;
    Some(json!({
        "must": [{
            "key": key,
            "range": { "lte": f }
        }]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_equal_to_qdrant() {
        let filter = Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        };
        let result = filter_to_qdrant(&filter).unwrap();
        assert_eq!(
            result["must"][0]["key"],
            "metadata.original_metadata.category"
        );
        assert_eq!(result["must"][0]["match"]["value"], "A");
    }

    #[test]
    fn test_in_to_qdrant() {
        let filter = Filter::In {
            field: "category".to_string(),
            values: vec![json!("A"), json!("B")],
        };
        let result = filter_to_qdrant(&filter).unwrap();
        assert!(result["must"][0]["match"]["any"].is_array());
        assert_eq!(
            result["must"][0]["match"]["any"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn test_and_to_qdrant() {
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
        let result = filter_to_qdrant(&filter).unwrap();
        let must = result["must"].as_array().unwrap();
        assert_eq!(must.len(), 2);
    }

    #[test]
    fn test_nested_field() {
        let filter = Filter::Equal {
            field: "user.name".to_string(),
            value: json!("John"),
        };
        let result = filter_to_qdrant(&filter).unwrap();
        assert_eq!(
            result["must"][0]["key"],
            "metadata.original_metadata.user.name"
        );
    }

    #[test]
    fn test_greater_than_to_qdrant() {
        let filter = Filter::GreaterThan {
            field: "price".to_string(),
            value: json!(100),
        };
        let result = filter_to_qdrant(&filter).unwrap();
        assert!(result["must"][0]["range"].get("gt").is_some());
    }

    #[test]
    fn test_not_to_qdrant() {
        let filter = Filter::Not {
            filter: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("A"),
            }),
        };
        let result = filter_to_qdrant(&filter).unwrap();
        assert!(result["must_not"].is_array());
        assert!(!result["must_not"].as_array().unwrap().is_empty());
    }
}
