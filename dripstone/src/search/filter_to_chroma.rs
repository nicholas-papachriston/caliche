//! Convert dripstone [`Filter`] DSL to Chroma `where` JSON.
//!
//! User metadata fields map directly to keys in the Chroma metadata filter object
//! (callers store `original_metadata` separately when encoding records).

use crate::errors::DripstoneError;
use crate::search::Filter;
use serde_json::{Value, json};

/// Convert a filter to Chroma's `where` clause JSON.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn filter_to_chroma(filter: &Filter) -> Result<Value, DripstoneError> {
    convert_filter(filter)
}

fn convert_filter(filter: &Filter) -> Result<Value, DripstoneError> {
    match filter {
        Filter::Equal { field, value } => Ok(json!({ field: { "$eq": value } })),
        Filter::NotEqual { field, value } => Ok(json!({ field: { "$ne": value } })),
        Filter::In { field, values } => Ok(json!({ field: { "$in": values } })),
        Filter::NotIn { field, values } => Ok(json!({ field: { "$nin": values } })),
        Filter::GreaterThan { field, value } => Ok(json!({ field: { "$gt": value } })),
        Filter::LessThan { field, value } => Ok(json!({ field: { "$lt": value } })),
        Filter::GreaterThanOrEqual { field, value } => Ok(json!({ field: { "$gte": value } })),
        Filter::LessThanOrEqual { field, value } => Ok(json!({ field: { "$lte": value } })),
        Filter::And { left, right } => Ok(json!({
            "$and": [convert_filter(left)?, convert_filter(right)?]
        })),
        Filter::Or { left, right } => Ok(json!({
            "$or": [convert_filter(left)?, convert_filter(right)?]
        })),
        Filter::Not { filter } => Ok(json!({ "$not": convert_filter(filter)? })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn equal_to_chroma() {
        let filter = Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        };
        let result = filter_to_chroma(&filter).expect("chroma where");
        assert_eq!(result["category"]["$eq"], "A");
    }

    #[test]
    fn in_to_chroma() {
        let filter = Filter::In {
            field: "category".to_string(),
            values: vec![json!("A"), json!("B")],
        };
        let result = filter_to_chroma(&filter).expect("chroma where");
        assert!(result["category"]["$in"].is_array());
        assert_eq!(result["category"]["$in"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn and_to_chroma() {
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
        let result = filter_to_chroma(&filter).expect("chroma where");
        let and = result["$and"].as_array().expect("$and array");
        assert_eq!(and.len(), 2);
    }

    #[test]
    fn not_to_chroma() {
        let filter = Filter::Not {
            filter: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("A"),
            }),
        };
        let result = filter_to_chroma(&filter).expect("chroma where");
        assert_eq!(result["$not"]["category"]["$eq"], "A");
    }
}
