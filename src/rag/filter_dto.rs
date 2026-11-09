//! Filter Data Transfer Object for deserializing complex filter queries from JSON.
//!
//! This module provides a DTO that mirrors `dripstone::search::Filter` but can be deserialized
//! from JSON input (e.g., from LLM tool calls).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Filter DTO that mirrors dripstone::search::Filter but is deserializable
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FilterDto {
    /// Exact equality match: `field == value`
    Equal { field: String, value: Value },
    /// Inequality match: `field != value`
    NotEqual { field: String, value: Value },
    /// Field value is in a set of values: `field IN [value1, value2, ...]`
    In { field: String, values: Vec<Value> },
    /// Field value is not in a set of values: `field NOT IN [value1, value2, ...]`
    NotIn { field: String, values: Vec<Value> },
    /// Greater than comparison: `field > value`
    GreaterThan { field: String, value: Value },
    /// Less than comparison: `field < value`
    LessThan { field: String, value: Value },
    /// Greater than or equal: `field >= value`
    GreaterThanOrEqual { field: String, value: Value },
    /// Less than or equal: `field <= value`
    LessThanOrEqual { field: String, value: Value },
    /// Logical AND: `left AND right`
    And {
        left: Box<FilterDto>,
        right: Box<FilterDto>,
    },
    /// Logical OR: `left OR right`
    Or {
        left: Box<FilterDto>,
        right: Box<FilterDto>,
    },
    /// Logical NOT: `NOT filter`
    Not { filter: Box<FilterDto> },
}

/// Convert FilterDto to Filter
fn filter_dto_to_filter(dto: FilterDto) -> Result<crate::data::storage::vector::Filter> {
    match dto {
        FilterDto::Equal { field, value } => {
            Ok(crate::data::storage::vector::Filter::Equal { field, value })
        }
        FilterDto::NotEqual { field, value } => {
            Ok(crate::data::storage::vector::Filter::NotEqual { field, value })
        }
        FilterDto::In { field, values } => {
            Ok(crate::data::storage::vector::Filter::In { field, values })
        }
        FilterDto::NotIn { field, values } => {
            Ok(crate::data::storage::vector::Filter::NotIn { field, values })
        }
        FilterDto::GreaterThan { field, value } => {
            Ok(crate::data::storage::vector::Filter::GreaterThan { field, value })
        }
        FilterDto::LessThan { field, value } => {
            Ok(crate::data::storage::vector::Filter::LessThan { field, value })
        }
        FilterDto::GreaterThanOrEqual { field, value } => {
            Ok(crate::data::storage::vector::Filter::GreaterThanOrEqual { field, value })
        }
        FilterDto::LessThanOrEqual { field, value } => {
            Ok(crate::data::storage::vector::Filter::LessThanOrEqual { field, value })
        }
        FilterDto::And { left, right } => Ok(crate::data::storage::vector::Filter::And {
            left: Box::new(filter_dto_to_filter(*left)?),
            right: Box::new(filter_dto_to_filter(*right)?),
        }),
        FilterDto::Or { left, right } => Ok(crate::data::storage::vector::Filter::Or {
            left: Box::new(filter_dto_to_filter(*left)?),
            right: Box::new(filter_dto_to_filter(*right)?),
        }),
        FilterDto::Not { filter } => Ok(crate::data::storage::vector::Filter::Not {
            filter: Box::new(filter_dto_to_filter(*filter)?),
        }),
    }
}

/// Try to convert a JSON Value to a Filter
pub fn filter_from_value(value: &Value) -> Result<crate::data::storage::vector::Filter> {
    // First try to deserialize as FilterDto
    let dto: FilterDto =
        serde_json::from_value(value.clone()).context("Failed to deserialize filter")?;
    filter_dto_to_filter(dto)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_filter_dto_equal() {
        let json = json!({
            "type": "equal",
            "field": "category",
            "value": "electronics"
        });
        let dto: FilterDto = serde_json::from_value(json).expect("Should deserialize FilterDto");
        let filter = filter_dto_to_filter(dto).expect("Should convert FilterDto to Filter");
        assert!(matches!(
            filter,
            crate::data::storage::vector::Filter::Equal { .. }
        ));
    }

    #[test]
    fn test_filter_dto_and() {
        let json = json!({
            "type": "and",
            "left": {
                "type": "equal",
                "field": "category",
                "value": "electronics"
            },
            "right": {
                "type": "greater_than",
                "field": "price",
                "value": 100
            }
        });
        let dto: FilterDto = serde_json::from_value(json).expect("Should deserialize FilterDto");
        let filter = filter_dto_to_filter(dto).expect("Should convert FilterDto to Filter");
        assert!(matches!(
            filter,
            crate::data::storage::vector::Filter::And { .. }
        ));
    }
}
