//! Advanced metadata filtering DSL for complex query construction.
//!
//! Provides a flexible, builder-based filtering system for complex metadata queries
//! supporting logical operators (AND, OR, NOT) and comparison operators.

use serde_json::Value;
use std::cmp::Ordering;
use std::fmt;

/// A filter expression for matching metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
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
    And { left: Box<Self>, right: Box<Self> },
    /// Logical OR: `left OR right`
    Or { left: Box<Self>, right: Box<Self> },
    /// Logical NOT: `NOT filter`
    Not { filter: Box<Self> },
}

impl Filter {
    /// Check if this filter matches the given metadata.
    #[must_use]
    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Self::Equal { field, value } => {
                Self::get_field_value(metadata, field).is_some_and(|v| v == value)
            }
            Self::NotEqual { field, value } => {
                Self::get_field_value(metadata, field) != Some(value)
            }
            Self::In { field, values } => {
                Self::get_field_value(metadata, field).is_some_and(|v| values.contains(v))
            }
            Self::NotIn { field, values } => {
                Self::get_field_value(metadata, field).is_none_or(|v| !values.contains(v))
            }
            Self::GreaterThan { field, value } => {
                Self::compare_value(metadata, field, value, |ord| ord == Ordering::Greater)
            }
            Self::LessThan { field, value } => {
                Self::compare_value(metadata, field, value, |ord| ord == Ordering::Less)
            }
            Self::GreaterThanOrEqual { field, value } => {
                Self::compare_value(metadata, field, value, |ord| {
                    matches!(ord, Ordering::Greater | Ordering::Equal)
                })
            }
            Self::LessThanOrEqual { field, value } => {
                Self::compare_value(metadata, field, value, |ord| {
                    matches!(ord, Ordering::Less | Ordering::Equal)
                })
            }
            Self::And { left, right } => left.matches(metadata) && right.matches(metadata),
            Self::Or { left, right } => left.matches(metadata) || right.matches(metadata),
            Self::Not { filter } => !filter.matches(metadata),
        }
    }

    /// Get a field value from metadata, supporting nested paths with dot notation.
    fn get_field_value<'a>(metadata: &'a Value, field: &str) -> Option<&'a Value> {
        if field.contains('.') {
            // Handle nested paths like "user.name"
            let parts: Vec<&str> = field.split('.').collect();
            let mut current = metadata;
            for part in parts {
                current = current.get(part)?;
            }
            Some(current)
        } else {
            metadata.get(field)
        }
    }

    /// Compare two values from metadata using the provided ordering
    /// predicate. Numbers use `f64` comparison, strings use true
    /// lexicographic `Ord`, booleans use `bool` ordering. Mismatched or
    /// non-comparable types (`Null`, `Array`, `Object`) yield `false`.
    fn compare_value<F>(metadata: &Value, field: &str, value: &Value, pred: F) -> bool
    where
        F: Fn(Ordering) -> bool,
    {
        let Some(field_value) = Self::get_field_value(metadata, field) else {
            return false;
        };

        match (field_value, value) {
            (Value::Number(a), Value::Number(b)) => {
                let Some(a_f) = a.as_f64() else { return false };
                let Some(b_f) = b.as_f64() else { return false };
                let Some(ord) = a_f.partial_cmp(&b_f) else {
                    return false;
                };
                pred(ord)
            }
            (Value::String(a), Value::String(b)) => pred(a.as_str().cmp(b.as_str())),
            (Value::Bool(a), Value::Bool(b)) => pred(a.cmp(b)),
            _ => false,
        }
    }
}

/// Builder for constructing complex filter expressions programmatically.
#[derive(Debug, Default)]
pub struct FilterBuilder {
    filter: Option<Filter>,
}

impl FilterBuilder {
    /// Create a new filter builder.
    #[must_use]
    pub const fn new() -> Self {
        Self { filter: None }
    }

    /// Add an equality filter: `field == value`
    #[must_use]
    pub fn equal(self, field: impl Into<String>, value: Value) -> Self {
        self.add(Filter::Equal {
            field: field.into(),
            value,
        })
    }

    /// Add a not-equal filter: `field != value`
    #[must_use]
    pub fn not_equal(self, field: impl Into<String>, value: Value) -> Self {
        self.add(Filter::NotEqual {
            field: field.into(),
            value,
        })
    }

    /// Add an IN filter: `field IN [values...]`
    #[must_use]
    pub fn in_values(self, field: impl Into<String>, values: Vec<Value>) -> Self {
        self.add(Filter::In {
            field: field.into(),
            values,
        })
    }

    /// Add a NOT IN filter: `field NOT IN [values...]`
    #[must_use]
    pub fn not_in(self, field: impl Into<String>, values: Vec<Value>) -> Self {
        self.add(Filter::NotIn {
            field: field.into(),
            values,
        })
    }

    /// Add a greater-than filter: `field > value`
    #[must_use]
    pub fn greater_than(self, field: impl Into<String>, value: Value) -> Self {
        self.add(Filter::GreaterThan {
            field: field.into(),
            value,
        })
    }

    /// Add a less-than filter: `field < value`
    #[must_use]
    pub fn less_than(self, field: impl Into<String>, value: Value) -> Self {
        self.add(Filter::LessThan {
            field: field.into(),
            value,
        })
    }

    /// Add a greater-than-or-equal filter: `field >= value`
    #[must_use]
    pub fn greater_than_or_equal(self, field: impl Into<String>, value: Value) -> Self {
        self.add(Filter::GreaterThanOrEqual {
            field: field.into(),
            value,
        })
    }

    /// Add a less-than-or-equal filter: `field <= value`
    #[must_use]
    pub fn less_than_or_equal(self, field: impl Into<String>, value: Value) -> Self {
        self.add(Filter::LessThanOrEqual {
            field: field.into(),
            value,
        })
    }

    /// Combine with AND: `self AND other`
    #[must_use]
    pub fn and(mut self, other: Filter) -> Self {
        if let Some(existing) = self.filter {
            self.filter = Some(Filter::And {
                left: Box::new(existing),
                right: Box::new(other),
            });
        } else {
            self.filter = Some(other);
        }
        self
    }

    /// Combine with OR: `self OR other`
    #[must_use]
    pub fn or(mut self, other: Filter) -> Self {
        if let Some(existing) = self.filter {
            self.filter = Some(Filter::Or {
                left: Box::new(existing),
                right: Box::new(other),
            });
        } else {
            self.filter = Some(other);
        }
        self
    }

    /// Negate the filter: `NOT self`
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn not(mut self) -> Self {
        if let Some(existing) = self.filter {
            self.filter = Some(Filter::Not {
                filter: Box::new(existing),
            });
        }
        self
    }

    /// Build the final filter.
    #[must_use]
    pub fn build(self) -> Option<Filter> {
        self.filter
    }

    /// Add a filter, combining with existing using AND if present.
    fn add(mut self, filter: Filter) -> Self {
        if let Some(existing) = self.filter {
            self.filter = Some(Filter::And {
                left: Box::new(existing),
                right: Box::new(filter),
            });
        } else {
            self.filter = Some(filter);
        }
        self
    }
}

impl From<Filter> for FilterBuilder {
    fn from(filter: Filter) -> Self {
        Self {
            filter: Some(filter),
        }
    }
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equal { field, value } => write!(f, "{field} == {value:?}"),
            Self::NotEqual { field, value } => write!(f, "{field} != {value:?}"),
            Self::In { field, values } => write!(f, "{field} IN {values:?}"),
            Self::NotIn { field, values } => write!(f, "{field} NOT IN {values:?}"),
            Self::GreaterThan { field, value } => write!(f, "{field} > {value:?}"),
            Self::LessThan { field, value } => write!(f, "{field} < {value:?}"),
            Self::GreaterThanOrEqual { field, value } => {
                write!(f, "{field} >= {value:?}")
            }
            Self::LessThanOrEqual { field, value } => write!(f, "{field} <= {value:?}"),
            Self::And { left, right } => write!(f, "({left}) AND ({right})"),
            Self::Or { left, right } => write!(f, "({left}) OR ({right})"),
            Self::Not { filter: inner } => write!(f, "NOT ({inner})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_equal_filter() {
        let filter = Filter::Equal {
            field: "category".to_string(),
            value: json!("A"),
        };
        let metadata = json!({"category": "A"});
        assert!(filter.matches(&metadata));

        let metadata2 = json!({"category": "B"});
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_and_filter() {
        let filter = Filter::And {
            left: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("A"),
            }),
            right: Box::new(Filter::GreaterThan {
                field: "price".to_string(),
                value: json!(100),
            }),
        };
        let metadata = json!({"category": "A", "price": 150});
        assert!(filter.matches(&metadata));

        let metadata2 = json!({"category": "A", "price": 50});
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_or_filter() {
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
        let metadata1 = json!({"category": "A"});
        assert!(filter.matches(&metadata1));

        let metadata2 = json!({"category": "B"});
        assert!(filter.matches(&metadata2));

        let metadata3 = json!({"category": "C"});
        assert!(!filter.matches(&metadata3));
    }

    #[test]
    fn test_builder() {
        let filter = FilterBuilder::new()
            .equal("category", json!("A"))
            .and(Filter::GreaterThan {
                field: "price".to_string(),
                value: json!(100),
            })
            .build()
            .expect("Failed to build filter");

        let metadata = json!({"category": "A", "price": 150});
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_in_filter() {
        let filter = Filter::In {
            field: "category".to_string(),
            values: vec![json!("A"), json!("B")],
        };
        let metadata1 = json!({"category": "A"});
        assert!(filter.matches(&metadata1));

        let metadata2 = json!({"category": "C"});
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_string_lexicographic_comparison() {
        let filter = Filter::GreaterThan {
            field: "name".to_string(),
            value: json!("apple"),
        };
        assert!(filter.matches(&json!({"name": "banana"})));
        assert!(!filter.matches(&json!({"name": "apple"})));
        assert!(!filter.matches(&json!({"name": "aardvark"})));

        let shorter_but_greater = Filter::GreaterThan {
            field: "name".to_string(),
            value: json!("alpha"),
        };
        assert!(shorter_but_greater.matches(&json!({"name": "b"})));

        let longer_but_less = Filter::LessThan {
            field: "name".to_string(),
            value: json!("b"),
        };
        assert!(longer_but_less.matches(&json!({"name": "alpha_very_long"})));
    }

    #[test]
    fn test_numeric_cross_int_float() {
        let filter = Filter::GreaterThanOrEqual {
            field: "price".to_string(),
            value: json!(100),
        };
        assert!(filter.matches(&json!({"price": 100.0})));
        assert!(filter.matches(&json!({"price": 100.5})));
        assert!(!filter.matches(&json!({"price": 99.9})));
    }

    #[test]
    fn test_nested_field() {
        let filter = Filter::Equal {
            field: "user.name".to_string(),
            value: json!("John"),
        };
        let metadata = json!({"user": {"name": "John"}});
        assert!(filter.matches(&metadata));
    }
}
