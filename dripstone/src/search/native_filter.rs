use crate::errors::DripstoneError;
use crate::search::Filter;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug, PartialEq, Eq, Hash))]
pub struct NativeFilterKey {
    pub field: String,
    pub value_json: String,
}

/// Native-filter plan compiled from a [`Filter`] and namespace constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeFilterPlan {
    /// No filterable constraints; the ANN index should search normally.
    MatchAll,
    /// Equality-only conjunction that can be answered from sidecar postings.
    EqualityConjunction(EqualityConjunctionPlan),
    /// OR-of-conjunctions that can be answered from sidecar postings.
    Disjunction(Vec<EqualityConjunctionPlan>),
    /// Filter shape is not supported by the native ANN filter path.
    Unsupported,
}

impl NativeFilterPlan {
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqualityConjunctionPlan {
    pub namespace: Option<String>,
    pub terms: Vec<EqualityTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EqualityTerm {
    Equal { field: String, value: Value },
    In { field: String, values: Vec<Value> },
}

///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn json_key(value: &Value) -> Result<String, DripstoneError> {
    serde_json::to_string(value)
        .map_err(|err| DripstoneError::storage(format!("Failed to encode filter value: {err}")))
}

///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn collect_filter_keys(
    metadata: &Value,
    prefix: Option<&str>,
    out: &mut Vec<NativeFilterKey>,
) -> Result<(), DripstoneError> {
    let Some(map) = metadata.as_object() else {
        return Ok(());
    };

    for (key, value) in map {
        let field = prefix.map_or_else(|| key.clone(), |prefix| format!("{prefix}.{key}"));

        if value.is_object() {
            collect_filter_keys(value, Some(&field), out)?;
            continue;
        }

        out.push(NativeFilterKey {
            field,
            value_json: json_key(value)?,
        });
    }

    Ok(())
}

pub fn compile_native_filter_plan(
    namespace: Option<&str>,
    filter: Option<&Filter>,
) -> NativeFilterPlan {
    let Some(filter) = filter else {
        return base_plan(namespace);
    };

    let Some(mut conjunctions) = compile_supported_conjunctions(filter) else {
        return NativeFilterPlan::Unsupported;
    };

    if conjunctions.is_empty() {
        return base_plan(namespace);
    }

    for conjunction in &mut conjunctions {
        conjunction.namespace = namespace.map(ToOwned::to_owned);
    }

    if conjunctions.len() == 1 {
        let Some(conjunction) = conjunctions.pop() else {
            return base_plan(namespace);
        };
        NativeFilterPlan::EqualityConjunction(conjunction)
    } else {
        NativeFilterPlan::Disjunction(conjunctions)
    }
}

fn base_plan(namespace: Option<&str>) -> NativeFilterPlan {
    namespace.map_or(NativeFilterPlan::MatchAll, |namespace| {
        NativeFilterPlan::EqualityConjunction(EqualityConjunctionPlan {
            namespace: Some(namespace.to_string()),
            terms: Vec::new(),
        })
    })
}

fn compile_supported_conjunctions(filter: &Filter) -> Option<Vec<EqualityConjunctionPlan>> {
    match filter {
        Filter::Equal { field, value } => Some(vec![EqualityConjunctionPlan {
            namespace: None,
            terms: vec![EqualityTerm::Equal {
                field: field.clone(),
                value: value.clone(),
            }],
        }]),
        Filter::In { field, values } => Some(vec![EqualityConjunctionPlan {
            namespace: None,
            terms: vec![EqualityTerm::In {
                field: field.clone(),
                values: values.clone(),
            }],
        }]),
        Filter::And { left, right } => {
            let left_plans = compile_supported_conjunctions(left)?;
            let right_plans = compile_supported_conjunctions(right)?;
            let mut conjunctions = Vec::with_capacity(left_plans.len() * right_plans.len());
            for left_plan in &left_plans {
                for right_plan in &right_plans {
                    let mut terms =
                        Vec::with_capacity(left_plan.terms.len() + right_plan.terms.len());
                    terms.extend(left_plan.terms.iter().cloned());
                    terms.extend(right_plan.terms.iter().cloned());
                    conjunctions.push(EqualityConjunctionPlan {
                        namespace: None,
                        terms,
                    });
                }
            }
            Some(conjunctions)
        }
        Filter::Or { left, right } => {
            let mut left_plans = compile_supported_conjunctions(left)?;
            left_plans.extend(compile_supported_conjunctions(right)?);
            Some(left_plans)
        }
        Filter::NotEqual { .. }
        | Filter::NotIn { .. }
        | Filter::GreaterThan { .. }
        | Filter::LessThan { .. }
        | Filter::GreaterThanOrEqual { .. }
        | Filter::LessThanOrEqual { .. }
        | Filter::Not { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EqualityTerm, NativeFilterPlan, compile_native_filter_plan};
    use crate::search::Filter;

    #[test]
    fn compiles_namespace_only_plan() {
        let plan = compile_native_filter_plan(Some("forge"), None);
        assert!(matches!(
            plan,
            NativeFilterPlan::EqualityConjunction(ref eq)
            if eq.namespace.as_deref() == Some("forge") && eq.terms.is_empty()
        ));
    }

    #[test]
    fn compiles_equality_conjunction() {
        let filter = Filter::And {
            left: Box::new(Filter::Equal {
                field: "domain".to_string(),
                value: json!("docs"),
            }),
            right: Box::new(Filter::In {
                field: "kind".to_string(),
                values: vec![json!("artifact"), json!("chunk")],
            }),
        };

        let plan = compile_native_filter_plan(Some("forge"), Some(&filter));
        assert!(matches!(
            plan,
            NativeFilterPlan::EqualityConjunction(ref eq)
            if eq.namespace.as_deref() == Some("forge")
                && eq.terms
                    == vec![
                        EqualityTerm::Equal {
                            field: "domain".to_string(),
                            value: json!("docs"),
                        },
                        EqualityTerm::In {
                            field: "kind".to_string(),
                            values: vec![json!("artifact"), json!("chunk")],
                        },
                    ]
        ));
    }

    #[test]
    fn compiles_or_of_supported_conjunctions() {
        let filter = Filter::Or {
            left: Box::new(Filter::Equal {
                field: "domain".to_string(),
                value: json!("docs"),
            }),
            right: Box::new(Filter::And {
                left: Box::new(Filter::Equal {
                    field: "domain".to_string(),
                    value: json!("code"),
                }),
                right: Box::new(Filter::In {
                    field: "kind".to_string(),
                    values: vec![json!("artifact"), json!("chunk")],
                }),
            }),
        };

        let plan = compile_native_filter_plan(Some("forge"), Some(&filter));
        assert!(matches!(
            plan,
            NativeFilterPlan::Disjunction(ref conjunctions)
                if conjunctions.len() == 2
                    && conjunctions[0].namespace.as_deref() == Some("forge")
                    && conjunctions[0].terms
                        == vec![EqualityTerm::Equal {
                            field: "domain".to_string(),
                            value: json!("docs"),
                        }]
                    && conjunctions[1].namespace.as_deref() == Some("forge")
                    && conjunctions[1].terms
                        == vec![
                            EqualityTerm::Equal {
                                field: "domain".to_string(),
                                value: json!("code"),
                            },
                            EqualityTerm::In {
                                field: "kind".to_string(),
                                values: vec![json!("artifact"), json!("chunk")],
                            },
                        ]
        ));
    }

    #[test]
    fn marks_or_with_unsupported_branch_unsupported() {
        let filter = Filter::Or {
            left: Box::new(Filter::Equal {
                field: "domain".to_string(),
                value: json!("docs"),
            }),
            right: Box::new(Filter::GreaterThan {
                field: "score".to_string(),
                value: json!(42),
            }),
        };

        assert!(matches!(
            compile_native_filter_plan(None, Some(&filter)),
            NativeFilterPlan::Unsupported
        ));
    }

    #[test]
    fn marks_range_filters_unsupported() {
        let filter = Filter::GreaterThan {
            field: "score".to_string(),
            value: json!(42),
        };

        assert!(matches!(
            compile_native_filter_plan(None, Some(&filter)),
            NativeFilterPlan::Unsupported
        ));
    }
}
