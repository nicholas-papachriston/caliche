//! Exact post-filtering for ANN search candidates after native index filtering.
//!
//! Native indexes apply [`NativeFilterPlan`] constraints (equality conjunctions and
//! namespace postings). This module applies the remaining exact checks: namespace
//! when the plan does not already constrain it, and full [`Filter::matches`] on
//! metadata for operators the native path cannot represent.

use serde_json::Value;

use crate::search::{Filter, NativeFilterPlan};

/// Context for exact post-filtering of ANN candidates.
#[derive(Debug, Clone, Copy)]
pub struct SearchPostFilter<'a> {
    pub query_namespace: Option<&'a str>,
    pub filter: Option<&'a Filter>,
    pub native_plan: &'a NativeFilterPlan,
}

impl SearchPostFilter<'_> {
    /// Returns whether `candidate_namespace` / `metadata` survive exact post-filtering.
    #[must_use]
    pub fn passes(&self, candidate_namespace: Option<&str>, metadata: &Value) -> bool {
        search_post_filter_passes(
            self.query_namespace,
            candidate_namespace,
            self.filter,
            metadata,
            self.native_plan,
        )
    }
}

/// Returns whether the native ANN plan already constrains `query_namespace`.
#[must_use]
pub fn native_plan_covers_namespace(
    plan: &NativeFilterPlan,
    query_namespace: Option<&str>,
) -> bool {
    let Some(query_ns) = query_namespace else {
        return true;
    };

    match plan {
        NativeFilterPlan::MatchAll | NativeFilterPlan::Unsupported => false,
        NativeFilterPlan::EqualityConjunction(conjunction) => {
            conjunction.namespace.as_deref() == Some(query_ns)
        }
        NativeFilterPlan::Disjunction(conjunctions) => {
            !conjunctions.is_empty()
                && conjunctions
                    .iter()
                    .all(|c| c.namespace.as_deref() == Some(query_ns))
        }
    }
}

/// Returns whether an ANN candidate should be kept after exact namespace and metadata checks.
#[must_use]
pub fn search_post_filter_passes(
    query_namespace: Option<&str>,
    candidate_namespace: Option<&str>,
    filter: Option<&Filter>,
    metadata: &Value,
    native_plan: &NativeFilterPlan,
) -> bool {
    if let Some(ns_filter) = query_namespace
        && !native_plan_covers_namespace(native_plan, query_namespace)
    {
        let candidate_ns = candidate_namespace.unwrap_or("default");
        if candidate_ns != ns_filter {
            return false;
        }
    }

    if let Some(filter) = filter
        && !filter.matches(metadata)
    {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{SearchPostFilter, native_plan_covers_namespace, search_post_filter_passes};
    use crate::search::{EqualityTerm, Filter, NativeFilterPlan, compile_native_filter_plan};

    #[test]
    fn no_constraints_always_passes() {
        let plan = NativeFilterPlan::MatchAll;
        assert!(search_post_filter_passes(
            None,
            None,
            None,
            &json!({}),
            &plan
        ));
    }

    #[test]
    fn metadata_filter_rejects_non_matching() {
        let filter = Filter::Equal {
            field: "domain".to_string(),
            value: json!("docs"),
        };
        let plan = compile_native_filter_plan(None, Some(&filter));
        let metadata = json!({"domain": "other"});
        assert!(!search_post_filter_passes(
            None,
            None,
            Some(&filter),
            &metadata,
            &plan
        ));
        assert!(search_post_filter_passes(
            None,
            None,
            Some(&filter),
            &json!({"domain": "docs"}),
            &plan
        ));
    }

    #[test]
    fn namespace_check_when_plan_does_not_cover() {
        let plan = NativeFilterPlan::Unsupported;
        assert!(!search_post_filter_passes(
            Some("forge"),
            Some("other"),
            None,
            &json!({}),
            &plan
        ));
        assert!(search_post_filter_passes(
            Some("forge"),
            Some("forge"),
            None,
            &json!({}),
            &plan
        ));
    }

    #[test]
    fn namespace_default_when_candidate_missing() {
        let plan = NativeFilterPlan::Unsupported;
        assert!(search_post_filter_passes(
            Some("default"),
            None,
            None,
            &json!({}),
            &plan
        ));
        assert!(!search_post_filter_passes(
            Some("forge"),
            None,
            None,
            &json!({}),
            &plan
        ));
    }

    #[test]
    fn skips_namespace_when_native_plan_covers() {
        let plan = compile_native_filter_plan(Some("forge"), None);
        assert!(native_plan_covers_namespace(&plan, Some("forge")));
        assert!(search_post_filter_passes(
            Some("forge"),
            Some("other"),
            None,
            &json!({}),
            &plan
        ));
    }

    #[test]
    fn search_post_filter_struct_delegates() {
        let filter = Filter::GreaterThan {
            field: "tier".to_string(),
            value: json!(0),
        };
        let plan = compile_native_filter_plan(Some("forge"), Some(&filter));
        let ctx = SearchPostFilter {
            query_namespace: None,
            filter: Some(&filter),
            native_plan: &plan,
        };
        assert!(ctx.passes(Some("forge"), &json!({"tier": 1})));
        assert!(!ctx.passes(Some("forge"), &json!({"tier": 0})));
    }

    #[test]
    fn disjunction_plan_covers_namespace_when_all_branches_do() {
        let plan = NativeFilterPlan::Disjunction(vec![
            crate::search::EqualityConjunctionPlan {
                namespace: Some("forge".to_string()),
                terms: vec![EqualityTerm::Equal {
                    field: "domain".to_string(),
                    value: json!("docs"),
                }],
            },
            crate::search::EqualityConjunctionPlan {
                namespace: Some("forge".to_string()),
                terms: vec![EqualityTerm::Equal {
                    field: "domain".to_string(),
                    value: json!("code"),
                }],
            },
        ]);
        assert!(native_plan_covers_namespace(&plan, Some("forge")));
    }
}
