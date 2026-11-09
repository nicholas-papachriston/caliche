//! Workflow condition evaluation (comparisons and lightweight JSON paths).

use crate::utils::json_path;
use serde_json::Value;

/// Evaluate a condition expression against context / node output.
///
/// Supports:
/// - `status == 'approved'`
/// - `count < 3` / `count > 3`
/// - `$.result.success == true`
/// - bare key truthiness (`approved`)
#[must_use]
pub fn evaluate_condition(condition: &str, context: &Value, node_output: Option<&Value>) -> bool {
    let eval_target = node_output.unwrap_or(context);
    let condition = condition.trim();

    if condition.starts_with("$.") {
        return evaluate_json_path_condition(condition, eval_target);
    }

    if let Some((key, expected)) = split_binary(condition, "==") {
        return compare_equality(eval_target.get(key.trim()), expected);
    }

    if let Some((key, rhs)) = split_binary(condition, "<") {
        return compare_numeric(eval_target.get(key.trim()), rhs, std::cmp::Ordering::Less);
    }

    if let Some((key, rhs)) = split_binary(condition, ">") {
        return compare_numeric(
            eval_target.get(key.trim()),
            rhs,
            std::cmp::Ordering::Greater,
        );
    }

    if let Some(val) = eval_target.get(condition) {
        if let Some(b) = val.as_bool() {
            return b;
        }
        return !val.is_null();
    }

    false
}

fn evaluate_json_path_condition(condition: &str, eval_target: &Value) -> bool {
    if let Some((path, expected)) = split_binary(condition, "==") {
        let Ok(current) = json_path::evaluate(eval_target, path.trim()) else {
            return false;
        };
        return compare_equality(Some(&current), expected);
    }

    match json_path::evaluate(eval_target, condition) {
        Ok(current) => !current.is_null(),
        Err(_) => false,
    }
}

fn split_binary<'a>(condition: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let (left, right) = condition.split_once(op)?;
    Some((left, right.trim().trim_matches('\'').trim_matches('"')))
}

fn compare_equality(val: Option<&Value>, expected: &str) -> bool {
    let Some(val) = val else {
        return false;
    };
    if let Some(val_str) = val.as_str() {
        return val_str == expected;
    }
    if let Some(val_num) = val.as_f64()
        && let Ok(expected_num) = expected.parse::<f64>()
    {
        return (val_num - expected_num).abs() < f64::EPSILON;
    }
    if let Some(val_bool) = val.as_bool()
        && let Ok(expected_bool) = expected.parse::<bool>()
    {
        return val_bool == expected_bool;
    }
    false
}

fn compare_numeric(val: Option<&Value>, rhs: &str, ordering: std::cmp::Ordering) -> bool {
    let Ok(threshold) = rhs.trim().parse::<f64>() else {
        return false;
    };
    let Some(val) = val else {
        return false;
    };
    let Some(val_num) = val.as_f64().or_else(|| val.as_i64().map(|n| n as f64)) else {
        return false;
    };
    val_num.partial_cmp(&threshold) == Some(ordering)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn equality_and_path() {
        let ctx = json!({"status": "approved", "result": {"ok": true}});
        assert!(evaluate_condition("status == 'approved'", &ctx, None));
        assert!(evaluate_condition("$.result.ok == true", &ctx, None));
        assert!(!evaluate_condition("status == 'denied'", &ctx, None));
    }

    #[test]
    fn numeric_and_truthy() {
        let ctx = json!({"count": 2, "flag": true});
        assert!(evaluate_condition("count < 3", &ctx, None));
        assert!(evaluate_condition("count > 1", &ctx, None));
        assert!(evaluate_condition("flag", &ctx, None));
    }
}
