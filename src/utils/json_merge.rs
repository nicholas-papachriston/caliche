//! Canonical JSON value merging with explicit merge modes.
//!
//! SharedState uses [`JsonMergeMode::Deep`]; the `merge_json` data-manipulation tool
//! supports all modes via [`json_merge_mode_from_str`].
//!
//! Merging applies only when both `target` and `incoming` are JSON objects at the
//! current level. For any other type pairing (array vs object, scalar vs object, etc.),
//! `incoming` fully replaces `target`.
//!
//! Arrays are never merged element-wise; an incoming array always replaces the target
//! array when types do not both resolve to object-object merging at that level.
//!
//! # Merge modes
//!
//! - [`JsonMergeMode::Deep`]: For each key in `incoming`, recurse when the key already
//!   exists on `target` and both values are objects; otherwise insert or replace.
//! - [`JsonMergeMode::Shallow`]: At the current object level only, insert keys from
//!   `incoming` when `target` does not already have that key; existing keys are left
//!   unchanged (no recursion into nested objects).
//! - [`JsonMergeMode::Replace`]: At the current object level, every key from `incoming`
//!   overwrites or inserts on `target` without recursing into nested objects.

use anyhow::{Result, bail};

/// How [`merge_json_value`] combines `incoming` into `target`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonMergeMode {
    /// Recursive object merge; non-object conflicts replace `target`.
    Deep,
    /// Insert only missing keys at the current object level.
    Shallow,
    /// Incoming keys always win at the current object level.
    Replace,
}

/// Parse a tool/API merge mode string (`deep`, `shallow`, `replace`).
///
/// Matching is case-insensitive and trims surrounding whitespace.
pub fn json_merge_mode_from_str(mode: &str) -> Result<JsonMergeMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "deep" => Ok(JsonMergeMode::Deep),
        "shallow" => Ok(JsonMergeMode::Shallow),
        "replace" => Ok(JsonMergeMode::Replace),
        other => bail!("Unknown merge mode: {other}"),
    }
}

/// Merge `incoming` into `target` in place according to `mode`.
pub fn merge_json_value(
    target: &mut serde_json::Value,
    incoming: serde_json::Value,
    mode: JsonMergeMode,
) {
    match mode {
        JsonMergeMode::Deep => merge_deep(target, incoming),
        JsonMergeMode::Shallow => merge_shallow(target, incoming),
        JsonMergeMode::Replace => merge_replace(target, incoming),
    }
}

fn merge_deep(target: &mut serde_json::Value, incoming: serde_json::Value) {
    match (target, incoming) {
        (serde_json::Value::Object(target_obj), serde_json::Value::Object(incoming_obj)) => {
            for (key, incoming_value) in incoming_obj {
                match target_obj.get_mut(&key) {
                    Some(existing_value) => merge_deep(existing_value, incoming_value),
                    None => {
                        target_obj.insert(key, incoming_value);
                    }
                }
            }
        }
        (target_value, incoming_value) => {
            *target_value = incoming_value;
        }
    }
}

fn merge_shallow(target: &mut serde_json::Value, incoming: serde_json::Value) {
    match (target, incoming) {
        (serde_json::Value::Object(target_obj), serde_json::Value::Object(incoming_obj)) => {
            for (key, incoming_value) in incoming_obj {
                if !target_obj.contains_key(&key) {
                    target_obj.insert(key, incoming_value);
                }
            }
        }
        (target_value, incoming_value) => {
            *target_value = incoming_value;
        }
    }
}

fn merge_replace(target: &mut serde_json::Value, incoming: serde_json::Value) {
    match (target, incoming) {
        (serde_json::Value::Object(target_obj), serde_json::Value::Object(incoming_obj)) => {
            for (key, incoming_value) in incoming_obj {
                target_obj.insert(key, incoming_value);
            }
        }
        (target_value, incoming_value) => {
            *target_value = incoming_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deep_nested_x_preserved_y_added() {
        let mut target = json!({ "nested": { "x": 1 } });
        merge_json_value(
            &mut target,
            json!({ "nested": { "y": 2 } }),
            JsonMergeMode::Deep,
        );
        assert_eq!(target, json!({ "nested": { "x": 1, "y": 2 } }));
    }

    #[test]
    fn shallow_existing_top_level_key_not_overwritten() {
        let mut target = json!({ "existing": 1 });
        merge_json_value(
            &mut target,
            json!({ "existing": 99, "new": 2 }),
            JsonMergeMode::Shallow,
        );
        assert_eq!(target, json!({ "existing": 1, "new": 2 }));
    }

    #[test]
    fn replace_conflicting_scalar_replaced() {
        let mut target = json!({ "scalar": 1 });
        merge_json_value(&mut target, json!({ "scalar": 2 }), JsonMergeMode::Replace);
        assert_eq!(target, json!({ "scalar": 2 }));
    }

    #[test]
    fn deep_array_and_null_replace_objects() {
        let mut target = json!({ "nested": { "x": 1 } });
        merge_json_value(
            &mut target,
            json!({ "nested": [1, 2] }),
            JsonMergeMode::Deep,
        );
        assert_eq!(target, json!({ "nested": [1, 2] }));

        merge_json_value(&mut target, json!({ "nested": null }), JsonMergeMode::Deep);
        assert_eq!(target, json!({ "nested": null }));
    }

    #[test]
    fn shallow_does_not_recurse_into_nested_objects() {
        let mut target = json!({ "nested": { "x": 1 } });
        merge_json_value(
            &mut target,
            json!({ "nested": { "y": 2 }, "top": 3 }),
            JsonMergeMode::Shallow,
        );
        assert_eq!(target, json!({ "nested": { "x": 1 }, "top": 3 }));
    }

    #[test]
    fn merge_mode_from_str_is_case_insensitive() {
        assert_eq!(
            json_merge_mode_from_str("Deep").unwrap(),
            JsonMergeMode::Deep
        );
        assert_eq!(
            json_merge_mode_from_str(" SHALLOW ").unwrap(),
            JsonMergeMode::Shallow
        );
        assert_eq!(
            json_merge_mode_from_str("Replace").unwrap(),
            JsonMergeMode::Replace
        );
        let err = json_merge_mode_from_str("merge").unwrap_err();
        assert!(err.to_string().contains("Unknown merge mode"));
    }
}
