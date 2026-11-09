use crate::plugins::step_dsl::{StepContext, TransformOperation};
use anyhow::{Result, anyhow};
use regex::Regex;

/// Apply transform operations to a resolved input value (used by `StepRuntime` and tests).
///
/// - **filter**: On arrays, keeps elements where `resolve_path(elem, path) == value`. On any
///   other JSON value, keeps the value if the path matches, else returns JSON null.
/// - **flatten**: One-level flatten of JSON arrays only; non-arrays are errors.
/// - **string_format**: Replaces every `{{value}}` with the current value, then runs [`super::interpolate`].
/// - **regex_extract**: Group `0` is the full match; group `1` is the first capture, etc.
/// - **merge**: Shallow merge; keys from `with` overwrite keys already in the current object.
pub(crate) fn run_transform_operations(
    mut current: serde_json::Value,
    operations: &[TransformOperation],
    ctx: &StepContext,
) -> Result<serde_json::Value> {
    for op in operations {
        current = apply_transform_operation(current, op, ctx)?;
    }
    Ok(current)
}

fn apply_transform_operation(
    current: serde_json::Value,
    op: &TransformOperation,
    ctx: &StepContext,
) -> Result<serde_json::Value> {
    match op {
        TransformOperation::JsonPath { path } => super::resolve_path(&current, path),
        TransformOperation::MapFields { mapping } => apply_map_fields(current, mapping),
        TransformOperation::Filter { path, value } => Ok(apply_filter(current, path, value)),
        TransformOperation::Flatten { path } => apply_flatten(current, path),
        TransformOperation::StringFormat { template } => {
            apply_string_format(current, template, ctx)
        }
        TransformOperation::RegexExtract { pattern, group } => {
            apply_regex_extract(&current, pattern, *group)
        }
        TransformOperation::Merge { with } => apply_merge(current, with, ctx),
    }
}

fn apply_map_fields(
    current: serde_json::Value,
    mapping: &std::collections::HashMap<String, String>,
) -> Result<serde_json::Value> {
    current
        .as_object()
        .ok_or_else(|| anyhow!("map_fields requires object"))?;
    let mut out = serde_json::Map::new();
    for (k, v) in mapping {
        let val = super::resolve_path(&current, v)?;
        out.insert(k.clone(), val);
    }
    Ok(serde_json::Value::Object(out))
}

fn apply_filter(
    current: serde_json::Value,
    path: &str,
    value: &serde_json::Value,
) -> serde_json::Value {
    match &current {
        serde_json::Value::Array(items) => {
            let filtered: Vec<serde_json::Value> = items
                .iter()
                .filter(|elem| filter_elem_matches(elem, path, value))
                .cloned()
                .collect();
            serde_json::Value::Array(filtered)
        }
        _ => {
            let got = super::resolve_path(&current, path).unwrap_or(serde_json::Value::Null);
            if got == *value {
                current
            } else {
                serde_json::Value::Null
            }
        }
    }
}

fn filter_elem_matches(elem: &serde_json::Value, path: &str, value: &serde_json::Value) -> bool {
    super::resolve_path(elem, path)
        .map(|v| &v == value)
        .unwrap_or(false)
}

fn apply_flatten(current: serde_json::Value, path: &str) -> Result<serde_json::Value> {
    let subject = if path.is_empty() {
        current
    } else {
        super::resolve_path(&current, path)?
    };
    let arr = subject.as_array().ok_or_else(|| flatten_type_error(path))?;
    Ok(serde_json::Value::Array(flatten_one_level(arr)))
}

fn flatten_type_error(path: &str) -> anyhow::Error {
    if path.is_empty() {
        anyhow!("flatten requires the transform input to be a JSON array")
    } else {
        anyhow!("flatten requires a JSON array at path {path:?}; got non-array value")
    }
}

fn flatten_one_level(arr: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    for el in arr {
        if let serde_json::Value::Array(inner) = el {
            out.extend(inner.iter().cloned());
        } else {
            out.push(el.clone());
        }
    }
    out
}

fn apply_string_format(
    current: serde_json::Value,
    template: &str,
    ctx: &StepContext,
) -> Result<serde_json::Value> {
    let value_display = super::transform_value_placeholder(&current);
    let with_value = template.replace("{{value}}", &value_display);
    super::interpolate(&with_value, ctx)
}

fn apply_regex_extract(
    current: &serde_json::Value,
    pattern: &str,
    group: Option<usize>,
) -> Result<serde_json::Value> {
    let haystack = match current {
        serde_json::Value::String(s) => s.clone(),
        _ => super::value_to_string(current),
    };
    let re = Regex::new(pattern).map_err(|e| anyhow!("regex_extract invalid pattern: {e}"))?;
    let caps = re
        .captures(haystack.as_str())
        .ok_or_else(|| anyhow!("regex_extract: no match"))?;
    let idx = group.unwrap_or(0);
    let m = caps
        .get(idx)
        .ok_or_else(|| anyhow!("regex_extract: capture group {idx} not present"))?;
    Ok(serde_json::Value::String(m.as_str().to_owned()))
}

fn apply_merge(
    current: serde_json::Value,
    with: &str,
    ctx: &StepContext,
) -> Result<serde_json::Value> {
    let other = super::resolve_expression(with.trim(), ctx)?;
    let mut base = current
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("merge requires current value to be a JSON object"))?;
    let other_obj = other
        .as_object()
        .ok_or_else(|| anyhow!("merge requires 'with' to resolve to a JSON object"))?;
    for (k, v) in other_obj {
        base.insert(k.clone(), v.clone());
    }
    Ok(serde_json::Value::Object(base))
}
