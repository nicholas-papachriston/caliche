//! Argument parsing for `fs op=edit`.
//!
//! Line numbers may arrive as JSON numbers, integral floats, or numeric strings.
//! A hash field may be `ab12` or a pasted anchor `42:ab12| content`. `operations`
//! may be an array, one object, or a JSON string of either.

use crate::agent::tools::common::params;
use crate::agent::tools::hashline;
use anyhow::{Result, bail};

pub(crate) fn require_line(op: &serde_json::Value, key: &str, idx: usize) -> Result<usize> {
    match op.get(key) {
        None | Some(serde_json::Value::Null) => line_missing(key, idx),
        Some(value) => params::coerce_u64(value)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| invalid_line(key, idx, value)),
    }
}

fn line_missing(key: &str, idx: usize) -> Result<usize> {
    bail!(
        "Operation {}: '{key}' is required (1-indexed line number). \
         Hashline edits copy the tag from fs op=read, for example start_line=42 start_hash=\"ab12\".",
        idx + 1
    )
}

fn invalid_line(key: &str, idx: usize, value: &serde_json::Value) -> anyhow::Error {
    anyhow::anyhow!(
        "Operation {}: '{key}' must be a 1-indexed line number, got {value}. \
         Numeric strings such as \"42\" are accepted.",
        idx + 1
    )
}

/// Line number, or the line embedded in a `42:ab12` hash anchor when the line field is absent.
pub(crate) fn line_or_anchor(
    op: &serde_json::Value,
    line_key: &str,
    hash_key: &str,
    idx: usize,
) -> Result<usize> {
    if op.get(line_key).is_some() && !op[line_key].is_null() {
        return require_line(op, line_key, idx);
    }
    if let Some(raw) = op.get(hash_key).and_then(serde_json::Value::as_str)
        && let Ok(anchor) = hashline::parse_hash_anchor(raw)
        && let Some(line) = anchor.line_hint
    {
        return Ok(line);
    }
    line_missing(line_key, idx)
}

pub(crate) fn require_hash<'a>(
    op: &'a serde_json::Value,
    key: &str,
    idx: usize,
) -> Result<&'a str> {
    match op.get(key).and_then(serde_json::Value::as_str) {
        Some(hash) if !hash.trim().is_empty() => Ok(hash),
        _ => bail!(
            "Operation {}: '{key}' is required. Copy the 4 hex characters from fs op=read \
             (for example ab12, or the anchor 42:ab12). A legacy 2-hex hash is accepted only \
             at the hinted line.",
            idx + 1
        ),
    }
}

pub(crate) fn coerce_operations(
    arguments: &serde_json::Value,
) -> Result<Vec<serde_json::Value>, String> {
    match arguments.get("operations") {
        Some(serde_json::Value::Array(ops)) if !ops.is_empty() => Ok(ops.clone()),
        Some(serde_json::Value::String(raw)) => parse_operations_string(raw),
        Some(serde_json::Value::Object(op)) => Ok(vec![serde_json::Value::Object(op.clone())]),
        _ => Err(
            "fs edit requires 'operations' as a non-empty array. Example: \
             [{\"type\":\"hashline_delete\",\"start_line\":10,\"end_line\":12,\"start_hash\":\"ab12\",\"end_hash\":\"cd34\"}]"
                .to_string(),
        ),
    }
}

fn parse_operations_string(raw: &str) -> Result<Vec<serde_json::Value>, String> {
    let parsed = serde_json::from_str::<serde_json::Value>(raw.trim()).map_err(|error| {
        format!(
            "operations was a string and is not JSON ({error}). Pass an array of edit objects, not a string."
        )
    })?;
    match parsed {
        serde_json::Value::Array(ops) if !ops.is_empty() => Ok(ops),
        serde_json::Value::Object(op) => Ok(vec![serde_json::Value::Object(op)]),
        _ => Err(
            "operations string must be a JSON array of edit objects, or one edit object."
                .to_string(),
        ),
    }
}
