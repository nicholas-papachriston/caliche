//! Lightweight JSON path evaluation for tools, workflows, and plugins.
//!
//! Supported forms:
//! - `$` (whole document)
//! - `$.key`, `$.key.sub`
//! - `$.key[0]`, `$[0]`, `$.key[*]`
//! - Dot paths without `$` (`key.sub[0]`) for plugin templates

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

/// Evaluate a JSON path against `data`.
///
/// Paths may start with `$` / `$.` (tool / workflow style) or omit the prefix
/// (plugin template style).
pub fn evaluate(data: &Value, path: &str) -> Result<Value> {
    let path = path.trim();
    if path.is_empty() || path == "$" {
        return Ok(data.clone());
    }

    let body = path
        .strip_prefix("$.")
        .or_else(|| path.strip_prefix('$'))
        .unwrap_or(path)
        .trim_start_matches('.');

    if body.is_empty() {
        return Ok(data.clone());
    }

    let mut current = data;
    for part in split_segments(body) {
        if part.is_empty() {
            continue;
        }
        match apply_path_segment(current, part)? {
            PathStep::Value(next) => current = next,
            PathStep::Wildcard(value) => return Ok(value),
        }
    }

    Ok(current.clone())
}

enum PathStep<'a> {
    Value(&'a Value),
    Wildcard(Value),
}

fn apply_path_segment<'a>(current: &'a Value, part: &str) -> Result<PathStep<'a>> {
    let Some(bracket_idx) = part.find('[') else {
        let next = current
            .get(part)
            .ok_or_else(|| anyhow!("Key '{part}' not found"))?;
        return Ok(PathStep::Value(next));
    };

    let key = &part[..bracket_idx];
    let remainder = &part[bracket_idx..];
    let mut current = current;
    if !key.is_empty() {
        current = current
            .get(key)
            .ok_or_else(|| anyhow!("Key '{key}' not found"))?;
    }
    apply_bracket_access(current, remainder)
}

fn apply_bracket_access<'a>(mut current: &'a Value, mut remainder: &str) -> Result<PathStep<'a>> {
    while remainder.starts_with('[') {
        let Some(end) = remainder.find(']') else {
            bail!("Invalid array access syntax");
        };
        let inner = &remainder[1..end];
        remainder = &remainder[end + 1..];

        if inner == "*" {
            if !remainder.is_empty() {
                bail!("[*] must be the final path segment");
            }
            let arr = current
                .as_array()
                .ok_or_else(|| anyhow!("Cannot use [*] on non-array"))?;
            return Ok(PathStep::Wildcard(json!(arr.clone())));
        }

        let idx: usize = inner
            .parse()
            .map_err(|_| anyhow!("Invalid array index '{inner}'"))?;
        let arr = current
            .as_array()
            .ok_or_else(|| anyhow!("Cannot index non-array"))?;
        current = arr
            .get(idx)
            .ok_or_else(|| anyhow!("Index {idx} out of bounds"))?;
    }

    if !remainder.is_empty() {
        bail!("Invalid array access syntax");
    }
    Ok(PathStep::Value(current))
}

/// Split on `.` outside of `[...]` so `a[0].b` stays intact as `a[0]`, `b`.
fn split_segments(path: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let bytes = path.as_bytes();
    let mut i = 0;
    let mut in_brackets = false;

    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                in_brackets = true;
                i += 1;
            }
            b']' => {
                in_brackets = false;
                i += 1;
            }
            b'.' if !in_brackets => {
                if i > start {
                    segments.push(path[start..i].trim());
                }
                start = i + 1;
                i += 1;
            }
            _ => i += 1,
        }
    }

    if start < bytes.len() {
        segments.push(path[start..].trim());
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn root_and_dollar() {
        let data = json!({"a": 1});
        assert_eq!(evaluate(&data, "$").unwrap(), data);
        assert_eq!(evaluate(&data, "").unwrap(), data);
    }

    #[test]
    fn nested_keys_and_index() {
        let data = json!({"items": [{"name": "x"}, {"name": "y"}]});
        assert_eq!(evaluate(&data, "$.items[1].name").unwrap(), json!("y"));
        assert_eq!(evaluate(&data, "items[0].name").unwrap(), json!("x"));
    }

    #[test]
    fn wildcard_returns_array() {
        let data = json!({"items": [1, 2, 3]});
        assert_eq!(evaluate(&data, "$.items[*]").unwrap(), json!([1, 2, 3]));
    }

    #[test]
    fn nested_indexes() {
        let data = json!({"grid": [[1, 2], [3, 4]]});
        assert_eq!(evaluate(&data, "$.grid[1][0]").unwrap(), json!(3));
    }

    #[test]
    fn unclosed_bracket_is_error() {
        let data = json!({"items": [1]});
        assert!(evaluate(&data, "$.items[0").is_err());
        assert!(evaluate(&data, "$.items[*][0]").is_err());
    }
}
