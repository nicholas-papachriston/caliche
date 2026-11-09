//! JSON and JSONL document loader.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use serde_json::{Map, Value};
use std::path::PathBuf;
use tokio::fs;

/// Resolve a dotted / bracketed JSON path (e.g. `"data.items[0].content"`).
///
/// Supports:
/// - Dotted object keys: `a.b.c`
/// - Array indices: `a[0]` or `a.0`
///
/// Returns `None` when any segment is missing.
fn resolve_pointer<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    let mut buf = String::new();
    let mut chars = path.chars();

    let commit = |buf: &mut String, current: &mut &Value| -> bool {
        if buf.is_empty() {
            return true;
        }
        let segment = std::mem::take(buf);
        let next = match *current {
            Value::Object(map) => map.get(segment.as_str()),
            Value::Array(arr) => segment.parse::<usize>().ok().and_then(|idx| arr.get(idx)),
            _ => None,
        };
        next.map_or(false, |v| {
            *current = v;
            true
        })
    };

    while let Some(c) = chars.next() {
        match c {
            '.' => {
                if !commit(&mut buf, &mut current) {
                    return None;
                }
            }
            '[' => {
                if !commit(&mut buf, &mut current) {
                    return None;
                }
                let mut index_buf = String::new();
                for ic in chars.by_ref() {
                    if ic == ']' {
                        break;
                    }
                    index_buf.push(ic);
                }
                let idx: usize = index_buf.parse().ok()?;
                current = match current {
                    Value::Array(arr) => arr.get(idx)?,
                    _ => return None,
                };
            }
            other => buf.push(other),
        }
    }
    if !commit(&mut buf, &mut current) {
        return None;
    }
    Some(current)
}

fn value_to_content(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

fn value_to_metadata_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        v => v.to_string(),
    }
}

fn flatten_metadata(
    obj: &Map<String, Value>,
    skip_top_key: Option<&str>,
) -> std::collections::HashMap<String, String> {
    let mut metadata = std::collections::HashMap::with_capacity(obj.len());
    for (key, val) in obj {
        if Some(key.as_str()) == skip_top_key {
            continue;
        }
        metadata.insert(key.clone(), value_to_metadata_string(val));
    }
    metadata
}

/// Load documents from JSON or JSONL files.
#[derive(Debug, Clone)]
pub struct JsonLoader {
    path: PathBuf,
    content_key: Option<String>,
    is_jsonl: bool,
}

impl JsonLoader {
    /// Create a new JSON loader.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let is_jsonl = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext == "jsonl");

        Self {
            path,
            content_key: None,
            is_jsonl,
        }
    }

    /// Set which key contains the main content.
    #[must_use]
    pub fn with_content_key(mut self, key: impl Into<String>) -> Self {
        self.content_key = Some(key.into());
        self
    }

    /// Set whether this is JSONL format (one JSON object per line).
    #[must_use]
    pub const fn with_jsonl(mut self, is_jsonl: bool) -> Self {
        self.is_jsonl = is_jsonl;
        self
    }

    fn extract_document(&self, value: Value) -> Result<Document, LoaderError> {
        match value {
            Value::String(s) => Ok(Document::new(s)),
            Value::Object(obj) => {
                let content_path = self.content_key.as_deref();
                // Only mask the top-level key from metadata when the pointer
                // has no dots/brackets; dotted paths can target nested values
                // and should not strip sibling metadata.
                let top_key_hint = content_path.and_then(|path| {
                    if path.contains('.') || path.contains('[') {
                        None
                    } else {
                        Some(path)
                    }
                });

                // Wrap once in a `Value::Object` so `resolve_pointer` can hand
                // back borrowed references without cloning the underlying map.
                let value = Value::Object(obj);
                #[expect(
                    clippy::unreachable,
                    reason = "Value::Object was just constructed above"
                )]
                let Value::Object(ref obj_ref) = value else {
                    unreachable!("value is a Value::Object we just constructed");
                };

                let content = if let Some(path) = content_path {
                    match resolve_pointer(&value, path) {
                        Some(v) => value_to_content(v),
                        None => serde_json::to_string_pretty(&value).map_err(|e| {
                            LoaderError::ParseError(format!(
                                "Failed to serialize JSON for content fallback: {e}"
                            ))
                        })?,
                    }
                } else {
                    serde_json::to_string_pretty(&value).map_err(|e| {
                        LoaderError::ParseError(format!(
                            "Failed to serialize JSON for content fallback: {e}"
                        ))
                    })?
                };

                let metadata = flatten_metadata(obj_ref, top_key_hint);
                Ok(Document::new(content).with_metadata_map(metadata))
            }
            v => Ok(Document::new(v.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_pointer_dotted_path() {
        let v: Value = serde_json::json!({
            "data": {
                "items": [
                    {"content": "hello world"},
                    {"content": "second"}
                ]
            }
        });

        let found = resolve_pointer(&v, "data.items[0].content").and_then(|x| x.as_str());
        assert_eq!(found, Some("hello world"));

        let found2 = resolve_pointer(&v, "data.items[1].content").and_then(|x| x.as_str());
        assert_eq!(found2, Some("second"));
    }

    #[test]
    fn resolve_pointer_missing_returns_none() {
        let v: Value = serde_json::json!({"a": {"b": 1}});
        assert!(resolve_pointer(&v, "a.c").is_none());
        assert!(resolve_pointer(&v, "a[0]").is_none());
    }
}

#[async_trait]
impl DocumentLoader for JsonLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let content = fs::read_to_string(&self.path).await?;

        if self.is_jsonl {
            // Parse JSONL (one JSON per line)
            let mut documents = Vec::new();
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(line)
                    .map_err(|e| LoaderError::ParseError(format!("Invalid JSON line: {e}")))?;
                documents.push(self.extract_document(value)?);
            }
            Ok(documents)
        } else {
            // Parse regular JSON
            let value: Value = serde_json::from_str(&content)
                .map_err(|e| LoaderError::ParseError(format!("Invalid JSON: {e}")))?;

            match value {
                Value::Array(arr) => {
                    let mut documents = Vec::new();
                    for item in arr {
                        documents.push(self.extract_document(item)?);
                    }
                    Ok(documents)
                }
                v => Ok(vec![self.extract_document(v)?]),
            }
        }
    }

    fn name(&self) -> &'static str {
        if self.is_jsonl { "jsonl" } else { "json" }
    }
}
