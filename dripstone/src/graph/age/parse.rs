//! AGE agtype parsing helpers.

/// Split AGE internal map by top-level semicolons or commas (respecting nested `{}`).
pub(crate) fn split_age_map_parts(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth: i32 = 0;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b';' | b',' if depth == 0 => {
                parts.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < bytes.len() {
        parts.push(s[start..].trim().to_string());
    }
    parts
}

/// Parse a single value from AGE internal format: number, 'string', true/false, null, or { map }.
pub(crate) fn parse_age_internal_value(s: &str) -> Option<serde_json::Value> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if s == "null" {
        return Some(serde_json::Value::Null);
    }
    if s == "true" {
        return Some(serde_json::Value::Bool(true));
    }
    if s == "false" {
        return Some(serde_json::Value::Bool(false));
    }
    if let Some(quoted) = s.strip_prefix('\'').and_then(|t| t.strip_suffix('\'')) {
        return Some(serde_json::Value::String(quoted.to_string()));
    }
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len() - 1];
        let mut obj = serde_json::Map::new();
        for part in split_age_map_parts(inner) {
            if let Some((key, val_str)) = part.split_once(':') {
                let key = key
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'')
                    .to_string();
                let value =
                    parse_age_internal_value(val_str.trim()).unwrap_or(serde_json::Value::Null);
                obj.insert(key, value);
            }
        }
        return Some(serde_json::Value::Object(obj));
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(serde_json::Value::Number(n.into()));
    }
    if let Ok(n) = s.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(n) {
            return Some(serde_json::Value::Number(num));
        }
    }
    None
}
