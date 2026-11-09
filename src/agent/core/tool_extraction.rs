const OPEN_FUNCTION_CALLS: &str = "<function_calls>";
const CLOSE_FUNCTION_CALLS: &str = "</function_calls>";
const OPEN_TOOL_CALL: &str = "<tool_call>";
const CLOSE_TOOL_CALL: &str = "</tool_call>";

#[must_use]
pub fn extract_tool_calls_from_text(response: &str) -> Vec<crate::agent::tools::tool::ToolCall> {
    let from_function_calls = extract_from_function_calls_blocks(response);
    if !from_function_calls.is_empty() {
        return from_function_calls;
    }

    let from_tool_call = extract_from_tool_call_blocks(response);
    if !from_tool_call.is_empty() {
        return from_tool_call;
    }

    let trimmed = response.trim();
    let from_json = extract_from_perplexity_json(trimmed);
    if !from_json.is_empty() {
        return from_json;
    }

    extract_function_style_tool_calls(trimmed)
}

fn extract_from_function_calls_blocks(response: &str) -> Vec<crate::agent::tools::tool::ToolCall> {
    let mut tool_calls = Vec::new();
    let mut search_start = 0;
    while let Some(start) = response[search_start..].find(OPEN_FUNCTION_CALLS) {
        let absolute_start = search_start + start;
        let Some(end) = response[absolute_start..].find(CLOSE_FUNCTION_CALLS) else {
            break;
        };
        let absolute_end = absolute_start + end;
        let json_str = response[absolute_start + OPEN_FUNCTION_CALLS.len()..absolute_end].trim();
        if json_str.is_empty() || json_str == "[]" {
            search_start = absolute_end + CLOSE_FUNCTION_CALLS.len();
            continue;
        }
        append_json_array_tool_calls(json_str, &mut tool_calls);
        search_start = absolute_end + CLOSE_FUNCTION_CALLS.len();
    }
    tool_calls
}

fn append_json_array_tool_calls(
    json_str: &str,
    tool_calls: &mut Vec<crate::agent::tools::tool::ToolCall>,
) {
    let Ok(calls) = serde_json::from_str::<Vec<serde_json::Value>>(json_str) else {
        return;
    };
    for call in calls {
        let Some(tool_call) = tool_call_from_json_object(&call) else {
            continue;
        };
        tool_calls.push(tool_call);
    }
}

fn tool_call_from_json_object(
    call: &serde_json::Value,
) -> Option<crate::agent::tools::tool::ToolCall> {
    let name = call
        .get("tool_name")
        .or_else(|| call.get("name"))
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string)?;
    let arguments = call
        .get("arguments")
        .or_else(|| call.get("params"))
        .or_else(|| call.get("parameters"))
        .cloned()?;
    Some(crate::agent::tools::tool::ToolCall {
        name: beef::lean::Cow::owned(name),
        arguments,
    })
}

fn extract_from_tool_call_blocks(response: &str) -> Vec<crate::agent::tools::tool::ToolCall> {
    let mut tool_calls = Vec::new();
    let mut search_start = 0;
    while let Some(start) = response[search_start..].find(OPEN_TOOL_CALL) {
        let absolute_start = search_start + start;
        let Some(end) = response[absolute_start..].find(CLOSE_TOOL_CALL) else {
            break;
        };
        let absolute_end = absolute_start + end;
        let tool_json = &response[absolute_start + OPEN_TOOL_CALL.len()..absolute_end];
        if let Ok(tool_call) =
            serde_json::from_str::<crate::agent::tools::tool::ToolCall>(tool_json)
        {
            tool_calls.push(tool_call);
        }
        search_start = absolute_end + CLOSE_TOOL_CALL.len();
    }
    tool_calls
}

fn extract_from_perplexity_json(trimmed: &str) -> Vec<crate::agent::tools::tool::ToolCall> {
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Vec::new();
    }
    let Ok(json_value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return Vec::new();
    };
    if let Some(tool_call) = tool_call_from_perplexity_object(&json_value) {
        return vec![tool_call];
    }
    extract_perplexity_array(&json_value)
}

fn tool_call_from_perplexity_object(
    json_value: &serde_json::Value,
) -> Option<crate::agent::tools::tool::ToolCall> {
    let tool_name = json_value.get("tool").and_then(|v| v.as_str())?;
    let arguments = json_value.get("arguments")?;
    Some(crate::agent::tools::tool::ToolCall {
        name: beef::lean::Cow::owned(tool_name.to_string()),
        arguments: arguments.clone(),
    })
}

fn extract_perplexity_array(
    json_value: &serde_json::Value,
) -> Vec<crate::agent::tools::tool::ToolCall> {
    let Some(array) = json_value.as_array() else {
        return Vec::new();
    };
    let mut tool_calls = Vec::new();
    for item in array {
        if let Some(tool_call) = tool_call_from_perplexity_object(item) {
            tool_calls.push(tool_call);
        }
    }
    tool_calls
}

/// Convert extracted calls into wire `ToolCallInfo` values with synthetic ids.
#[must_use]
pub(crate) fn tool_calls_as_infos(
    calls: &[crate::agent::tools::tool::ToolCall],
) -> Vec<crate::messages::ToolCallInfo> {
    calls
        .iter()
        .enumerate()
        .map(|(idx, call)| {
            let arguments_json =
                serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string());
            crate::messages::ToolCallInfo::from_json(
                format!("call_{idx}"),
                call.name.as_ref(),
                arguments_json,
            )
        })
        .collect()
}

/// Parse standalone `name(key=value, ...)` lines in a response.
///
/// Models such as Llama 4 and Claude on OpenRouter often emit this instead
/// of native `tool_calls` or `<tool_call>` XML. Only a full line that is a
/// call is extracted, so prose that mentions `fs(...)` is not executed.
fn extract_function_style_tool_calls(trimmed: &str) -> Vec<crate::agent::tools::tool::ToolCall> {
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut calls: Vec<crate::agent::tools::tool::ToolCall> = Vec::new();
    for line in trimmed.lines() {
        let line = unwrap_standalone_call_line(line);
        if line.is_empty() {
            continue;
        }
        let Some(call) = parse_function_style_call(line) else {
            continue;
        };
        if calls
            .iter()
            .any(|existing| existing.name == call.name && existing.arguments == call.arguments)
        {
            continue;
        }
        calls.push(call);
    }
    calls
}

fn unwrap_standalone_call_line(line: &str) -> &str {
    let line = line.trim();
    line.strip_prefix('`')
        .and_then(|inner| inner.strip_suffix('`'))
        .map_or(line, str::trim)
}

fn parse_function_style_call(line: &str) -> Option<crate::agent::tools::tool::ToolCall> {
    let open = line.find('(')?;
    if !line.ends_with(')') {
        return None;
    }
    let name = line[..open].trim();
    if !is_tool_name(name) {
        return None;
    }
    let args_src = line.get(open.checked_add(1)?..line.len().checked_sub(1)?)?;
    let arguments = parse_keyword_arguments(args_src)?;
    Some(crate::agent::tools::tool::ToolCall {
        name: beef::lean::Cow::owned(name.to_string()),
        arguments,
    })
}

fn is_tool_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn parse_keyword_arguments(src: &str) -> Option<serde_json::Value> {
    let src = src.trim();
    if src.is_empty() {
        return None;
    }
    let parts = split_top_level_commas(src)?;
    let mut map = serde_json::Map::new();
    for part in parts {
        if part.is_empty() {
            return None;
        }
        let eq = part.find('=')?;
        let key = part[..eq].trim();
        if !is_argument_key(key) {
            return None;
        }
        let value = parse_argument_value(part[eq.checked_add(1)?..].trim())?;
        map.insert(key.to_string(), value);
    }
    if map.is_empty() {
        return None;
    }
    Some(serde_json::Value::Object(map))
}

fn is_argument_key(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn parse_argument_value(raw: &str) -> Option<serde_json::Value> {
    if raw.is_empty() {
        return None;
    }
    if (raw.starts_with('\'') && raw.ends_with('\''))
        || (raw.starts_with('"') && raw.ends_with('"'))
    {
        let inner = raw.get(1..raw.len().checked_sub(1)?)?;
        return Some(serde_json::Value::String(unescape_quoted(inner)));
    }
    match raw {
        "true" => return Some(serde_json::Value::Bool(true)),
        "false" => return Some(serde_json::Value::Bool(false)),
        "null" => return Some(serde_json::Value::Null),
        _ => {}
    }
    if let Ok(number) = raw.parse::<i64>() {
        return Some(serde_json::Value::Number(number.into()));
    }
    if let Ok(number) = raw.parse::<f64>()
        && let Some(json_number) = serde_json::Number::from_f64(number)
    {
        return Some(serde_json::Value::Number(json_number));
    }
    if raw
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '-' || ch == '/')
    {
        return Some(serde_json::Value::String(raw.to_string()));
    }
    None
}

fn unescape_quoted(inner: &str) -> String {
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some(escaped) => out.push(escaped),
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn split_top_level_commas(src: &str) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (idx, ch) in src.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote.is_some() && ch == '\\' {
            escaped = true;
            continue;
        }
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            continue;
        }
        if ch == ',' {
            parts.push(src[start..idx].trim());
            start = idx.checked_add(ch.len_utf8())?;
        }
    }
    if quote.is_some() {
        return None;
    }
    parts.push(src[start..].trim());
    Some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_function_style_fs_list_call() {
        let calls =
            extract_tool_calls_from_text("fs(op=list, path='.', max_depth=2, max_entries=10)");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_ref(), "fs");
        assert_eq!(calls[0].arguments["op"], "list");
        assert_eq!(calls[0].arguments["path"], ".");
        assert_eq!(calls[0].arguments["max_depth"], 2);
        assert_eq!(calls[0].arguments["max_entries"], 10);
    }

    #[test]
    fn extract_function_style_ignores_prose_that_mentions_a_call() {
        let calls =
            extract_tool_calls_from_text("Use fs(op=list, path='.') to inspect the workspace.");
        assert!(calls.is_empty());
    }

    #[test]
    fn extract_function_style_from_mixed_prose() {
        let response = "To answer your question, I'll need to explore the repo.\n\
First, I'll list the files and directories:\n\
fs(op=list, path='.', max_depth=3, max_entries=20)\n\
Once I have a sense of the repo's structure, I can start reading key files.\n\
I'll begin with the list command:\n\
fs(op=list, path='.', max_depth=3, max_entries=20)\n";
        let calls = extract_tool_calls_from_text(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_ref(), "fs");
        assert_eq!(calls[0].arguments["op"], "list");
        assert_eq!(calls[0].arguments["path"], ".");
        assert_eq!(calls[0].arguments["max_depth"], 3);
        assert_eq!(calls[0].arguments["max_entries"], 20);
    }

    #[test]
    fn extract_function_style_backtick_wrapped_line() {
        let calls = extract_tool_calls_from_text("`fs(op=read, path='README.md')`");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn extract_function_style_parallel_lines() {
        let calls =
            extract_tool_calls_from_text("fs(op=list, path='src')\nfs(op=read, path='README.md')");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["path"], "src");
        assert_eq!(calls[1].name.as_ref(), "fs");
        assert_eq!(calls[1].arguments["op"], "read");
    }

    #[test]
    fn xml_tool_call_still_wins() {
        let calls = extract_tool_calls_from_text(
            r#"<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_ref(), "bash");
    }

    #[test]
    fn tool_calls_as_infos_assigns_synthetic_ids() {
        let calls = extract_tool_calls_from_text("fs(op=list, path='.')");
        let infos = tool_calls_as_infos(&calls);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id.as_ref(), "call_0");
        assert_eq!(infos[0].name.as_ref(), "fs");
        assert_eq!(infos[0].arguments["op"], "list");
    }
}
