//! Redact credential-shaped text before it is logged or returned to a client.

const REDACTED: &str = "[redacted]";

/// Replace API keys, bearer tokens, and secret assignments.
#[must_use]
pub fn redact_secrets(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut index = 0;
    while index < input.len() {
        if let Some(end) = secret_span_end(input, index) {
            out.push_str(REDACTED);
            index = end;
            continue;
        }
        let Some(ch) = input[index..].chars().next() else {
            break;
        };
        out.push(ch);
        index = index.saturating_add(ch.len_utf8());
    }
    out
}

fn secret_span_end(input: &str, start: usize) -> Option<usize> {
    let rest = input.get(start..)?;
    if let Some(len) = sk_token_len(rest) {
        return Some(start.saturating_add(len));
    }
    if let Some(len) = bearer_token_len(rest) {
        return Some(start.saturating_add(len));
    }
    assignment_secret_len(rest).map(|len| start.saturating_add(len))
}

fn sk_token_len(rest: &str) -> Option<usize> {
    let body = rest.strip_prefix("sk-")?;
    let len = ascii_token_len(body);
    if len < 8 {
        return None;
    }
    Some("sk-".len().saturating_add(len))
}

fn bearer_token_len(rest: &str) -> Option<usize> {
    let body = rest.strip_prefix("Bearer ")?;
    let len = body
        .bytes()
        .take_while(|byte| !byte.is_ascii_whitespace())
        .count();
    if len < 8 {
        return None;
    }
    Some("Bearer ".len().saturating_add(len))
}

fn assignment_secret_len(rest: &str) -> Option<usize> {
    const KEYS: [&str; 4] = ["api_key", "token", "secret", "authorization"];
    let key_len = KEYS
        .into_iter()
        .find(|key| starts_with_ignore_ascii_case(rest, key))
        .map(str::len)?;
    let after_key = rest.get(key_len..)?.trim_start();
    let after_sep = after_key.strip_prefix(['=', ':'])?.trim_start();
    let skipped = rest.len().saturating_sub(after_sep.len());
    let unquoted = after_sep.trim_start_matches(['"', '\'']);
    let quotes = after_sep.len().saturating_sub(unquoted.len());
    let value_len = unquoted
        .chars()
        .take_while(|ch| {
            !ch.is_whitespace() && *ch != '"' && *ch != '\'' && *ch != ',' && *ch != '}'
        })
        .map(char::len_utf8)
        .sum::<usize>();
    if value_len == 0 {
        return None;
    }
    Some(skipped.saturating_add(quotes).saturating_add(value_len))
}

fn ascii_token_len(text: &str) -> usize {
    text.bytes()
        .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'-' || *byte == b'_')
        .count()
}

fn starts_with_ignore_ascii_case(text: &str, prefix: &str) -> bool {
    let Some(head) = text.get(..prefix.len()) else {
        return false;
    };
    head.eq_ignore_ascii_case(prefix)
}

#[cfg(test)]
mod tests {
    use super::redact_secrets;

    #[test]
    fn redacts_sk_bearer_and_assignments() {
        let raw = format!(
            "anthropic API error (400 Bad Request): bad {}-ant-api03-SUPERSECRETVALUE \
                   Bearer abcdefghijklmnop api_key=plain-secret token: \"another-secret\"",
            "sk"
        );
        let redacted = redact_secrets(&raw);
        assert!(!redacted.contains("SUPERSECRETVALUE"));
        assert!(!redacted.contains("abcdefghijklmnop"));
        assert!(!redacted.contains("plain-secret"));
        assert!(!redacted.contains("another-secret"));
        assert!(redacted.contains("[redacted]"));
        assert!(redacted.contains("prompt") || !raw.contains("prompt_tokens"));
        assert_eq!(redact_secrets("prompt_tokens: 12"), "prompt_tokens: 12");
    }
}
