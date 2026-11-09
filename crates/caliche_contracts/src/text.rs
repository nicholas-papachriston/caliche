//! Text helpers shared by the Caliche crates.

/// Returns the longest prefix of `text` that is at most `max_bytes` bytes long and
/// ends on a char boundary. Text that fits is returned whole.
///
/// Slicing with `&text[..max_bytes]` panics when `max_bytes` falls inside a
/// multi-byte character; this cuts before that character instead.
#[must_use]
pub fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    &text[..text.floor_char_boundary(max_bytes)]
}

/// Finds the JSON object or array in a model reply.
///
/// The search region is the body of the first ```` ``` ```` fence (its language
/// tag dropped, the closing fence optional), or the whole text when there is no
/// fence. Returns the first balanced object or array in that region, starting at
/// whichever of `{` or `[` comes first. Brackets inside JSON strings do not
/// count. Returns `None` when there is no opener or the value is not closed.
#[must_use]
pub fn extract_json(text: &str) -> Option<&str> {
    let region = match text.find("```") {
        Some(open) => {
            let body = text[open + 3..]
                .trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-');
            body.find("```").map_or(body, |close| &body[..close])
        }
        None => text,
    };
    let start = region.find(['{', '['])?;
    let value = &region[start..];
    let (open, close) = if value.starts_with('{') {
        ('{', '}')
    } else {
        ('[', ']')
    };
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, ch) in value.char_indices() {
        if in_string {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
        } else if ch == '"' {
            in_string = true;
        } else if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return value.get(..=i);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{extract_json, truncate_utf8};

    #[test]
    fn ascii_is_cut_at_the_limit() {
        assert_eq!(truncate_utf8("abcdef", 4), "abcd");
    }

    #[test]
    fn a_two_byte_character_across_the_limit_is_dropped() {
        // "é" is bytes 2..4, so a limit of 3 falls inside it.
        assert_eq!(truncate_utf8("abé", 3), "ab");
        assert_eq!(truncate_utf8("abé", 4), "abé");
    }

    #[test]
    fn a_three_byte_character_across_the_limit_is_dropped() {
        // "東" is bytes 1..4.
        assert_eq!(truncate_utf8("a東京", 2), "a");
        assert_eq!(truncate_utf8("a東京", 3), "a");
        assert_eq!(truncate_utf8("a東京", 4), "a東");
    }

    #[test]
    fn a_four_byte_character_across_the_limit_is_dropped() {
        // "🚀" is bytes 1..5.
        assert_eq!(truncate_utf8("a🚀b", 4), "a");
        assert_eq!(truncate_utf8("a🚀b", 5), "a🚀");
    }

    #[test]
    fn a_zero_limit_gives_an_empty_string() {
        assert_eq!(truncate_utf8("é", 0), "");
        assert_eq!(truncate_utf8("abc", 0), "");
    }

    #[test]
    fn text_within_the_limit_is_returned_whole() {
        assert_eq!(truncate_utf8("ab東", 5), "ab東");
        assert_eq!(truncate_utf8("ab東", 100), "ab東");
        assert_eq!(truncate_utf8("", 3), "");
    }

    #[test]
    fn extract_json_reads_fences() {
        assert_eq!(extract_json("```json\n{\"a\": 1}\n```"), Some("{\"a\": 1}"));
        assert_eq!(extract_json("```JSON\n{\"a\": 1}\n```"), Some("{\"a\": 1}"));
        assert_eq!(extract_json("```\n{\"a\": 1}\n```"), Some("{\"a\": 1}"));
        assert_eq!(extract_json("```json\n{\"a\": 1}"), Some("{\"a\": 1}"));
        assert_eq!(
            extract_json("Here it is:\n```json\n{\"a\": 1}\n```\nDone."),
            Some("{\"a\": 1}")
        );
        assert_eq!(
            extract_json("```json\n{\"a\": 1}\n```\nand\n```json\n{\"b\": 2}\n```"),
            Some("{\"a\": 1}")
        );
        assert_eq!(
            extract_json("Résumé:\n```json\n{\"a\": 1}\n```"),
            Some("{\"a\": 1}")
        );
    }

    #[test]
    fn extract_json_returns_the_outermost_value() {
        assert_eq!(extract_json("{\"a\": 1}"), Some("{\"a\": 1}"));
        assert_eq!(extract_json("```json\n[1, 2]\n```"), Some("[1, 2]"));
        assert_eq!(
            extract_json("```json\n{\"a\": [1, [2, 3]]}\n```"),
            Some("{\"a\": [1, [2, 3]]}")
        );
        assert_eq!(
            extract_json("```json\n[{\"a\": 1}, {\"b\": 2}]\n```"),
            Some("[{\"a\": 1}, {\"b\": 2}]")
        );
        assert_eq!(
            extract_json("Items: [{\"a\": 1}] end"),
            Some("[{\"a\": 1}]")
        );
        assert_eq!(extract_json("Note] {\"a\": 1}"), Some("{\"a\": 1}"));
        assert_eq!(extract_json("Résumé: {\"a\": 1}"), Some("{\"a\": 1}"));
    }

    #[test]
    fn extract_json_skips_brackets_in_strings() {
        assert_eq!(
            extract_json(r#"x {"a": "}{\"]", "b": [1]} y"#),
            Some(r#"{"a": "}{\"]", "b": [1]}"#)
        );
    }

    #[test]
    fn extract_json_without_a_closed_value_is_none() {
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json("```] then {"), None);
        assert_eq!(extract_json("{\"a\": [1, 2]"), None);
        assert_eq!(extract_json(""), None);
    }
}
