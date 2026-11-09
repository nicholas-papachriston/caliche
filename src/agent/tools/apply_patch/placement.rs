//! Indent-aware placement for `apply_patch` hunks.

/// Leading spaces/tabs of a line.
#[must_use]
pub(super) fn leading_ws(s: &str) -> &str {
    let rest_len = s.trim_start_matches([' ', '\t']).len();
    &s[..s.len() - rest_len]
}

/// Shift `new_lines` by the indent delta between a file line and a pattern line.
#[must_use]
pub(super) fn reindent_to_match(
    file_line: &str,
    pattern_line: &str,
    new_lines: &[String],
) -> Vec<String> {
    let file_ws = leading_ws(file_line);
    let pat_ws = leading_ws(pattern_line);
    if file_ws == pat_ws {
        return new_lines.to_vec();
    }
    if let Some(extra) = file_ws.strip_prefix(pat_ws) {
        return new_lines
            .iter()
            .map(|line| {
                if line.is_empty() {
                    line.clone()
                } else {
                    let mut out = String::with_capacity(extra.len() + line.len());
                    out.push_str(extra);
                    out.push_str(line);
                    out
                }
            })
            .collect();
    }
    if let Some(extra) = pat_ws.strip_prefix(file_ws) {
        return new_lines
            .iter()
            .map(|line| {
                line.strip_prefix(extra)
                    .map(str::to_string)
                    .unwrap_or_else(|| line.clone())
            })
            .collect();
    }
    new_lines.to_vec()
}

/// Original-file index for a `+`-only hunk.
///
/// Context (`@@`) advances `line_index` to the line after the match. Insert
/// there so new code lands next to that context instead of at EOF. `*** End of
/// File` still appends.
#[must_use]
pub(super) const fn insertion_index(
    original_len: usize,
    line_index: usize,
    is_end_of_file: bool,
) -> usize {
    if is_end_of_file {
        original_len
    } else if line_index > original_len {
        original_len
    } else {
        line_index
    }
}

/// Match insert-only hunk indent to the line being inserted before, else the previous line.
#[must_use]
pub(super) fn reindent_insert(
    original_lines: &[String],
    insertion_idx: usize,
    new_lines: &[String],
) -> Vec<String> {
    let Some(first_new) = new_lines.first() else {
        return Vec::new();
    };
    let neighbor = original_lines.get(insertion_idx).or_else(|| {
        insertion_idx
            .checked_sub(1)
            .and_then(|idx| original_lines.get(idx))
    });
    let Some(neighbor) = neighbor else {
        return new_lines.to_vec();
    };
    reindent_to_match(neighbor, first_new, new_lines)
}

/// Lift new lines that sit left of the matched file indent (forgotten sibling indent).
#[must_use]
pub(super) fn lift_underindented_lines(file_line: &str, new_lines: &[String]) -> Vec<String> {
    let file_ws = leading_ws(file_line);
    new_lines
        .iter()
        .map(|line| {
            if line.is_empty() {
                return line.clone();
            }
            let line_ws = leading_ws(line);
            if file_ws.starts_with(line_ws) && file_ws != line_ws {
                let rest = line.trim_start_matches([' ', '\t']);
                let mut out = String::with_capacity(file_ws.len() + rest.len());
                out.push_str(file_ws);
                out.push_str(rest);
                out
            } else {
                line.clone()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_index_uses_context_not_eof() {
        assert_eq!(insertion_index(10, 3, false), 3);
        assert_eq!(insertion_index(10, 3, true), 10);
        assert_eq!(insertion_index(10, 99, false), 10);
    }

    #[test]
    fn reindent_prefixes_under_indented_new_lines() {
        let out = reindent_to_match("    foo()", "foo()", &["bar()".to_string()]);
        assert_eq!(out, vec!["    bar()".to_string()]);
    }

    #[test]
    fn reindent_strips_over_indented_new_lines() {
        let out = reindent_to_match("foo()", "    foo()", &["    bar()".to_string()]);
        assert_eq!(out, vec!["bar()".to_string()]);
    }

    #[test]
    fn reindent_insert_matches_line_being_inserted_before() {
        let lines = vec!["impl Foo {".to_string(), "    fn a() {}".to_string()];
        let out = reindent_insert(&lines, 1, &["fn b() {}".to_string()]);
        assert_eq!(out, vec!["    fn b() {}".to_string()]);
    }

    #[test]
    fn lift_underindented_sibling_keeps_deeper_body() {
        let out = lift_underindented_lines(
            "    fn a() {}",
            &[
                "    fn a() {}".to_string(),
                "fn b() {}".to_string(),
                "        let x = 1;".to_string(),
            ],
        );
        assert_eq!(
            out,
            vec![
                "    fn a() {}".to_string(),
                "    fn b() {}".to_string(),
                "        let x = 1;".to_string(),
            ]
        );
    }
}
