//! Line-window parsing for `fs op=read`.
//!
//! `start_line`/`end_line` and `offset`/`limit` select a slice. Unparseable
//! window arguments are errors. They must not fall through to a from-the-start
//! read.

use anyhow::{Result, bail};
use serde_json::Value;

use super::common::params::coerce_u64;

/// Inclusive 1-indexed line window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadWindow {
    /// First line to return (1-indexed).
    pub start_line: usize,
    /// Last line to return, inclusive. `None` means through end of file.
    pub end_line: Option<usize>,
}

/// Resolve a read window from tool arguments.
///
/// `offset` is a 1-indexed alias of `start_line` for reads (`0` means the
/// start of the file). `limit` is the number of lines. Numeric strings and
/// integral floats are accepted. A present but unparseable field is an error.
pub fn resolve_read_window(args: &Value) -> Result<Option<ReadWindow>> {
    let start = optional_line(args, "start_line")?;
    let end = optional_line(args, "end_line")?;
    let offset = optional_line(args, "offset")?;
    let limit = optional_line(args, "limit")?;

    if start.is_none() && end.is_none() && limit.is_none() && matches!(offset, None | Some(0)) {
        return Ok(None);
    }

    if let (Some(start_line), Some(offset_line)) = (start, offset)
        && offset_line != 0
        && start_line != offset_line
    {
        bail!(
            "start_line ({start_line}) and offset ({offset_line}) disagree. \
             For fs op=read, offset is a 1-indexed alias of start_line. Pass one of them."
        );
    }

    let start_line = start
        .or_else(|| offset.filter(|line| *line > 0))
        .unwrap_or(1)
        .max(1);
    let end_line = match (end, limit) {
        (Some(end_line), _) if end_line < start_line => {
            bail!(
                "end_line ({end_line}) is before start line {start_line}. \
                 Lines are 1-indexed and inclusive, or use offset plus limit."
            );
        }
        (Some(end_line), _) => Some(end_line),
        (None, Some(0)) => {
            bail!("limit must be >= 1 (the number of lines to return).");
        }
        (None, Some(limit)) => Some(start_line.saturating_add(limit.saturating_sub(1))),
        (None, None) => None,
    };

    Ok(Some(ReadWindow {
        start_line,
        end_line,
    }))
}

fn optional_line(args: &Value, key: &str) -> Result<Option<usize>> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(number) = coerce_u64(value) else {
        bail!(
            "fs read {key} must be a 1-indexed line number, got {value}. \
             Example: start_line=120 end_line=160, or offset=120 limit=40. \
             Numeric strings such as \"120\" are accepted."
        );
    };
    let line = usize::try_from(number).map_err(|_| {
        anyhow::anyhow!("fs read {key} is too large ({number}). Use a 1-indexed line number.")
    })?;
    Ok(Some(line))
}

#[cfg(test)]
mod tests {
    use super::resolve_read_window;
    use serde_json::json;

    #[test]
    fn absent_window_reads_from_the_start() {
        let window = resolve_read_window(&json!({"path": "a.rs"})).expect("parse");
        assert!(window.is_none());
    }

    #[test]
    fn offset_zero_without_limit_is_not_a_window() {
        let window = resolve_read_window(&json!({"offset": 0})).expect("parse");
        assert!(window.is_none());
    }

    #[test]
    fn offset_and_limit_select_a_slice() {
        let window = resolve_read_window(&json!({"offset": 10, "limit": 3}))
            .expect("parse")
            .expect("window");
        assert_eq!(window.start_line, 10);
        assert_eq!(window.end_line, Some(12));
    }

    #[test]
    fn string_line_numbers_are_accepted() {
        let window = resolve_read_window(&json!({"start_line": "5", "end_line": "6"}))
            .expect("parse")
            .expect("window");
        assert_eq!(window.start_line, 5);
        assert_eq!(window.end_line, Some(6));
    }

    #[test]
    fn unparseable_start_line_is_an_error() {
        let error = resolve_read_window(&json!({"start_line": "head"})).expect_err("bad");
        let text = error.to_string();
        assert!(text.contains("start_line"), "{text}");
        assert!(text.contains("head"), "{text}");
    }
}
