//! Scan edited Rust library sources for panic paths the coding contract forbids.

use std::path::{Path, PathBuf};

use super::VerificationRun;

const SCAN_COMMAND: &str = "panic-path scan";

/// Findings from a library-code panic-path scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicPathHit {
    pub path: PathBuf,
    pub line: usize,
    pub kind: &'static str,
}

/// Scan changed Rust files and return a failed verification run when library
/// code uses unwrap/expect/panic/todo/unimplemented/unreachable.
#[must_use]
pub fn panic_path_verification(root: &Path, changed_files: &[PathBuf]) -> Option<VerificationRun> {
    let hits = scan_changed_rust_files(root, changed_files);
    if hits.is_empty() {
        return None;
    }
    let mut stderr = String::from(
        "Library code must not use unwrap, expect, panic, todo, unimplemented, or unreachable.\n",
    );
    for hit in &hits {
        let display = hit.path.display();
        let _ = std::fmt::Write::write_fmt(
            &mut stderr,
            format_args!("{display}:{}: {}\n", hit.line, hit.kind),
        );
    }
    Some(VerificationRun {
        command: SCAN_COMMAND.to_string(),
        exit_code: Some(1),
        stdout_tail: String::new(),
        stderr_tail: stderr,
        duration_ms: 0,
    })
}

fn scan_changed_rust_files(root: &Path, changed_files: &[PathBuf]) -> Vec<PanicPathHit> {
    let mut hits = Vec::new();
    for path in changed_files {
        if !is_library_rust_file(path) {
            continue;
        }
        let abs = if path.is_absolute() {
            path.clone()
        } else {
            root.join(path)
        };
        let Ok(source) = std::fs::read_to_string(&abs) else {
            continue;
        };
        for (line, kind) in rust_panic_paths(&source) {
            hits.push(PanicPathHit {
                path: path.clone(),
                line,
                kind,
            });
        }
    }
    hits
}

fn is_library_rust_file(path: &Path) -> bool {
    if path.extension().is_none_or(|ext| ext != "rs") {
        return false;
    }
    if path
        .components()
        .any(|component| component.as_os_str() == "tests")
    {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if name == "tests.rs" || name.ends_with("_test.rs") || name.ends_with("_tests.rs") {
        return false;
    }
    true
}

/// Return 1-based line numbers of forbidden panic paths outside `#[cfg(test)]`.
#[must_use]
pub fn rust_panic_paths(source: &str) -> Vec<(usize, &'static str)> {
    let mut hits = Vec::new();
    let mut skip_depth = 0i32;
    let mut pending_test_mod = false;

    for (idx, raw) in source.lines().enumerate() {
        let line = strip_line_comment(raw);
        if skip_depth == 0 && line.contains("#[cfg(test)]") {
            pending_test_mod = true;
        }

        let opens = count_char(line, '{');
        let closes = count_char(line, '}');

        if skip_depth > 0 {
            skip_depth = skip_depth.saturating_add(opens).saturating_sub(closes);
            continue;
        }

        if pending_test_mod && opens > 0 {
            skip_depth = opens.saturating_sub(closes);
            pending_test_mod = false;
            continue;
        }

        if let Some(kind) = panic_kind(line) {
            hits.push((idx.saturating_add(1), kind));
        }
    }
    hits
}

fn strip_line_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_string = !in_string;
        } else if !in_string && b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            return line.get(..i).unwrap_or(line);
        }
        i = i.saturating_add(1);
    }
    line
}

fn count_char(line: &str, needle: char) -> i32 {
    i32::try_from(line.chars().filter(|ch| *ch == needle).count()).unwrap_or(i32::MAX)
}

fn panic_kind(line: &str) -> Option<&'static str> {
    if line.contains(".unwrap()") || line.contains(".unwrap_err()") {
        return Some("unwrap");
    }
    if line.contains(".expect(") {
        return Some("expect");
    }
    if line.contains("panic!(") {
        return Some("panic");
    }
    if line.contains("todo!(") {
        return Some("todo");
    }
    if line.contains("unimplemented!(") {
        return Some("unimplemented");
    }
    if line.contains("unreachable!(") {
        return Some("unreachable");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_library_unwrap_but_not_unwrap_or() {
        let src = "pub fn f() {\n    let _ = None::<u8>.unwrap();\n    let _ = None::<u8>.unwrap_or(0);\n}\n";
        assert_eq!(rust_panic_paths(src), vec![(2, "unwrap")]);
    }

    #[test]
    fn ignores_test_modules_and_comments() {
        let src = r#"
pub fn ok() {}
// let _ = x.unwrap();
#[cfg(test)]
mod tests {
    #[test]
    fn t() {
        None::<u8>.unwrap();
        panic!("nope");
    }
}
"#;
        assert!(rust_panic_paths(src).is_empty());
    }

    #[test]
    fn flags_unreachable_in_library_code() {
        let src = "pub fn f() -> i32 {\n    unreachable!()\n}\n";
        assert_eq!(rust_panic_paths(src), vec![(2, "unreachable")]);
    }

    #[test]
    fn skips_integration_test_paths() {
        assert!(!is_library_rust_file(Path::new("tests/foo.rs")));
        assert!(is_library_rust_file(Path::new("src/parse.rs")));
        assert!(!is_library_rust_file(Path::new("src/foo_test.rs")));
    }
}
