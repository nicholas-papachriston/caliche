//! Lightweight code validators for fs post-write checks.
//! Catches common syntax issues and typos before the agent continues.

mod lua;

use std::path::Path;

/// Result of validation: Ok(()) or list of issue messages.
pub type ValidationResult = Result<(), Vec<String>>;

/// Validates source code for a given extension.
pub fn validate(path: &Path, content: &str) -> ValidationResult {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase);
    match ext.as_deref() {
        Some("lua") => lua::validate(content),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_validate_unknown_extension() {
        assert!(validate(Path::new("foo.txt"), "hello").is_ok());
    }
}
