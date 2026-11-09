//! Compile-time identity printed by `doctor` and `--version`.
//!
//! `BUILD_ID` compares a PATH binary with the tip that produced it.
//! It is not a security property.

/// Git commit embedded at compile time, or `unknown` when git was absent.
///
/// Resolution order is `CALICHE_BUILD_ID`, then the superproject `HEAD`
/// when this crate is a submodule (the caliche checkout), then this
/// crate's `HEAD`. A dirty worktree appends `-dirty`.
pub const BUILD_ID: &str = env!("CALICHE_BUILD_ID");

/// Text clap prints for `--version`: package version, then `build_id=`.
pub const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\nbuild_id=",
    env!("CALICHE_BUILD_ID"),
);

#[cfg(test)]
#[path = "build_id_select.rs"]
mod build_id_select;

#[cfg(test)]
mod tests {
    use super::{BUILD_ID, LONG_VERSION};

    fn is_full_sha(value: &str) -> bool {
        value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    #[test]
    fn build_id_is_sha_unknown_or_dirty_sha() {
        let ok = BUILD_ID == "unknown"
            || is_full_sha(BUILD_ID)
            || BUILD_ID.strip_suffix("-dirty").is_some_and(is_full_sha);
        assert!(ok, "build_id={BUILD_ID}");
    }

    #[test]
    fn long_version_has_its_own_build_id_line() {
        let mut lines = LONG_VERSION.lines();
        assert_eq!(lines.next(), Some(env!("CARGO_PKG_VERSION")));
        let build_line = format!("build_id={BUILD_ID}");
        assert_eq!(lines.next(), Some(build_line.as_str()));
        assert_eq!(lines.next(), None);
    }
}
