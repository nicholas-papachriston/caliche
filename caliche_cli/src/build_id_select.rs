//! Compile-time build id selection.
//!
//! `build.rs` includes this file. The library compiles it only for tests.
//! The id is tip literacy for a PATH binary. It is not a security property.

/// Choose the id embedded as `build_id`.
///
/// `explicit` is `CALICHE_BUILD_ID` when a packager set it.
/// `superproject` is the caliche checkout when this crate is a submodule,
/// as `(full sha, dirty worktree)`. `crate_head` is this crate's `HEAD`.
///
/// A 40-hex sha is kept. Anything else is skipped. Dirty checkouts get a
/// `-dirty` suffix so they are not mistaken for a clean tip.
#[must_use]
pub fn select_build_id(
    explicit: Option<&str>,
    superproject: Option<(&str, bool)>,
    crate_head: Option<(&str, bool)>,
) -> String {
    if let Some(id) = explicit.and_then(normalize_build_id) {
        return id;
    }
    if let Some(id) = git_candidate(superproject) {
        return id;
    }
    if let Some(id) = git_candidate(crate_head) {
        return id;
    }
    "unknown".to_owned()
}

fn git_candidate(candidate: Option<(&str, bool)>) -> Option<String> {
    candidate.and_then(|(sha, dirty)| format_git_id(sha, dirty))
}

fn normalize_build_id(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    match trimmed.strip_suffix("-dirty") {
        Some(hex) => format_git_id(hex, true),
        None => format_git_id(trimmed, false),
    }
}

fn format_git_id(hex: &str, dirty: bool) -> Option<String> {
    if hex.len() != 40 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = hex.to_ascii_lowercase();
    if dirty {
        out.push_str("-dirty");
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::select_build_id;

    const SUPER_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CRATE_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const EXPLICIT_SHA: &str = "cccccccccccccccccccccccccccccccccccccccc";

    #[test]
    fn explicit_sha_wins_over_git() {
        let id = select_build_id(
            Some(EXPLICIT_SHA),
            Some((SUPER_SHA, false)),
            Some((CRATE_SHA, true)),
        );
        assert_eq!(id, EXPLICIT_SHA);
    }

    #[test]
    fn explicit_dirty_suffix_is_preserved() {
        let raw = format!("{EXPLICIT_SHA}-dirty");
        let id = select_build_id(Some(&raw), Some((SUPER_SHA, false)), None);
        assert_eq!(id, format!("{EXPLICIT_SHA}-dirty"));
    }

    #[test]
    fn invalid_explicit_falls_through_to_superproject() {
        let id = select_build_id(
            Some("not-a-sha"),
            Some((SUPER_SHA, false)),
            Some((CRATE_SHA, false)),
        );
        assert_eq!(id, SUPER_SHA);
    }

    #[test]
    fn superproject_is_preferred_over_crate_head() {
        let id = select_build_id(None, Some((SUPER_SHA, false)), Some((CRATE_SHA, false)));
        assert_eq!(id, SUPER_SHA);
    }

    #[test]
    fn dirty_superproject_is_marked() {
        let id = select_build_id(None, Some((SUPER_SHA, true)), Some((CRATE_SHA, false)));
        assert_eq!(id, format!("{SUPER_SHA}-dirty"));
    }

    #[test]
    fn crate_head_is_used_when_there_is_no_superproject() {
        let id = select_build_id(None, None, Some((CRATE_SHA, false)));
        assert_eq!(id, CRATE_SHA);
    }

    #[test]
    fn missing_git_is_unknown() {
        let id = select_build_id(None, None, None);
        assert_eq!(id, "unknown");
    }

    #[test]
    fn short_sha_is_rejected() {
        let id = select_build_id(Some("abc123"), None, None);
        assert_eq!(id, "unknown");
    }

    #[test]
    fn uppercase_hex_is_normalized() {
        let upper = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let id = select_build_id(Some(upper), None, None);
        assert_eq!(id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    }
}
