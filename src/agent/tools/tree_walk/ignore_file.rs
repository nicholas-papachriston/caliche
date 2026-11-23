//! One ignore file, read as git reads it.
//!
//! The reader follows gitignore(5) and what the git binary does where the
//! page is silent: a UTF-8 byte order mark at the start is skipped, a
//! carriage return before each newline is dropped, a NUL ends the line, a
//! line starting with `#` is a comment, and trailing spaces are trimmed
//! unless escaped. Patterns are matched over bytes, so a line or a name
//! that is not UTF-8 is matched byte for byte. Within a file the last
//! matching pattern decides.
//!
//! Every ignore file is also read as the `ignore` crate reads it, since
//! the walks used that reading before: a path is hidden when either
//! reading hides it, and un-hidden only when both un-hide it. A file the
//! crate's reading hid, such as one a brace pattern names, stays hidden
//! though git lists it.

use std::path::{Path, PathBuf};

use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use super::ignore_pattern::Glob;

/// One pattern line.
#[derive(Debug, Clone)]
struct Pattern {
    glob: Glob,
    /// It starts with `!`: a match un-hides.
    negated: bool,
    /// It ends with `/`: it matches directories only.
    dir_only: bool,
    /// It has a `/` before its end: it matches the path from the file's
    /// directory, not the last name at any level.
    anchored: bool,
}

/// The patterns of one ignore file, relative to its directory.
#[derive(Debug, Clone)]
pub struct IgnoreFile {
    base: PathBuf,
    patterns: Vec<Pattern>,
    crate_reading: Option<Gitignore>,
}

impl IgnoreFile {
    /// Read the ignore file `bytes`, whose patterns are relative to `base`.
    #[must_use]
    pub fn parse(base: &Path, bytes: &[u8]) -> Self {
        const BOM: &[u8] = b"\xEF\xBB\xBF";
        let text = bytes.strip_prefix(BOM).unwrap_or(bytes);
        let patterns = text
            .split(|&byte| byte == b'\n')
            .filter_map(|line| {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                let line = line.split(|&byte| byte == 0).next().unwrap_or(line);
                if line.starts_with(b"#") {
                    return None;
                }
                pattern(trim_trailing_spaces(line))
            })
            .collect();
        Self {
            base: base.to_path_buf(),
            patterns,
            crate_reading: crate_reading(base, bytes),
        }
    }

    /// Whether the file hides `path` (`Some(true)`), un-hides it
    /// (`Some(false)`), or says nothing about it. A relative `path` is
    /// relative to the file's directory, a leading `./` dropped, as the
    /// `ignore` crate reads one. The git reading says nothing about a path
    /// outside that directory.
    #[must_use]
    pub fn decide(&self, path: &Path, is_dir: bool) -> Option<bool> {
        let git = relative_bytes(&self.base, path).and_then(|rel| self.git_decision(&rel, is_dir));
        let Some(reading) = &self.crate_reading else {
            return git;
        };
        let by_crate = match reading.matched(path, is_dir) {
            Match::Ignore(_) => Some(true),
            Match::Whitelist(_) => Some(false),
            Match::None => None,
        };
        either_hides(git, by_crate)
    }

    fn git_decision(&self, rel: &[u8], is_dir: bool) -> Option<bool> {
        let name = rel.rsplit(|&byte| byte == b'/').next().unwrap_or(rel);
        self.patterns
            .iter()
            .rev()
            .filter(|pattern| is_dir || !pattern.dir_only)
            .find(|pattern| {
                let text = if pattern.anchored { rel } else { name };
                pattern.glob.matches(text)
            })
            .map(|pattern| !pattern.negated)
    }
}

/// Two readings of one file: hidden when either hides, un-hidden only when
/// both un-hide, and otherwise nothing.
fn either_hides(one: Option<bool>, other: Option<bool>) -> Option<bool> {
    match (one, other) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}

/// `line` without its trailing spaces; a space after a backslash stays.
fn trim_trailing_spaces(line: &[u8]) -> &[u8] {
    let mut end = 0;
    let mut at = 0;
    while let Some(&byte) = line.get(at) {
        match byte {
            b'\\' if at + 1 < line.len() => {
                at += 2;
                end = at;
            }
            b' ' => at += 1,
            _ => {
                at += 1;
                end = at;
            }
        }
    }
    &line[..end]
}

/// The pattern on `line`, or `None` for a blank line or one that can match
/// nothing.
fn pattern(line: &[u8]) -> Option<Pattern> {
    let (negated, body) = match line.strip_prefix(b"!") {
        Some(rest) => (true, rest),
        None => (false, line),
    };
    let (dir_only, body) = match body.strip_suffix(b"/") {
        Some(rest) => (true, rest),
        None => (false, body),
    };
    let anchored = body.contains(&b'/');
    let body = body.strip_prefix(b"/").unwrap_or(body);
    Some(Pattern {
        glob: Glob::compile(body)?,
        negated,
        dir_only,
        anchored,
    })
}

/// The file as the `ignore` crate's own file reader reads it: line by
/// line as UTF-8 until a line that is not, with a byte order mark dropped
/// from the first line.
fn crate_reading(base: &Path, bytes: &[u8]) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(base);
    for (index, line) in bytes.split(|&byte| byte == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Ok(line) = std::str::from_utf8(line) else {
            break;
        };
        let line = if index == 0 {
            line.trim_start_matches('\u{feff}')
        } else {
            line
        };
        // A line the crate cannot parse adds nothing, as in its reader.
        let _ = builder.add_line(None, line);
    }
    builder.build().ok()
}

/// `path` below `base` as `/`-separated bytes: a relative `path` is taken
/// as below `base`, its `.` parts dropped. `None` when it is not below
/// `base` or climbs with `..`.
fn relative_bytes(base: &Path, path: &Path) -> Option<Vec<u8>> {
    use std::path::Component;
    let rel = if path.is_absolute() {
        path.strip_prefix(base).ok()?
    } else {
        path
    };
    let mut bytes = Vec::new();
    for part in rel.components() {
        match part {
            Component::Normal(name) => {
                if !bytes.is_empty() {
                    bytes.push(b'/');
                }
                bytes.extend_from_slice(name.as_encoded_bytes());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!bytes.is_empty()).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::IgnoreFile;

    /// A file of twenty thousand patterns is read and decides paths in
    /// time linear in its size, with no pattern's cost growing with the
    /// others.
    #[test]
    fn a_large_ignore_file_stays_linear() {
        let mut bytes = Vec::new();
        for index in 0..20_000 {
            bytes.extend(format!("d{index}/[[:alpha:]]*x?y**\n").as_bytes());
        }
        bytes.extend(b"*.secret\n");
        let start = std::time::Instant::now();
        let file = IgnoreFile::parse(Path::new("/r"), &bytes);
        let decided = ["/r/a.secret", "/r/d7/ok.rs", "/r/d7/axzyq"]
            .map(|path| file.decide(Path::new(path), false));
        assert_eq!(decided, [Some(true), None, Some(true)]);
        assert!(start.elapsed() < std::time::Duration::from_secs(10));
    }

    /// A path is read below the file's directory: an absolute one when it
    /// is there, a relative one with its `.` parts dropped. The git
    /// reading says nothing about a path outside or climbing out; the
    /// crate reading still decides those, as the crate did. A negation
    /// un-hides only when both readings un-hide, so the `keep.secret` rows
    /// show the git reading reads each relative form.
    #[test]
    fn paths_are_read_below_the_file_directory() {
        let base = Path::new("/r");
        let file = IgnoreFile::parse(base, b"*.secret\n!keep.secret\n");
        let decide = |path: &str| file.decide(Path::new(path), false);
        assert_eq!(
            [
                "/r/a.secret",
                "/r/d/a.secret",
                "a.secret",
                "./a.secret",
                "d/./a.secret",
                "../a.secret",
                "/elsewhere/a.secret",
                "/r",
                "/r/ok.rs",
                "/r/keep.secret",
                "keep.secret",
                "./keep.secret",
                "d/./keep.secret",
            ]
            .map(decide),
            [
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                None,
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(false),
            ]
        );
    }

    /// As git reads a bracket expression, it never matches a slash, so
    /// `!a[/]b` does not un-hide `a/b`, though the crate reading's does.
    #[test]
    fn a_bracket_expression_never_matches_a_slash() {
        let file = IgnoreFile::parse(Path::new("/r"), b"*\n!a[/]b\n");
        assert_eq!(file.decide(Path::new("/r/a/b"), false), Some(true));
    }
}
