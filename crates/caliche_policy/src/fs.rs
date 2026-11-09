//! Filesystem path resolver.
//!
//! One canonical form for absolute paths, relative paths, and `..`. A symlink
//! that exists is replaced by its real path. User text does not add grants.
//! An alternate data stream, a device path, and an 8.3 short name are refused.

use crate::action::{Action, PathPlace, ResolvedPath, ToolFacts};
use serde_json::Value;
use std::path::{Component, Path};

/// Why [`resolve_fs_path`] refused a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsFault {
    /// Empty or blank path.
    Empty,
    /// A path component contains NUL.
    Null,
    /// `name:stream` alternate data stream.
    AlternateDataStream,
    /// `\\.\` device namespace.
    DevicePath,
    /// 8.3 short name such as `PROGRA~1`.
    ShortName,
    /// A `~/` path and no home directory.
    HomeUnknown,
}

/// Working directory and workspace used to place a path.
#[derive(Clone, Copy, Debug)]
pub struct FsRoots<'a> {
    /// Directory relative paths join.
    pub cwd: &'a str,
    /// Workspace root. Empty places every path as [`PathPlace::NoWorkspace`],
    /// which the gate denies.
    pub workspace: &'a str,
}

impl<'a> FsRoots<'a> {
    /// Roots for one tool call.
    #[must_use]
    pub const fn new(cwd: &'a str, workspace: &'a str) -> Self {
        Self { cwd, workspace }
    }
}

/// Resolve one tool path against [`FsRoots`].
///
/// The result uses `/` separators. A missing workspace places a path that is
/// otherwise valid as [`PathPlace::NoWorkspace`].
#[must_use]
pub fn resolve_fs_path(raw: &str, roots: FsRoots<'_>) -> Result<ResolvedPath, FsFault> {
    let home = std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok());
    resolve_with_home(raw, roots, home.as_deref())
}

/// Resolve using an explicit home directory. Tests pass the home they need.
#[must_use]
pub fn resolve_with_home(
    raw: &str,
    roots: FsRoots<'_>,
    home: Option<&str>,
) -> Result<ResolvedPath, FsFault> {
    let lexical = lexical_path(raw, roots.cwd, home)?;
    let canonical = realpath_best_effort(&lexical);
    let place = place_path(&canonical, roots.workspace);
    Ok(ResolvedPath { canonical, place })
}

/// `apply_patch` header that adds a file.
pub const ADD_FILE_MARKER: &str = "*** Add File: ";
/// `apply_patch` header that deletes a file.
pub const DELETE_FILE_MARKER: &str = "*** Delete File: ";
/// `apply_patch` header that updates a file.
pub const UPDATE_FILE_MARKER: &str = "*** Update File: ";
/// `apply_patch` header that moves an updated file to a new path.
pub const MOVE_TO_MARKER: &str = "*** Move to: ";

/// Every `apply_patch` header that names a path the tool writes or removes.
/// The tool's parser imports these markers, so the two cannot drift apart.
pub const PATCH_PATH_MARKERS: [&str; 4] = [
    ADD_FILE_MARKER,
    UPDATE_FILE_MARKER,
    DELETE_FILE_MARKER,
    MOVE_TO_MARKER,
];

/// Paths named by an `apply_patch` body: every [`PATCH_PATH_MARKERS`] header.
#[must_use]
pub fn patch_header_paths(text: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        for marker in PATCH_PATH_MARKERS {
            if let Some(rest) = trimmed.strip_prefix(marker.trim_end()) {
                let path = rest.trim();
                if !path.is_empty() {
                    paths.push(path.to_owned());
                }
            }
        }
    }
    paths
}

/// Replace filesystem and patch paths with resolver output.
///
/// With no workspace root, a filesystem action that names no path still gets
/// one [`PathPlace::NoWorkspace`] entry, so the gate denies it.
pub fn annotate_fs_action(action: &mut Action, roots: FsRoots<'_>) {
    let Action::Tool { facts, .. } = action else {
        return;
    };
    let Some(paths) = fs_paths_mut(facts) else {
        return;
    };
    if paths.is_empty() && roots.workspace.trim().is_empty() {
        paths.push(ResolvedPath {
            canonical: roots.cwd.to_owned(),
            place: PathPlace::NoWorkspace,
        });
        return;
    }
    for path in paths {
        let raw = std::mem::take(&mut path.canonical);
        *path = match resolve_fs_path(&raw, roots) {
            Ok(resolved) => resolved,
            Err(_) => ResolvedPath::rejected(raw),
        };
    }
}

/// Raw path strings from tool arguments, including patch headers.
#[must_use]
pub fn raw_paths_from_args(args: &Value) -> Vec<String> {
    let mut paths = Vec::new();
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        paths.push(path.to_owned());
    }
    if let Some(list) = args.get("paths").and_then(Value::as_array) {
        for item in list {
            if let Some(path) = item.as_str() {
                paths.push(path.to_owned());
            }
        }
    }
    for key in ["patch", "input", "diff"] {
        if let Some(text) = args.get(key).and_then(Value::as_str) {
            paths.extend(patch_header_paths(text));
        }
    }
    paths
}

fn fs_paths_mut(facts: &mut ToolFacts) -> Option<&mut Vec<ResolvedPath>> {
    match facts {
        ToolFacts::Fs { paths, .. } | ToolFacts::Patch { paths } => Some(paths),
        _ => None,
    }
}

fn lexical_path(raw: &str, cwd: &str, home: Option<&str>) -> Result<String, FsFault> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(FsFault::Empty);
    }
    if trimmed.contains('\0') {
        return Err(FsFault::Null);
    }
    let slashed = trimmed.replace('\\', "/");
    let stripped = strip_namespace(&slashed)?;
    let expanded = expand_home(&stripped, home)?;
    let joined = if is_absolute(&expanded) {
        expanded
    } else {
        join_cwd(cwd, &expanded)
    };
    collapse(&joined)
}

fn strip_namespace(path: &str) -> Result<String, FsFault> {
    if let Some(rest) = path.strip_prefix("//?/") {
        if let Some(unc) = rest.strip_prefix("UNC/") {
            return Ok(format!("//{unc}"));
        }
        return Ok(rest.to_owned());
    }
    if path.starts_with("//./") {
        return Err(FsFault::DevicePath);
    }
    Ok(path.to_owned())
}

fn expand_home(path: &str, home: Option<&str>) -> Result<String, FsFault> {
    let Some(rest) = path.strip_prefix('~') else {
        return Ok(path.to_owned());
    };
    if !rest.is_empty() && !rest.starts_with('/') {
        return Ok(path.to_owned());
    }
    let Some(home) = home.map(str::trim).filter(|home| !home.is_empty()) else {
        return Err(FsFault::HomeUnknown);
    };
    let home = home.replace('\\', "/");
    if rest.is_empty() {
        return Ok(home);
    }
    Ok(format!("{home}{}", rest))
}

fn is_absolute(path: &str) -> bool {
    path.starts_with('/') || drive_letter(path).is_some()
}

fn drive_letter(path: &str) -> Option<char> {
    let mut chars = path.chars();
    let letter = chars.next()?;
    if chars.next() == Some(':') && letter.is_ascii_alphabetic() {
        Some(letter)
    } else {
        None
    }
}

fn join_cwd(cwd: &str, relative: &str) -> String {
    let cwd = cwd.replace('\\', "/");
    let cwd = cwd.trim_end_matches('/');
    if cwd.is_empty() || cwd == "." {
        return relative.to_owned();
    }
    format!("{cwd}/{relative}")
}

fn collapse(path: &str) -> Result<String, FsFault> {
    let absolute = path.starts_with('/');
    let anchored = absolute || drive_letter(path).is_some();
    let mut out: Vec<String> = Vec::new();
    if absolute {
        out.push(String::new());
    }
    for (index, component) in path.split('/').enumerate() {
        if component.is_empty() {
            continue;
        }
        check_component(component, index)?;
        push_component(&mut out, component, anchored);
    }
    if out.is_empty() {
        return Ok(".".to_owned());
    }
    if out.len() == 1 && out[0].is_empty() {
        return Ok("/".to_owned());
    }
    Ok(out.join("/"))
}

fn check_component(component: &str, index: usize) -> Result<(), FsFault> {
    if component.contains('\0') {
        return Err(FsFault::Null);
    }
    if component.contains(':') && !(index == 0 && drive_letter(component).is_some()) {
        return Err(FsFault::AlternateDataStream);
    }
    if is_short_name(component) {
        return Err(FsFault::ShortName);
    }
    Ok(())
}

fn push_component(out: &mut Vec<String>, component: &str, anchored: bool) {
    if component.is_empty() || component == "." {
        return;
    }
    if component == ".." {
        let can_pop = match out.last().map(String::as_str) {
            Some("..") | None => false,
            Some("") => false,
            Some(last) if drive_letter(last).is_some() && out.len() == 1 => false,
            Some(_) => true,
        };
        if can_pop {
            out.pop();
        } else if !anchored {
            out.push("..".to_owned());
        }
        return;
    }
    out.push(component.to_owned());
}

fn is_short_name(component: &str) -> bool {
    if component == "." || component == ".." {
        return false;
    }
    let name = match component.split_once('.') {
        Some((name, ext)) => {
            if ext.is_empty() || ext.len() > 3 || ext.contains('.') {
                return false;
            }
            name
        }
        None => component,
    };
    let Some((base, tail)) = name.rsplit_once('~') else {
        return false;
    };
    if base.is_empty() || base.len() > 6 || tail.len() != 1 {
        return false;
    }
    tail.as_bytes().first().is_some_and(u8::is_ascii_digit)
        && base.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn realpath_best_effort(lexical: &str) -> String {
    let path = Path::new(lexical);
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return display_path(&canonical);
    }
    let mut remaining = Vec::new();
    let mut cursor = path;
    while let Some(parent) = cursor.parent() {
        if parent.as_os_str().is_empty() {
            break;
        }
        if let Some(name) = cursor.file_name() {
            remaining.push(name.to_owned());
        }
        if let Ok(canonical) = std::fs::canonicalize(parent) {
            let mut joined = canonical;
            for name in remaining.iter().rev() {
                joined.push(name);
            }
            return display_path(&joined);
        }
        cursor = parent;
    }
    lexical.to_owned()
}

fn display_path(path: &Path) -> String {
    let mut text = String::new();
    for component in path.components() {
        match component {
            Component::RootDir => text.push('/'),
            Component::Normal(part) => {
                if !text.ends_with('/') && !text.is_empty() {
                    text.push('/');
                }
                text.push_str(&part.to_string_lossy());
            }
            Component::Prefix(prefix) => {
                text.push_str(&prefix.as_os_str().to_string_lossy().replace('\\', "/"));
            }
            Component::CurDir | Component::ParentDir => {}
        }
    }
    if text.is_empty() {
        lexical_fallback(path)
    } else {
        text
    }
}

fn lexical_fallback(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn place_path(canonical: &str, workspace: &str) -> PathPlace {
    let workspace = workspace.trim();
    if workspace.is_empty() {
        return PathPlace::NoWorkspace;
    }
    let workspace = realpath_best_effort(&workspace.replace('\\', "/"));
    if !paths_match(canonical, &workspace) {
        return PathPlace::Outside;
    }
    let below = canonical.get(workspace.len()..).unwrap_or_default();
    if below.split('/').any(is_git_dir_name) {
        PathPlace::GitMetadata
    } else {
        PathPlace::Inside
    }
}

/// Repository metadata entry. The policy gate and every sandbox backend use
/// this one name.
pub const GIT_DIR_NAME: &str = ".git";

/// Start of the line in a gitfile (a `.git` file) that names the git dir.
pub const GITFILE_PREFIX: &str = "gitdir:";

/// Index entry mode of a submodule (gitlink).
pub const GITLINK_MODE: u32 = 0o160_000;

/// File in a linked worktree's git dir that names the common dir.
pub const COMMONDIR_FILE: &str = "commondir";

/// Directory in a git dir that holds the submodules' git dirs.
pub const MODULES_DIR: &str = "modules";

/// File in a git dir that holds the index.
pub const INDEX_FILE: &str = "index";

/// File in a git dir (the common dir, for a linked worktree) that holds
/// the repository config.
pub const CONFIG_FILE: &str = "config";

/// File in a git dir that names the current branch. Git discovery treats a
/// directory holding it, with [`OBJECTS_DIR`] and [`REFS_DIR`], as a bare
/// repository.
pub const HEAD_FILE: &str = "HEAD";

/// Directory in a git dir that holds the object store.
pub const OBJECTS_DIR: &str = "objects";

/// Directory in a git dir that holds the refs.
pub const REFS_DIR: &str = "refs";

/// A repository config line, whitespace removed and compared without case,
/// that says object ids are SHA-256.
pub const SHA256_OBJECT_FORMAT: &str = "objectformat=sha256";

/// Signature of the split-index extension in an index file.
pub const SPLIT_INDEX_EXTENSION: &[u8; 4] = b"link";

/// Repository config key that names the program git runs to watch the
/// work tree. Host-side git turns it off for repositories it did not write.
pub const FSMONITOR_KEY: &str = "core.fsmonitor";

/// Config key that decides whether git discovery may use a bare repository
/// it finds. Host-side git sets it to `explicit`, so a repository planted
/// in the workspace or a parent is never picked up.
pub const SAFE_BARE_REPOSITORY_KEY: &str = "safe.bareRepository";

/// Whether a single path component names [`GIT_DIR_NAME`]. Compared
/// case-insensitively, because the default macOS and Windows filesystems
/// are, and with trailing dots and spaces ignored, because Windows strips
/// them (`.git.` and `.git ` open `.git`).
#[must_use]
pub fn is_git_dir_name(name: &str) -> bool {
    name.trim_end_matches(['.', ' '])
        .eq_ignore_ascii_case(GIT_DIR_NAME)
}

fn paths_match(candidate: &str, workspace: &str) -> bool {
    let fold = windows_style(candidate) || windows_style(workspace);
    if fold {
        let candidate = candidate.to_ascii_lowercase();
        let workspace = workspace.to_ascii_lowercase();
        return candidate == workspace || candidate.starts_with(&format!("{workspace}/"));
    }
    candidate == workspace || candidate.starts_with(&format!("{workspace}/"))
}

fn windows_style(path: &str) -> bool {
    drive_letter(path).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots<'a>(cwd: &'a str, workspace: &'a str) -> FsRoots<'a> {
        FsRoots::new(cwd, workspace)
    }

    #[test]
    fn relative_dotdot_outside_the_workspace_is_outside() {
        let resolved =
            resolve_with_home("../../secret", roots("/work/sub", "/work"), None).expect("path");
        assert_eq!(resolved.canonical, "/secret");
        assert_eq!(resolved.place, PathPlace::Outside);
    }

    #[test]
    fn blank_workspace_places_no_workspace() {
        for workspace in ["", "  "] {
            let resolved =
                resolve_with_home("src/lib.rs", roots("/work", workspace), None).expect("path");
            assert_eq!(resolved.place, PathPlace::NoWorkspace);
        }
    }

    #[test]
    fn relative_file_inside_the_workspace_is_inside() {
        let resolved =
            resolve_with_home("./src/../src/lib.rs", roots("/work", "/work"), None).expect("path");
        assert_eq!(resolved.canonical, "/work/src/lib.rs");
        assert_eq!(resolved.place, PathPlace::Inside);
    }

    #[test]
    fn absolute_path_outside_is_outside() {
        let resolved =
            resolve_with_home("/etc/passwd", roots("/work", "/work"), None).expect("path");
        assert_eq!(resolved.place, PathPlace::Outside);
    }

    #[test]
    fn alternate_data_stream_is_rejected() {
        let err = resolve_with_home("notes.txt:Zone.Identifier", roots("/work", "/work"), None)
            .expect_err("stream");
        assert_eq!(err, FsFault::AlternateDataStream);
    }

    #[test]
    fn short_name_is_rejected() {
        let err = resolve_with_home("C:/PROGRA~1/file", roots("C:/work", "C:/work"), None)
            .expect_err("short");
        assert_eq!(err, FsFault::ShortName);
    }

    #[test]
    fn device_path_is_rejected() {
        let err = resolve_with_home(r"\\.\PhysicalDrive0", roots("/work", "/work"), None)
            .expect_err("device");
        assert_eq!(err, FsFault::DevicePath);
    }

    #[test]
    fn extended_prefix_inside_a_windows_workspace() {
        let resolved = resolve_with_home(r"\\?\C:\work\a.txt", roots(r"C:\work", r"C:\work"), None)
            .expect("extended");
        assert_eq!(resolved.canonical, "C:/work/a.txt");
        assert_eq!(resolved.place, PathPlace::Inside);
    }

    #[test]
    fn windows_path_comparison_folds_case() {
        let resolved =
            resolve_with_home(r"c:\Work\A.txt", roots(r"C:\work", r"C:\Work"), None).expect("case");
        assert_eq!(resolved.place, PathPlace::Inside);
    }

    #[test]
    fn home_expansion_does_not_grant_the_workspace() {
        let resolved = resolve_with_home("~/secret", roots("/work", "/work"), Some("/Users/me"))
            .expect("home");
        assert_eq!(resolved.canonical, "/Users/me/secret");
        assert_eq!(resolved.place, PathPlace::Outside);
    }

    #[test]
    fn missing_home_rejects_tilde() {
        let err = resolve_with_home("~/secret", roots("/work", "/work"), None).expect_err("home");
        assert_eq!(err, FsFault::HomeUnknown);
    }

    #[test]
    fn patch_headers_keep_add_update_and_delete_paths() {
        let text = "\
*** Begin Patch
*** Update File: ../secret
*** Add File: src/a.rs
*** Delete File: src/b.rs
*** End Patch
";
        assert_eq!(
            patch_header_paths(text),
            vec!["../secret", "src/a.rs", "src/b.rs"]
        );
    }

    #[test]
    fn user_sentence_is_not_split_into_grants() {
        let resolved = resolve_with_home("Please read /etc/passwd", roots("/work", "/work"), None)
            .expect("one path");
        assert_ne!(resolved.canonical, "/etc/passwd");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_to_an_outside_file_is_outside() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!(
            "caliche-fs-resolver-{stamp}-{}",
            std::process::id()
        ));
        let workspace = root.join("ws");
        let outside = root.join("outside.txt");
        std::fs::create_dir_all(&workspace).expect("mkdir");
        std::fs::write(&outside, "x").expect("write");
        let link = workspace.join("link");
        std::os::unix::fs::symlink(&outside, &link).expect("symlink");
        let resolved = resolve_with_home(
            "link",
            roots(
                workspace.to_str().expect("utf8"),
                workspace.to_str().expect("utf8"),
            ),
            None,
        )
        .expect("link");
        assert_eq!(resolved.place, PathPlace::Outside);
        let _ = std::fs::remove_dir_all(&root);
    }
}
