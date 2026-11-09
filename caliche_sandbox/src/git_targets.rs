//! Workspace paths that host `git` reads as repository metadata.
//!
//! Host `git` run in the workspace root reads the root `.git` and, for each
//! submodule recorded in the index, the `.git` at that submodule's path (a
//! gitfile naming its git dir). A sandbox that lets a command write any of
//! them lets it plant config, such as `core.fsmonitor`, that host `git`
//! runs later.
//!
//! The targets are the root `.git` (even when it does not exist yet, so a
//! command cannot create one) and `<path>/.git` for every submodule path
//! recorded in the root index and, recursively, in each submodule's own
//! index. Only indexes are read; this module never walks the workspace
//! tree, so on macOS instance creation does no work that grows with build
//! output such as `target/` or `node_modules/`.
//!
//! A `.git` in an ordinary directory, or a nested repository that is not a
//! submodule, is not a target here. Host `git` in the workspace root does
//! not read it, but host `git` run inside that directory does. The macOS
//! profile denies writes to a `.git` at any depth by name; the Linux and
//! Docker backends add the ones a bounded scan finds (see `nested_git`).
//!
//! A target that is a gitfile names a git dir elsewhere, and that git dir
//! may name a common dir. Host `git` reads config and hooks there, so
//! [`refuse_exposed_git_dirs`] refuses either one when the sandbox can
//! write it. A workspace without a `.git` of its own is inside whatever
//! repository host `git` finds above it, so that repository's git dir and
//! common dir are checked the same way.

use crate::paths::git_refusal;
use anyhow::{Context, Result};
use caliche_policy::{
    CONFIG_FILE, GIT_DIR_LINKS, GIT_DIR_NAME, GITLINK_MODE, HEAD_FILE, INDEX_FILE, REFS_DIR,
    SHA256_OBJECT_FORMAT, SPLIT_INDEX_EXTENSION, WorkspaceGit, classify, read_gitfile,
};
use std::path::{Component, Path, PathBuf};

const MODE_TYPE_MASK: u32 = 0o170_000;
/// Fixed index entry fields before the object id: ctime, mtime, dev, ino,
/// mode, uid, gid, and size.
const ENTRY_STAT_LEN: usize = 40;
const EXTENDED_FLAG: u16 = 0x4000;

/// Refuse when the git dir or common dir of any existing target is where
/// the sandbox can write it: inside `workspace` but not under a target, or
/// under one of `writable`, or holding one of `writable`. Each dir is
/// resolved one component at a time, as the kernel does, and refused when
/// a symlink on the way is where the sandbox could retarget it.
/// `workspace` and `writable` are canonical.
pub(crate) fn refuse_exposed_git_dirs(
    workspace: &Path,
    targets: &[PathBuf],
    writable: &[PathBuf],
) -> Result<WorkspaceGit> {
    let class = classify_workspace(workspace)?;
    for target in targets {
        if let Some(git_dir) = named_git_dir(&workspace.join(target))? {
            refuse_exposed_git_dir(workspace, targets, writable, &git_dir)?;
        }
    }
    if let Some(git_dir) = discovered_metadata_dir(&class) {
        refuse_exposed_git_dir(workspace, targets, writable, git_dir)?;
    }
    Ok(class)
}

/// Refuse when `git_dir`, or the common dir it names, is where the sandbox
/// can write it.
fn refuse_exposed_git_dir(
    workspace: &Path,
    targets: &[PathBuf],
    writable: &[PathBuf],
    git_dir: &Path,
) -> Result<()> {
    let git_dir = resolve_git_dir(workspace, writable, git_dir)?;
    refuse_exposed(workspace, targets, writable, &git_dir)?;
    refuse_git_dir_links(workspace, targets, writable, &git_dir)?;
    let common = resolve_git_dir(workspace, writable, &common_dir(&git_dir)?)?;
    refuse_exposed(workspace, targets, writable, &common)?;
    refuse_git_dir_links(workspace, targets, writable, &common)
}

/// Refuse a `HEAD`, `objects`, `refs`, or `commondir` symlink inside `dir`
/// when the sandbox could retarget it or delete what it points at.
/// A missing entry is skipped: a linked worktree's git dir has no `objects`.
fn refuse_git_dir_links(
    workspace: &Path,
    targets: &[PathBuf],
    writable: &[PathBuf],
    dir: &Path,
) -> Result<()> {
    for name in GIT_DIR_LINKS {
        let path = dir.join(name);
        let meta = match std::fs::symlink_metadata(&path) {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(err).with_context(|| format!("stat {}", path.display())),
        };
        if !meta.file_type().is_symlink() {
            continue;
        }
        let resolved = resolve_git_dir(workspace, writable, &path)?;
        refuse_exposed(workspace, targets, writable, &resolved)?;
    }
    Ok(())
}

/// [`classify`] for `workspace`, with invalid entries and a stray `HEAD`
/// already refused. This is the only place those two results become errors.
pub(crate) fn classify_workspace(workspace: &Path) -> Result<WorkspaceGit> {
    classify(workspace).map_err(|fault| {
        anyhow::anyhow!(match fault {
            caliche_policy::ClassifyFault::InvalidGitEntry(path) => {
                git_refusal::invalid_git_entry(&path)
            }
            caliche_policy::ClassifyFault::StrayHead(path) => git_refusal::stray_head(&path),
        })
    })
}

/// The git dir host `git` discovers above the workspace, or the workspace
/// itself when it is a bare repository. The workspace's own `.git` is a
/// target and is not returned here.
fn discovered_metadata_dir(class: &WorkspaceGit) -> Option<&Path> {
    match class {
        WorkspaceGit::Own { .. } | WorkspaceGit::None => None,
        WorkspaceGit::Bare { git_dir } | WorkspaceGit::Enclosing { git_dir, .. } => Some(git_dir),
    }
}

fn refuse_exposed(
    workspace: &Path,
    targets: &[PathBuf],
    writable: &[PathBuf],
    dir: &Path,
) -> Result<()> {
    if let Ok(below) = dir.strip_prefix(workspace) {
        if targets.iter().any(|target| below.starts_with(target)) {
            return Ok(());
        }
        anyhow::bail!(git_refusal::git_dir_in_workspace(dir));
    }
    if let Some(root) = writable.iter().find(|root| dir.starts_with(root)) {
        anyhow::bail!(git_refusal::git_dir_under_writable_root(dir, root));
    }
    if let Some(root) = writable.iter().find(|root| root.starts_with(dir)) {
        anyhow::bail!(git_refusal::overlapping_root(root, dir));
    }
    Ok(())
}

/// `dir` (absolute) resolved one component at a time: a symlink is
/// followed and `..` goes to the parent of what was reached so far. A
/// symlink inside `workspace` or under a writable root is refused, because
/// the sandbox could retarget it after the check. When a component does not
/// exist, the rest of `dir` is appended without touching the disk: the
/// result is where the dir would be created, and the caller refuses it
/// when the sandbox could create it there.
fn resolve_git_dir(workspace: &Path, writable: &[PathBuf], dir: &Path) -> Result<PathBuf> {
    let mut out = PathBuf::new();
    let parts: Vec<Component<'_>> = dir.components().collect();
    for (index, part) in parts.iter().copied().enumerate() {
        let name = match part {
            Component::Prefix(_) | Component::RootDir => {
                out.push(part);
                continue;
            }
            Component::CurDir => continue,
            Component::ParentDir => {
                out.pop();
                continue;
            }
            Component::Normal(name) => name,
        };
        let next = out.join(name);
        let meta = match std::fs::symlink_metadata(&next) {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(missing_rest(next, &parts[index + 1..]));
            }
            Err(err) => return Err(err).with_context(|| format!("stat {}", next.display())),
        };
        if !meta.file_type().is_symlink() {
            out = next;
            continue;
        }
        if next.starts_with(workspace) {
            anyhow::bail!(git_refusal::git_dir_in_workspace(&next));
        }
        if let Some(root) = writable.iter().find(|root| next.starts_with(root)) {
            anyhow::bail!(git_refusal::git_dir_under_writable_root(&next, root));
        }
        out = follow_outside_symlink(&next)?;
    }
    Ok(out)
}

/// `missing`, the first path that does not exist, followed by `rest`, with
/// `.` dropped and `..` taken lexically.
fn missing_rest(missing: PathBuf, rest: &[Component<'_>]) -> PathBuf {
    let mut out = missing;
    for part in rest {
        match part {
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => out.push(name),
            Component::CurDir | Component::Prefix(_) | Component::RootDir => {}
        }
    }
    out
}

/// The canonical target of a symlink that is outside the workspace and
/// every writable root. A dangling `HEAD` that names `refs/…` is the
/// unborn-branch shape git accepts. Name it instead of a bare
/// `canonicalize` error.
fn follow_outside_symlink(path: &Path) -> Result<PathBuf> {
    match path.canonicalize() {
        Ok(canonical) => Ok(canonical),
        Err(err) => missing_symlink(path, err),
    }
}

fn missing_symlink(path: &Path, err: std::io::Error) -> Result<PathBuf> {
    if unborn_head_symlink(path) {
        Err(err).context(git_refusal::dangling_head(path))
    } else {
        Err(err).with_context(|| format!("canonicalize {}", path.display()))
    }
}

/// A `HEAD` symlink whose own text starts with `refs/` and whose target
/// is missing. Git treats that as an unborn branch.
fn unborn_head_symlink(path: &Path) -> bool {
    if path.file_name() != Some(std::ffi::OsStr::new(HEAD_FILE)) {
        return false;
    }
    let Ok(target) = std::fs::read_link(path) else {
        return false;
    };
    let text = target.to_string_lossy();
    if !(text.starts_with(REFS_DIR) && text.as_bytes().get(REFS_DIR.len()) == Some(&b'/')) {
        return false;
    }
    let Some(parent) = path.parent() else {
        return false;
    };
    std::fs::symlink_metadata(parent.join(target))
        .is_err_and(|err| err.kind() == std::io::ErrorKind::NotFound)
}

/// Deepest submodule nesting followed before giving up.
const MAX_SUBMODULE_DEPTH: usize = 16;

/// Every submodule path recorded in the root index and, recursively, in
/// each submodule's own index (checked out or not), relative to
/// `workspace`. Reads indexes only; it does not walk the tree.
fn index_submodules(workspace: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    if let Some(git_dir) = git_dir(workspace)? {
        nested_gitlinks(workspace, &git_dir, Path::new(""), 0, &mut paths)?;
    }
    Ok(paths)
}

/// The `.git` paths host `git` in the workspace root reads, relative to
/// `workspace`, sorted: the root `.git` and `<path>/.git` for every
/// submodule in [`index_submodules`]. Reads indexes only.
pub(crate) fn index_targets(workspace: &Path) -> Result<Vec<PathBuf>> {
    let mut targets: Vec<PathBuf> = index_submodules(workspace)?
        .into_iter()
        .map(|path| path.join(GIT_DIR_NAME))
        .collect();
    targets.push(PathBuf::from(GIT_DIR_NAME));
    targets.sort();
    targets.dedup();
    Ok(targets)
}

/// Directories that lead to a submodule below the workspace root, relative
/// to `workspace`, sorted: each directory on the path to every submodule
/// in [`index_submodules`]. The macOS profile denies writes to these
/// entries, so none can be moved aside, replaced or created in place of a
/// missing one. Refused when one that exists is a symlink: the kernel
/// would resolve through it to a path the deny does not name. A `.git`
/// that is not a submodule is not followed: host `git` in the workspace
/// root does not read it, and the profile denies writes to it by name.
#[cfg(any(test, all(feature = "sandbox-macos", target_os = "macos")))]
pub(crate) fn guarded_dirs(workspace: &Path) -> Result<Vec<PathBuf>> {
    let leads = index_submodules(workspace)?;
    let mut dirs = std::collections::BTreeSet::new();
    for lead in leads {
        let mut prefix = PathBuf::new();
        for part in lead.components() {
            let Component::Normal(name) = part else {
                anyhow::bail!(git_refusal::path_not_plain(&lead));
            };
            prefix.push(name);
            let path = workspace.join(&prefix);
            match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    anyhow::bail!(git_refusal::symlink_leads_to_git(&prefix));
                }
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(err).with_context(|| format!("stat {}", path.display())),
            }
            dirs.insert(prefix.clone());
        }
    }
    Ok(dirs.into_iter().collect())
}

/// Push every submodule path in the index of `git_dir`, under `prefix`,
/// and recurse into each one's git dir: its checked-out `.git`, or
/// `modules/<path>` in `git_dir` when it is not checked out.
fn nested_gitlinks(
    workspace: &Path,
    git_dir: &Path,
    prefix: &Path,
    depth: usize,
    out: &mut Vec<PathBuf>,
) -> Result<()> {
    if depth > MAX_SUBMODULE_DEPTH {
        anyhow::bail!(git_refusal::submodules_too_deep(prefix));
    }
    for path in index_gitlinks(git_dir)? {
        let full = prefix.join(&path);
        let checked_out = named_git_dir(&workspace.join(&full).join(GIT_DIR_NAME))?;
        let modules = git_dir.join(caliche_policy::MODULES_DIR).join(&path);
        let sub_git = checked_out.or_else(|| modules.is_dir().then_some(modules));
        if let Some(sub_git) = sub_git {
            nested_gitlinks(workspace, &sub_git, &full, depth + 1, out)?;
        }
        out.push(full);
    }
    Ok(())
}

/// Whether `targets` names only the root `.git`.
#[cfg_attr(
    not(any(
        test,
        feature = "docker",
        all(feature = "sandbox-linux", target_os = "linux")
    )),
    expect(dead_code, reason = "only the Linux and Docker backends pick a mode")
)]
pub(crate) fn only_root(targets: &[PathBuf]) -> bool {
    targets.iter().all(|path| path == Path::new(GIT_DIR_NAME))
}

/// Submodule paths recorded in the index in `git_dir`. None without an
/// index.
fn index_gitlinks(git_dir: &Path) -> Result<Vec<PathBuf>> {
    let index = git_dir.join(INDEX_FILE);
    let data = match std::fs::read(&index) {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err).with_context(|| format!("read {}", index.display())),
    };
    let oid_len = if uses_sha256(&git_dir)? { 32 } else { 20 };
    let paths = gitlinks(&data, oid_len)
        .with_context(|| format!("read submodules from {}", index.display()))?;
    Ok(paths
        .iter()
        .map(|path| path.split('/').collect::<PathBuf>())
        .collect())
}

/// The workspace git dir: `.git` itself, or the dir a `.git` file names.
fn git_dir(workspace: &Path) -> Result<Option<PathBuf>> {
    named_git_dir(&workspace.join(GIT_DIR_NAME))
}

/// The git dir of the `.git` entry `git`: the entry itself, or the dir it
/// names when it is a gitfile. None when it does not exist.
fn named_git_dir(git: &Path) -> Result<Option<PathBuf>> {
    let meta = match std::fs::symlink_metadata(git) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("stat {}", git.display())),
    };
    if !meta.is_file() {
        return Ok(Some(git.to_path_buf()));
    }
    match read_gitfile(git) {
        Ok(Some(path)) => Ok(Some(path)),
        Ok(None) => anyhow::bail!("{} names no gitdir", git.display()),
        Err(err) => Err(err).with_context(|| format!("read {}", git.display())),
    }
}

/// The common dir `git_dir` shares with its main repository: the dir its
/// `commondir` file names, or `git_dir` itself. Spaces in the file are
/// part of the path.
fn common_dir(git_dir: &Path) -> Result<PathBuf> {
    caliche_policy::common_dir(git_dir)
        .with_context(|| format!("read commondir in {}", git_dir.display()))
}

/// Whether the repository's object ids are SHA-256 (32 bytes in the index).
fn uses_sha256(git_dir: &Path) -> Result<bool> {
    let config = common_dir(git_dir)?.join(CONFIG_FILE);
    let text = match std::fs::read_to_string(&config) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err).with_context(|| format!("read {}", config.display())),
    };
    Ok(text.lines().any(|line| {
        let line: String = line.chars().filter(|c| !c.is_whitespace()).collect();
        line.eq_ignore_ascii_case(SHA256_OBJECT_FORMAT)
    }))
}

/// Paths of gitlink entries in a version 2, 3, or 4 index.
fn gitlinks(data: &[u8], oid_len: usize) -> Result<Vec<String>> {
    if data.get(..4) != Some(b"DIRC".as_slice()) {
        anyhow::bail!("not a git index");
    }
    let version = read_u32(data, 4)?;
    if !(2..=4).contains(&version) {
        anyhow::bail!("unsupported index version {version}");
    }
    let count = read_u32(data, 8)?;
    let mut pos = 12;
    let mut previous = Vec::new();
    let mut found = Vec::new();
    for _ in 0..count {
        let mode = read_u32(data, pos + 24)?;
        let flags_at = pos + ENTRY_STAT_LEN + oid_len;
        let flags = read_u16(data, flags_at)?;
        let mut name_at = flags_at + 2;
        if version >= 3 && flags & EXTENDED_FLAG != 0 {
            name_at += 2;
        }
        let (path, next) = if version == 4 {
            let (strip, suffix_at) = read_offset_varint(data, name_at)?;
            let keep = previous
                .len()
                .checked_sub(strip)
                .context("index path prefix longer than the previous path")?;
            let (suffix, end) = read_name(data, suffix_at)?;
            let mut path = previous[..keep].to_vec();
            path.extend_from_slice(suffix);
            (path, end + 1)
        } else {
            let (name, end) = read_name(data, name_at)?;
            let padded = (end - pos + 8) & !7;
            (name.to_vec(), pos + padded)
        };
        if mode & MODE_TYPE_MASK == GITLINK_MODE {
            found.push(String::from_utf8(path.clone()).context("submodule path is not UTF-8")?);
        }
        previous = path;
        pos = next;
    }
    refuse_split_index(data, pos, oid_len)?;
    Ok(found)
}

/// Refuse when any extension after the entries, which start at `pos`, is
/// the split index `link`. The index ends with a checksum of `oid_len`.
fn refuse_split_index(data: &[u8], mut pos: usize, oid_len: usize) -> Result<()> {
    let end = data
        .len()
        .checked_sub(oid_len)
        .context("index ends early")?;
    while pos + 8 <= end {
        if data.get(pos..pos + 4) == Some(SPLIT_INDEX_EXTENSION.as_slice()) {
            anyhow::bail!(git_refusal::SPLIT_INDEX);
        }
        let size = usize::try_from(read_u32(data, pos + 4)?)?;
        pos = pos
            .checked_add(8 + size)
            .context("index extension size overflows")?;
    }
    Ok(())
}

fn read_u32(data: &[u8], at: usize) -> Result<u32> {
    let bytes = data.get(at..at + 4).context("index ends early")?;
    Ok(u32::from_be_bytes(bytes.try_into()?))
}

fn read_u16(data: &[u8], at: usize) -> Result<u16> {
    let bytes = data.get(at..at + 2).context("index ends early")?;
    Ok(u16::from_be_bytes(bytes.try_into()?))
}

/// A NUL-terminated name and the offset of its NUL.
fn read_name(data: &[u8], at: usize) -> Result<(&[u8], usize)> {
    let rest = data.get(at..).context("index ends early")?;
    let len = rest
        .iter()
        .position(|&byte| byte == 0)
        .context("index path has no terminator")?;
    Ok((&rest[..len], at + len))
}

/// Git's offset varint, used for the version 4 path prefix length.
fn read_offset_varint(data: &[u8], mut at: usize) -> Result<(usize, usize)> {
    let mut byte = *data.get(at).context("index ends early")?;
    at += 1;
    let mut value = usize::from(byte & 0x7f);
    while byte & 0x80 != 0 {
        byte = *data.get(at).context("index ends early")?;
        at += 1;
        value = value
            .checked_add(1)
            .and_then(|v| v.checked_mul(128))
            .context("index path prefix length overflows")?
            | usize::from(byte & 0x7f);
    }
    Ok((value, at))
}

#[cfg(test)]
#[path = "git_targets_tests.rs"]
pub(crate) mod tests;
