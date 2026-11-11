//! `list_dir_beneath` lists a directory below a root without following links.
#![cfg(any(unix, windows))]

use caliche_sandbox::{DirEntryBeneath, EntryKind, list_dir_beneath};
use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical workspace");
    (dir, root)
}

fn entry(name: &str, kind: EntryKind, len: u64) -> DirEntryBeneath {
    DirEntryBeneath {
        name: OsString::from(name),
        kind,
        len,
    }
}

fn sorted(mut entries: Vec<DirEntryBeneath>) -> Vec<DirEntryBeneath> {
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

/// Link `link` to the directory `target`: a symlink on Unix, a junction on
/// Windows (which needs no privilege).
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn link_dir(target: &Path, link: &Path) {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, link).expect("symlink");
    #[cfg(windows)]
    {
        let status = std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .stdout(std::process::Stdio::null())
            .status()
            .expect("mklink");
        assert!(status.success(), "mklink /J failed");
    }
}

/// The refusal for a link at `sub`, the last component of `target`.
fn link_refusal(root: &Path, target: &Path) -> String {
    #[cfg(unix)]
    {
        let _ = root;
        format!(
            "refusing to follow symlink at 'sub' in '{}'",
            target.display()
        )
    }
    #[cfg(windows)]
    {
        format!(
            "refusing to follow reparse point between '{}' and '{}'",
            root.display(),
            target.display()
        )
    }
}

#[test]
fn lists_names_kinds_and_lengths_without_following_links() {
    let (_dir, root) = workspace();
    let outside = tempfile::tempdir().expect("outside");
    std::fs::write(outside.path().join("secret.txt"), b"secret").expect("outside file");
    std::fs::create_dir(root.join("sub")).expect("sub");
    std::fs::write(root.join("sub/a.txt"), b"abc").expect("a.txt");
    std::fs::create_dir(root.join("sub/inner")).expect("inner");
    link_dir(outside.path(), &root.join("sub/link"));

    let entries = list_dir_beneath(&root, &root.join("sub")).expect("list");

    assert_eq!(
        sorted(entries),
        vec![
            entry("a.txt", EntryKind::File, 3),
            entry("inner", EntryKind::Dir, 0),
            entry("link", EntryKind::Other, 0),
        ]
    );
}

#[test]
fn the_root_itself_is_listed() {
    let (_dir, root) = workspace();
    std::fs::write(root.join("top.txt"), b"12345").expect("top.txt");

    let entries = list_dir_beneath(&root, &root).expect("list");

    assert_eq!(entries, vec![entry("top.txt", EntryKind::File, 5)]);
}

#[test]
fn a_directory_swapped_for_a_link_is_refused_not_listed() {
    let (_dir, root) = workspace();
    let outside = tempfile::tempdir().expect("outside");
    std::fs::write(outside.path().join("secret.txt"), b"secret").expect("outside file");
    let target = root.join("sub");
    link_dir(outside.path(), &target);

    let err = list_dir_beneath(&root, &target).expect_err("a linked directory must be refused");

    assert_eq!(err.kind(), ErrorKind::PermissionDenied);
    assert_eq!(err.to_string(), link_refusal(&root, &target));
}

#[test]
fn a_path_outside_the_root_is_refused() {
    let (_dir, root) = workspace();
    let outside = tempfile::tempdir().expect("outside");
    let target = outside.path().canonicalize().expect("canonical outside");

    let err = list_dir_beneath(&root, &target).expect_err("outside the root");

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        err.to_string(),
        format!("'{}' is not below '{}'", target.display(), root.display())
    );
}

#[test]
fn a_file_is_not_listed() {
    let (_dir, root) = workspace();
    std::fs::write(root.join("file.txt"), b"x").expect("file");

    let err = list_dir_beneath(&root, &root.join("file.txt")).expect_err("a file is not a dir");

    assert_eq!(err.kind(), ErrorKind::NotADirectory);
}
