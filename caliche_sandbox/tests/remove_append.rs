//! `remove_beneath` and `append_file_beneath` act below a root without
//! following links.
#![cfg(any(unix, windows))]

use caliche_sandbox::{append_file_beneath, remove_beneath};
use std::io::Write as _;
use std::path::{Path, PathBuf};

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical workspace");
    (dir, root)
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

#[test]
fn a_file_and_a_directory_tree_are_removed() {
    let (_dir, root) = workspace();
    std::fs::write(root.join("a.txt"), "a").expect("a.txt");
    std::fs::create_dir_all(root.join("tree/sub")).expect("tree");
    std::fs::write(root.join("tree/one.txt"), "1").expect("one");
    std::fs::write(root.join("tree/sub/two.txt"), "2").expect("two");

    remove_beneath(&root, &root.join("a.txt"), false).expect("remove a.txt");
    let refused =
        remove_beneath(&root, &root.join("tree"), false).expect_err("a directory needs recursive");
    remove_beneath(&root, &root.join("tree"), true).expect("remove tree");

    assert!(!root.join("a.txt").exists());
    assert!(!root.join("tree").exists());
    assert!(
        refused.to_string().contains("removing it needs recursive"),
        "{refused}"
    );
}

/// A directory on the way that is a link is refused, and a link at the
/// target is removed itself: in both cases the linked directory's content
/// stays.
#[test]
fn a_link_is_never_followed_by_a_removal() {
    let (_dir, root) = workspace();
    let (_outside_dir, outside) = workspace();
    std::fs::write(outside.join("secret.txt"), "secret").expect("secret");
    link_dir(&outside, &root.join("escape"));

    remove_beneath(&root, &root.join("escape/secret.txt"), false)
        .expect_err("a link on the way is refused");
    remove_beneath(&root, &root.join("escape"), true).expect("the link itself is removed");

    assert_eq!(
        std::fs::read_to_string(outside.join("secret.txt")).expect("secret stays"),
        "secret"
    );
    assert!(std::fs::symlink_metadata(root.join("escape")).is_err());
}

#[test]
fn an_append_keeps_the_content_and_refuses_a_link_on_the_way() {
    let (_dir, root) = workspace();
    let (_outside_dir, outside) = workspace();
    std::fs::write(root.join("log.txt"), "one\n").expect("log");
    std::fs::write(outside.join("log.txt"), "outside\n").expect("outside log");
    link_dir(&outside, &root.join("escape"));

    append_file_beneath(&root, &root.join("log.txt"))
        .expect("open for append")
        .write_all(b"two\n")
        .expect("append");
    append_file_beneath(&root, &root.join("new/made.txt"))
        .expect("create for append")
        .write_all(b"made\n")
        .expect("append new");
    append_file_beneath(&root, &root.join("escape/log.txt"))
        .expect_err("a link on the way is refused");

    assert_eq!(
        std::fs::read_to_string(root.join("log.txt")).expect("log"),
        "one\ntwo\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("new/made.txt")).expect("made"),
        "made\n"
    );
    assert_eq!(
        std::fs::read_to_string(outside.join("log.txt")).expect("outside log"),
        "outside\n"
    );
}

/// The root itself is never removed, so `fs delete .` with recursive set
/// cannot empty the workspace.
#[test]
fn the_root_itself_is_not_removed() {
    let (_dir, root) = workspace();
    std::fs::write(root.join("keep.txt"), "keep").expect("keep");

    let refused = remove_beneath(&root, &root, true).expect_err("the root is refused");

    assert_eq!(
        refused.to_string(),
        format!(
            "'{}' names the sandbox root, not an entry below it",
            root.display()
        )
    );
    assert_eq!(
        std::fs::read_to_string(root.join("keep.txt")).expect("keep stays"),
        "keep"
    );
}

/// An append to a file with a second hard link is refused: the other name
/// can be a file outside the root.
#[test]
fn an_append_refuses_a_file_with_another_hard_link() {
    let (_dir, root) = workspace();
    let (_outside_dir, outside) = workspace();
    std::fs::write(outside.join("log.txt"), "outside\n").expect("outside log");
    std::fs::hard_link(outside.join("log.txt"), root.join("log.txt")).expect("hard link");

    let refused = append_file_beneath(&root, &root.join("log.txt"))
        .expect_err("a second hard link is refused");

    assert_eq!(
        refused.to_string(),
        format!(
            "refusing to write '{}': {}",
            root.join("log.txt").display(),
            caliche_sandbox::HARD_LINK_REFUSAL
        )
    );
    assert_eq!(
        std::fs::read_to_string(outside.join("log.txt")).expect("outside log"),
        "outside\n"
    );
}
