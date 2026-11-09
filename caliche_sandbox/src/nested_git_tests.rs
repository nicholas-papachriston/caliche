use super::*;
use crate::git_targets::tests::real_git_dir;
use caliche_policy::GIT_DIR_NAME;

/// A canonical temp dir holding a workspace `ws` with a git dir of its own.
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let ws = dir.path().canonicalize().expect("canonical").join("ws");
    real_git_dir(&ws.join(GIT_DIR_NAME));
    (dir, ws)
}

fn rel(paths: &[&str]) -> Vec<PathBuf> {
    paths.iter().map(PathBuf::from).collect()
}

fn targets(ws: &Path, writable: &[PathBuf], bounds: Bounds) -> Result<Vec<PathBuf>, String> {
    guarded_targets_within(ws, writable, bounds).map_err(|err| err.to_string())
}

/// A nested repository and a gitfile that no index lists are targets; a
/// `.git` reached only through a symlink is not followed.
#[cfg(unix)]
#[test]
fn every_nested_git_in_the_workspace_is_a_target() {
    let (dir, ws) = workspace();
    real_git_dir(&ws.join("vendor").join("dep").join(GIT_DIR_NAME));
    std::fs::create_dir(ws.join("pkg")).expect("pkg");
    std::fs::write(ws.join("pkg").join(GIT_DIR_NAME), "gitdir: elsewhere\n").expect("gitfile");
    let outside = dir.path().join("outside");
    real_git_dir(&outside.join(GIT_DIR_NAME));
    std::os::unix::fs::symlink(&outside, ws.join("link")).expect("symlink");

    let found = targets(&ws, std::slice::from_ref(&ws), BOUNDS);

    assert_eq!(found, Ok(rel(&[".git", "pkg/.git", "vendor/dep/.git"])));
}

/// Without a writable root that reaches the workspace there is nothing to
/// guard, so the workspace is not scanned.
#[test]
fn a_read_only_workspace_is_not_scanned() {
    let (_dir, ws) = workspace();
    real_git_dir(&ws.join("vendor").join(GIT_DIR_NAME));

    let found = targets(&ws, &[], BOUNDS);

    assert_eq!(found, Ok(rel(&[".git"])));
}

/// A `.git` under a writable root outside the workspace is refused.
#[test]
fn a_git_under_a_writable_root_outside_the_workspace_is_refused() {
    let (dir, ws) = workspace();
    let other = dir.path().canonicalize().expect("canonical").join("other");
    real_git_dir(&other.join("repo").join(GIT_DIR_NAME));

    let refused = refuse_git_outside_within(&ws, &[ws.clone(), other.clone()], BOUNDS)
        .map_err(|err| err.to_string());

    assert_eq!(
        refused,
        Err(git_refusal::git_dir_under_writable_root(
            &other.join("repo").join(GIT_DIR_NAME),
            &other
        ))
    );
}

#[test]
fn a_directory_past_the_depth_limit_is_refused() {
    let (_dir, ws) = workspace();
    std::fs::create_dir_all(ws.join("a").join("b")).expect("a/b");
    let bounds = Bounds { depth: 1, ..BOUNDS };

    let found = targets(&ws, std::slice::from_ref(&ws), bounds);

    assert_eq!(
        found,
        Err(git_refusal::scan_too_deep(&ws.join("a").join("b"), 1))
    );
}

#[test]
fn a_workspace_past_the_entry_limit_is_refused() {
    let (_dir, ws) = workspace();
    for name in ["a", "b", "c"] {
        std::fs::write(ws.join(name), "").expect("file");
    }
    let bounds = Bounds {
        entries: 3,
        ..BOUNDS
    };

    let found = targets(&ws, std::slice::from_ref(&ws), bounds);

    assert_eq!(found, Err(git_refusal::scan_too_many_entries(&ws, 3)));
}

#[test]
fn a_scan_past_the_time_limit_is_refused() {
    let (_dir, ws) = workspace();
    let bounds = Bounds {
        time: Duration::ZERO,
        ..BOUNDS
    };

    let found = targets(&ws, std::slice::from_ref(&ws), bounds);

    assert_eq!(found, Err(git_refusal::scan_too_slow(&ws, Duration::ZERO)));
}

/// The entries inside a `.git` the scan finds are not read, so a large
/// repository does not count against the limit.
#[test]
fn the_scan_does_not_read_inside_a_git_dir() {
    let (_dir, ws) = workspace();
    for index in 0..10 {
        std::fs::write(ws.join(GIT_DIR_NAME).join(format!("f{index}")), "").expect("file");
    }
    let bounds = Bounds {
        entries: 1,
        ..BOUNDS
    };

    let found = targets(&ws, std::slice::from_ref(&ws), bounds);

    assert_eq!(found, Ok(rel(&[".git"])));
}

#[cfg(unix)]
#[test]
fn an_unlistable_directory_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let (_dir, ws) = workspace();
    let locked = ws.join("locked");
    std::fs::create_dir(&locked).expect("locked");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");
    std::fs::read_dir(&locked).expect_err("mode 000 is unlistable");

    let found = targets(&ws, std::slice::from_ref(&ws), BOUNDS);

    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod back");
    assert_eq!(found, Err(git_refusal::unlistable_dir(&locked)));
}

/// A gitfile copied without its git dir (as `bun install` leaves a
/// package cloned with submodules) names a git dir a command could
/// create inside the workspace, so the workspace is refused.
#[test]
fn a_copied_gitfile_whose_git_dir_is_missing_is_refused() {
    let (_dir, ws) = workspace();
    let pkg = ws.join("node_modules").join("pkg");
    std::fs::create_dir_all(&pkg).expect("pkg");
    std::fs::write(pkg.join(GIT_DIR_NAME), "gitdir: ../.git/modules/pkg\n").expect("gitfile");
    let writable = [ws.clone()];

    let refused = guarded_targets(&ws, &writable)
        .and_then(|targets| crate::git_targets::refuse_exposed_git_dirs(&ws, &targets, &writable))
        .map(|_| ())
        .map_err(|err| err.to_string());

    assert_eq!(
        refused,
        Err(git_refusal::git_dir_in_workspace(
            &ws.join("node_modules")
                .join(GIT_DIR_NAME)
                .join("modules")
                .join("pkg")
        ))
    );
}
