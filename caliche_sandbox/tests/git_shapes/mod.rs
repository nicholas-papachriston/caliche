// Layouts where host git discovery finds a repository the sandbox could
// write, or could finish planting one. Each backend's create must refuse
// every one it applies to, with the error named here.

use caliche_policy::{HEAD_FILE, LOOSE_HEAD_CONTENTS, OBJECTS_DIR, REFS_DIR, write_loose_git_dir};
#[cfg(unix)]
use caliche_sandbox::paths::{GIT_DIR_NAME, git_refusal};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;

#[cfg(unix)]
pub struct Shape {
    pub name: &'static str,
    pub workspace: PathBuf,
    /// The writable root the backend gets: the workspace or a parent.
    pub writable: PathBuf,
    pub refusal: String,
}

/// A real git dir: `HEAD`, `objects/` and `refs/`.
#[expect(clippy::expect_used, reason = "test fixture")]
pub fn git_dir_at(dir: &Path) {
    write_loose_git_dir(dir).expect("git dir");
}

/// Every shape, each in its own directory under `base` (canonical). The
/// shapes use Unix modes and symlinks.
#[cfg(unix)]
#[expect(clippy::expect_used, reason = "test fixture")]
pub fn planted_repository_shapes(base: &Path) -> Vec<Shape> {
    let at = |name: &str| {
        let dir = base.join(name);
        std::fs::create_dir_all(&dir).expect("shape dir");
        dir
    };
    let mut shapes = Vec::new();

    // An interrupted init leaves an empty `.git`; git skips it and takes
    // the root, with the rest planted, as a bare repository.
    let ws = at("empty-root-git");
    std::fs::create_dir(ws.join(GIT_DIR_NAME)).expect(".git");
    git_dir_at(&ws);
    shapes.push(Shape {
        name: "an empty .git in the root",
        refusal: git_refusal::invalid_git_entry(&ws.join(GIT_DIR_NAME)),
        writable: ws.clone(),
        workspace: ws,
    });

    // Git skips the workspace's empty `.git` and uses the repository above.
    let repo = at("empty-sub-git");
    git_dir_at(&repo.join(GIT_DIR_NAME));
    let ws = repo.join("sub");
    std::fs::create_dir_all(ws.join(GIT_DIR_NAME)).expect("sub .git");
    shapes.push(Shape {
        name: "an empty .git in a subdirectory workspace",
        refusal: git_refusal::invalid_git_entry(&ws.join(GIT_DIR_NAME)),
        writable: ws.clone(),
        workspace: ws,
    });

    // Git only needs `objects` and `refs` to be searchable.
    let parent = at("exec-files");
    for part in [OBJECTS_DIR, REFS_DIR] {
        let file = parent.join(part);
        std::fs::write(&file, "").expect("part file");
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }
    std::fs::write(parent.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    let ws = parent.join("ws");
    std::fs::create_dir(&ws).expect("ws");
    shapes.push(Shape {
        name: "executable files for objects and refs in a parent",
        refusal: git_refusal::git_dir_under_writable_root(&parent, &parent),
        writable: parent,
        workspace: ws,
    });

    // Git accepts a `HEAD` symlink into `refs/` that does not resolve.
    let parent = at("dangling-head");
    for part in [OBJECTS_DIR, REFS_DIR] {
        std::fs::create_dir(parent.join(part)).expect("part dir");
    }
    std::os::unix::fs::symlink("refs/heads/main", parent.join(HEAD_FILE)).expect("HEAD link");
    let ws = parent.join("ws");
    std::fs::create_dir(&ws).expect("ws");
    shapes.push(Shape {
        name: "a dangling HEAD symlink in a parent",
        refusal: git_refusal::git_dir_under_writable_root(&parent, &parent),
        writable: parent,
        workspace: ws,
    });

    // A `HEAD` from an earlier session could be finished into a repository.
    let ws = at("head-in-root");
    std::fs::write(ws.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    shapes.push(Shape {
        name: "a HEAD already in the root",
        refusal: git_refusal::stray_head(&ws.join(HEAD_FILE)),
        writable: ws.clone(),
        workspace: ws,
    });

    let parent = at("head-in-parent");
    std::fs::write(parent.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    let ws = parent.join("ws");
    std::fs::create_dir(&ws).expect("ws");
    shapes.push(Shape {
        name: "a HEAD already in a parent",
        refusal: git_refusal::stray_head(&parent.join(HEAD_FILE)),
        writable: ws.clone(),
        workspace: ws,
    });

    // Git uses the `.git`. A sibling `HEAD` would be a bare repository if
    // that `.git` were later rejected.
    let ws = at("head-beside-git");
    git_dir_at(&ws.join(GIT_DIR_NAME));
    std::fs::write(ws.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    shapes.push(Shape {
        name: "a HEAD beside a valid .git",
        refusal: git_refusal::stray_head(&ws.join(HEAD_FILE)),
        writable: ws.clone(),
        workspace: ws,
    });

    // Deleting the link target makes git reject `.git` and use the root.
    let ws = at("objects-link");
    let git = ws.join(GIT_DIR_NAME);
    git_dir_at(&git);
    std::fs::create_dir(ws.join("store")).expect("store");
    std::fs::remove_dir(git.join(OBJECTS_DIR)).expect("objects");
    std::os::unix::fs::symlink("../store", git.join(OBJECTS_DIR)).expect("link");
    shapes.push(Shape {
        name: "an objects symlink into the workspace",
        refusal: git_refusal::git_dir_in_workspace(&git.join(OBJECTS_DIR)),
        writable: ws.clone(),
        workspace: ws,
    });

    shapes
}
