//! No host write tool reaches git metadata, in the widest mode
//! (`danger-full-access`, host paths unlocked). Approval modes such as
//! accept-all sit in front of these tools, so they cannot bypass the guard.

use super::common::{set_tool_host_fs_unlocked, set_tool_workspace_root};
use super::edit_file::EditFileTool;
use super::fs_tool::FsTool;
use super::write_file::WriteFileTool;
use super::{ApplyPatchTool, Tool};
use crate::policy::{CONFIG_FILE, HEAD_FILE, LOOSE_HEAD_CONTENTS, OBJECTS_DIR, REFS_DIR};
use serde_json::{Value, json};
use std::path::Path;
use tempfile::TempDir;

const CONFIG: &str = "[core]\n\tbare = false\n";

/// Paths that name a git config: the workspace repository, a case variant,
/// a trailing-dot variant, a nested repository, a submodule git dir, a
/// symlink to `.git`, a symlink into `.git` (caught only once the path is
/// resolved), and a git dir that a `.git` file points to.
const TARGETS: &[&str] = &[
    ".git/config",
    ".GIT/config",
    ".git./config",
    "nested/.git/config",
    ".git/modules/sub/config",
    "link/config",
    "hooks/pre-commit",
    "store/config",
];

fn make_git_dir(dir: &Path) {
    for sub in [OBJECTS_DIR, REFS_DIR] {
        std::fs::create_dir_all(dir.join(sub)).expect("git dir");
    }
    std::fs::write(dir.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    std::fs::write(dir.join(CONFIG_FILE), CONFIG).expect("config");
}

fn workspace() -> TempDir {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path();
    make_git_dir(&root.join(".git"));
    make_git_dir(&root.join(".git/modules/sub"));
    make_git_dir(&root.join("nested/.git"));
    make_git_dir(&root.join("store"));
    std::fs::create_dir_all(root.join("worktree")).expect("worktree");
    std::fs::write(
        root.join("worktree/.git"),
        format!("{} ../store\n", crate::policy::GITFILE_PREFIX),
    )
    .expect("gitdir file");
    std::fs::create_dir_all(root.join(".git/hooks")).expect("hooks");
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(".git", root.join("link")).expect("symlink");
        std::os::unix::fs::symlink(".git/hooks", root.join("hooks")).expect("symlink");
    }
    #[cfg(not(unix))]
    {
        make_git_dir(&root.join("link"));
        make_git_dir(&root.join("hooks"));
    }
    std::fs::write(root.join("notes.txt"), "notes\n").expect("notes");
    tmp
}

fn calls(target: &str) -> Vec<(&'static str, Box<dyn Tool>, Value)> {
    let operations = json!([{
        "type": "str_replace",
        "old_string": "bare = false",
        "new_string": "fsmonitor = touch pwned",
    }]);
    let patch = |body: String| json!({ "patch": format!("*** Begin Patch\n{body}*** End Patch") });
    vec![
        (
            "fs write",
            Box::new(FsTool::new()),
            json!({ "op": "write", "path": target, "content": "x" }),
        ),
        (
            "fs edit",
            Box::new(FsTool::new()),
            json!({ "op": "edit", "path": target, "operations": operations }),
        ),
        (
            "fs delete",
            Box::new(FsTool::new()),
            json!({ "op": "delete", "path": target }),
        ),
        (
            "write_file",
            Box::new(WriteFileTool),
            json!({ "path": target, "content": "x" }),
        ),
        (
            "edit_file",
            Box::new(EditFileTool),
            json!({ "path": target, "operations": operations }),
        ),
        (
            "apply_patch add",
            Box::new(ApplyPatchTool::new()),
            patch(format!("*** Add File: {target}\n+x\n")),
        ),
        (
            "apply_patch update",
            Box::new(ApplyPatchTool::new()),
            patch(format!(
                "*** Update File: {target}\n@@\n-\tbare = false\n+\tfsmonitor = touch pwned\n"
            )),
        ),
        (
            "apply_patch delete",
            Box::new(ApplyPatchTool::new()),
            patch(format!("*** Delete File: {target}\n")),
        ),
        (
            "apply_patch move",
            Box::new(ApplyPatchTool::new()),
            patch(format!(
                "*** Update File: notes.txt\n*** Move to: {target}\n@@\n-notes\n+[core]\n"
            )),
        ),
    ]
}

#[tokio::test]
async fn no_host_tool_writes_git_metadata() {
    let tmp = workspace();
    let root = tmp.path().canonicalize().expect("canonical root");
    set_tool_workspace_root(Some(root.clone()));
    set_tool_host_fs_unlocked(true);

    let mut reached = Vec::new();
    assert_eq!(
        std::fs::read_to_string(root.join(".git/config")).expect("config"),
        CONFIG
    );
    for target in TARGETS {
        for (tool, call, args) in calls(target) {
            // `.git.` is the same directory as `.git` only on Windows;
            // elsewhere it must still not exist afterwards.
            let before = std::fs::read_to_string(root.join(target)).ok();
            let result = call.execute(args).await;
            let after = std::fs::read_to_string(root.join(target)).ok();
            let notes = std::fs::read_to_string(root.join("notes.txt")).unwrap_or_default();
            if result.success || after != before || notes != "notes\n" {
                reached.push(format!("{tool} -> {target}"));
            }
        }
    }

    set_tool_host_fs_unlocked(false);
    set_tool_workspace_root(None);
    assert_eq!(reached, Vec::<String>::new());
}

#[tokio::test]
async fn host_tools_still_write_ordinary_files() {
    let tmp = workspace();
    let root = tmp.path().canonicalize().expect("canonical root");
    set_tool_workspace_root(Some(root.clone()));
    set_tool_host_fs_unlocked(true);

    let written = FsTool::new()
        .execute(json!({ "op": "write", "path": "src/gitignore.rs", "content": "ok\n" }))
        .await;
    let patched = ApplyPatchTool::new()
        .execute(json!({
            "patch": "*** Begin Patch\n*** Update File: notes.txt\n*** Move to: docs/git.md\n@@\n-notes\n+moved\n*** End Patch"
        }))
        .await;

    set_tool_host_fs_unlocked(false);
    set_tool_workspace_root(None);
    assert!(written.success, "{written:?}");
    assert!(patched.success, "{patched:?}");
    assert_eq!(
        std::fs::read_to_string(root.join("src/gitignore.rs")).expect("written"),
        "ok\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("docs/git.md")).expect("moved"),
        "moved\n"
    );
}

/// `apply_patch` resolves a hunk path three ways: absolute, joined to a
/// given `cwd`, and joined to the tool workspace. Each refuses `.git`.
#[tokio::test]
async fn apply_patch_refuses_git_metadata_by_absolute_path_and_cwd() {
    let tmp = workspace();
    let root = tmp.path().canonicalize().expect("canonical root");
    set_tool_workspace_root(Some(root.clone()));
    set_tool_host_fs_unlocked(true);
    let add = |path: &str| format!("*** Begin Patch\n*** Add File: {path}\n+x\n*** End Patch");

    let absolute = ApplyPatchTool::new()
        .execute(json!({ "patch": add(&root.join(".git/config").display().to_string()) }))
        .await;
    let joined = super::apply_patch::apply_hunks_to_cwd(&add(".git/config"), Some(&root)).await;

    set_tool_host_fs_unlocked(false);
    set_tool_workspace_root(None);
    assert!(!absolute.success, "{absolute:?}");
    assert_eq!(
        joined.expect_err("cwd hunk").to_string(),
        format!(
            "refusing to write '{}': it is inside a git repository's metadata",
            root.join(".git/config").display()
        )
    );
    assert_eq!(
        std::fs::read_to_string(root.join(".git/config")).expect("config"),
        CONFIG
    );
}

/// Git takes a directory whose `objects` and `refs` are executable files
/// as a git dir, so the guard does too.
#[cfg(unix)]
#[tokio::test]
async fn a_git_dir_with_executable_files_for_objects_and_refs_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path().canonicalize().expect("canonical root");
    let store = root.join("store");
    std::fs::create_dir(&store).expect("store");
    for part in [OBJECTS_DIR, REFS_DIR] {
        std::fs::write(store.join(part), "").expect("part file");
        std::fs::set_permissions(store.join(part), std::fs::Permissions::from_mode(0o755))
            .expect("chmod");
    }
    std::fs::write(store.join(HEAD_FILE), LOOSE_HEAD_CONTENTS).expect("HEAD");
    let add = "*** Begin Patch\n*** Add File: store/config\n+x\n*** End Patch";

    let joined = super::apply_patch::apply_hunks_to_cwd(add, Some(&root)).await;

    assert_eq!(
        joined.expect_err("refused").to_string(),
        format!(
            "refusing to write '{}': it is inside a git repository's metadata",
            store.join("config").display()
        )
    );
    assert!(!store.join("config").exists());
}
