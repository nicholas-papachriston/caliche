//! A path the policy gate allowed as inside the workspace is swapped for a
//! link to a file outside it before the tool runs. The tool must not read
//! or write that outside file.

use super::common::{set_tool_host_fs_unlocked, set_tool_workspace_root};
use super::execution::middleware::{
    HookExecutionMiddleware, ToolExecutionContext, ToolExecutionMiddleware as _,
};
use super::fs_tool::FsTool;
use super::{ApplyPatchTool, Tool};
use crate::agent::hooks::HookRegistry;
use crate::policy::ModeOverlay;
use serde_json::{Value, json};
use std::path::Path;
use tempfile::TempDir;

const INSIDE: &str = "inside\n";
const SECRET: &str = "OUTSIDE-SECRET\n";

/// A workspace holding `a.txt` and `d/a.txt`, and an outside directory
/// holding `secret.txt` and `d/a.txt`, both with [`SECRET`].
fn layout() -> (TempDir, TempDir) {
    let workspace = TempDir::new().expect("workspace");
    let outside = TempDir::new().expect("outside");
    std::fs::write(workspace.path().join("a.txt"), INSIDE).expect("a.txt");
    std::fs::create_dir(workspace.path().join("d")).expect("d");
    std::fs::write(workspace.path().join("d/a.txt"), INSIDE).expect("d/a.txt");
    std::fs::write(outside.path().join("secret.txt"), SECRET).expect("secret");
    std::fs::create_dir(outside.path().join("d")).expect("outside d");
    std::fs::write(outside.path().join("d/a.txt"), SECRET).expect("outside d/a.txt");
    (workspace, outside)
}

/// Replace `a.txt` with a link to the outside secret, or the directory `d`
/// with a link to the outside `d`.
fn swap(workspace: &Path, outside: &Path, target: &str) {
    if target == "a.txt" {
        std::fs::remove_file(workspace.join("a.txt")).expect("remove a.txt");
        std::os::unix::fs::symlink(outside.join("secret.txt"), workspace.join("a.txt"))
            .expect("link a.txt");
    } else {
        std::fs::remove_dir_all(workspace.join("d")).expect("remove d");
        std::os::unix::fs::symlink(outside.join("d"), workspace.join("d")).expect("link d");
    }
}

fn outside_file(outside: &Path, target: &str) -> std::path::PathBuf {
    if target == "a.txt" {
        outside.join("secret.txt")
    } else {
        outside.join("d/a.txt")
    }
}

fn calls(target: &str) -> Vec<(&'static str, Box<dyn Tool>, Value)> {
    let edit = json!([{
        "type": "str_replace",
        "old_string": "OUTSIDE-SECRET",
        "new_string": "PWNED",
    }]);
    let patch = |body: String| json!({ "patch": format!("*** Begin Patch\n{body}*** End Patch") });
    vec![
        (
            "fs read",
            Box::new(FsTool::new()),
            json!({ "op": "read", "path": target }),
        ),
        (
            "fs write",
            Box::new(FsTool::new()),
            json!({ "op": "write", "path": target, "content": "PWNED\n", "force": true }),
        ),
        (
            "fs write append",
            Box::new(FsTool::new()),
            json!({ "op": "write", "path": target, "content": "PWNED\n", "append": true }),
        ),
        (
            "fs edit",
            Box::new(FsTool::new()),
            json!({ "op": "edit", "path": target, "operations": edit }),
        ),
        (
            "apply_patch update",
            Box::new(ApplyPatchTool::new()),
            patch(format!(
                "*** Update File: {target}\n@@\n-OUTSIDE-SECRET\n+PWNED\n"
            )),
        ),
    ]
}

/// The gate allows the call while `target` is a plain workspace file, the
/// file is then swapped for a link out, and the tool runs: it neither
/// returns nor changes the outside file.
#[tokio::test]
async fn a_link_swapped_in_after_the_gate_takes_no_tool_outside() {
    for target in ["a.txt", "d/a.txt"] {
        for (label, tool, args) in calls(target) {
            let (workspace, outside) = layout();
            let root = workspace.path().canonicalize().expect("canonical root");
            set_tool_workspace_root(Some(root.clone()));
            set_tool_host_fs_unlocked(false);
            let middleware = HookExecutionMiddleware::new(HookRegistry::new())
                .with_workspace(root.to_string_lossy())
                .with_overlay(ModeOverlay::AcceptEdits);
            let name = tool.name().to_string();
            let ctx = ToolExecutionContext {
                tool_name: &name,
                arguments: &args,
                call_id: "call-race",
            };
            middleware
                .before_execute(&ctx)
                .await
                .unwrap_or_else(|err| panic!("{label} {target}: the gate allows it: {err}"));

            swap(&root, outside.path(), target);
            let result = tool.execute(args.clone()).await;

            assert!(
                !result.content.contains("OUTSIDE-SECRET"),
                "{label} {target} returned the outside file: {}",
                result.content
            );
            assert_eq!(
                std::fs::read_to_string(outside_file(outside.path(), target))
                    .expect("outside file"),
                SECRET,
                "{label} {target} changed the outside file"
            );
            assert!(
                !result.success,
                "{label} {target} succeeded: {}",
                result.content
            );
        }
    }
    set_tool_workspace_root(None);
}

fn delete_calls(target: &str) -> Vec<(&'static str, Box<dyn Tool>, Value)> {
    let patch = |body: String| json!({ "patch": format!("*** Begin Patch\n{body}*** End Patch") });
    vec![
        (
            "fs delete",
            Box::new(FsTool::new()),
            json!({ "op": "delete", "path": target }),
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
                "*** Update File: {target}\n*** Move to: moved.txt\n@@\n-OUTSIDE-SECRET\n+OUTSIDE-SECRET\n"
            )),
        ),
    ]
}

/// The same swap before a delete or a move: the outside file stays. A link
/// swapped in for the file itself may be removed, since that removes only
/// the link; a link swapped in for a directory on the way is refused.
#[tokio::test]
async fn a_link_swapped_in_after_the_gate_deletes_nothing_outside() {
    for target in ["a.txt", "d/a.txt"] {
        for (label, tool, args) in delete_calls(target) {
            let (workspace, outside) = layout();
            let root = workspace.path().canonicalize().expect("canonical root");
            set_tool_workspace_root(Some(root.clone()));
            set_tool_host_fs_unlocked(false);
            let middleware = HookExecutionMiddleware::new(HookRegistry::new())
                .with_workspace(root.to_string_lossy())
                .with_overlay(ModeOverlay::AcceptEdits);
            let name = tool.name().to_string();
            let ctx = ToolExecutionContext {
                tool_name: &name,
                arguments: &args,
                call_id: "call-race-delete",
            };
            middleware
                .before_execute(&ctx)
                .await
                .unwrap_or_else(|err| panic!("{label} {target}: the gate allows it: {err}"));

            swap(&root, outside.path(), target);
            let result = tool.execute(args.clone()).await;

            assert_eq!(
                std::fs::read_to_string(outside_file(outside.path(), target))
                    .ok()
                    .as_deref(),
                Some(SECRET),
                "{label} {target} removed or changed the outside file: {}",
                result.content
            );
            assert!(
                !result.content.contains("OUTSIDE-SECRET"),
                "{label} {target} returned the outside file: {}",
                result.content
            );
            if target == "d/a.txt" {
                assert!(
                    !result.success,
                    "{label} {target} succeeded: {}",
                    result.content
                );
            }
        }
    }
    set_tool_workspace_root(None);
}
