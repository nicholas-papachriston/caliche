//! Integration coverage for mutation style and apply_patch XOR.

use caliche::agent::service::ToolRegistryBuilder;
use caliche::agent::tools::{ApplyPatchTool, FsTool, MutationStyle, Tool, apply_hunks_to_cwd};
use caliche::llm::ProviderType;
use serde_json::json;
use tempfile::tempdir;

#[tokio::test]
async fn apply_patch_tool_mutates_files() {
    let dir = tempdir().expect("tempdir");
    let target = dir.path().join("note.txt");
    let patch = format!(
        "*** Begin Patch\n*** Add File: {}\n+alpha\n+beta\n*** End Patch\n",
        target.display()
    );
    let tool = ApplyPatchTool::new();
    let result = tool.execute(json!({ "patch": patch })).await;
    assert!(result.success, "{:?}", result.error);
    let body = std::fs::read_to_string(&target).expect("read");
    assert_eq!(body, "alpha\nbeta\n");
}

#[tokio::test]
async fn fs_apply_patch_style_rejects_write_ops() {
    let tool = FsTool::with_mutation_style(MutationStyle::ApplyPatch);
    let result = tool
        .execute(json!({
            "op": "write",
            "path": "x.txt",
            "content": "nope"
        }))
        .await;
    assert!(!result.success);
    assert!(
        result
            .error
            .as_deref()
            .is_some_and(|e| e.contains("apply_patch"))
    );
}

#[tokio::test]
async fn registry_builder_xor_is_consistent_at_runtime() {
    let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
        .with_mutation_style(MutationStyle::ApplyPatch)
        .with_tool_names(Some(vec!["fs".into(), "bash".into()]))
        .build()
        .await
        .expect("build");

    assert!(registry.get("apply_patch").is_some());
    let fs = registry.get("fs").expect("fs");
    let denied = fs
        .execute(json!({"op":"edit","path":"a","operations":[]}))
        .await;
    assert!(!denied.success);
}

#[tokio::test]
async fn apply_hunks_update_relative_to_cwd() {
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("f.txt"), "old\n").expect("write");
    let patch = "*** Begin Patch\n*** Update File: f.txt\n@@\n-old\n+new\n*** End Patch\n";
    apply_hunks_to_cwd(patch, Some(dir.path()))
        .await
        .expect("apply");
    let body = std::fs::read_to_string(dir.path().join("f.txt")).expect("read");
    assert_eq!(body, "new\n");
}
