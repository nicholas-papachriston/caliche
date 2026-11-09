//! Integration coverage for Goose-style tool inspection.

use caliche::agent::hooks::{
    HookMatcher, HookRegistry, InspectionHookAdapter, PermissionMode, ToolInspectorRequest,
    default_inspection_manager,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::test]
async fn inspection_hook_blocks_deny_all_via_registry() {
    let manager = Arc::new(default_inspection_manager(PermissionMode::DenyAll, None));
    let mut hooks = HookRegistry::new();
    hooks.register(
        HookMatcher::All,
        Arc::new(InspectionHookAdapter::new(manager)),
    );
    let matching = hooks.get_matching("bash");
    assert_eq!(matching.len(), 1);
    let args = json!({"command":"echo hi"});
    let ctx = caliche::agent::hooks::HookContext::new("bash", &args, "c1");
    let decision = matching[0].on_pre_tool_use(&ctx).await;
    assert!(decision.is_denied());
}

#[tokio::test]
async fn path_security_inspector_allows_relative_workspace_paths() {
    let root = PathBuf::from("/tmp/ws");
    let manager = default_inspection_manager(PermissionMode::AcceptAll, Some(root));
    let args = json!({"op":"read","path":"lib.rs"});
    let request = ToolInspectorRequest {
        tool_name: "fs",
        arguments: &args,
        call_id: "x",
    };
    let action = manager.inspect_one(&request).await.expect("inspect");
    assert_eq!(action, caliche::agent::hooks::InspectionAction::Allow);
}
