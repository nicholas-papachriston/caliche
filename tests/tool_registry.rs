//! Integration coverage for active/deferred tool registry controls.

use caliche::agent::tools::{ApplyPatchTool, CalculatorTool, ToolRegistry};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn inactive_tools_are_hidden_from_schemas_and_execution() {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool));
    registry.register_deferred(Arc::new(ApplyPatchTool::new()));

    assert_eq!(registry.get_tool_schemas().len(), 1);
    let inactive = registry
        .execute(
            "apply_patch",
            json!({
                "patch": "*** Begin Patch\n*** Add File: /tmp/x\n+y\n*** End Patch\n"
            }),
        )
        .await;
    assert!(!inactive.success);

    registry.activate(["apply_patch"]);
    assert_eq!(registry.get_tool_schemas().len(), 2);
}
