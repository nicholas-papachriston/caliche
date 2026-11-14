//! A tool runs only with the gate's permit, and the gate refuses a denied call.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use caliche::agent::hooks::{HookContext, HookMatcher, HookRegistry, HookResult, LifecycleHook};
use caliche::agent::tools::execution::HookExecutionMiddleware;
use caliche::agent::tools::{Tool, ToolMeta, ToolPermit, ToolRegistry, ToolResult};
use serde_json::{Value, json};

#[derive(Debug, Default)]
struct CountingTool {
    runs: AtomicUsize,
}

impl ToolMeta for CountingTool {
    fn name(&self) -> &str {
        "counting"
    }

    fn description(&self) -> &str {
        "Counts its runs."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
}

#[async_trait]
impl Tool for CountingTool {
    async fn execute(&self, _arguments: Value, _permit: &ToolPermit<'_>) -> ToolResult {
        self.runs.fetch_add(1, Ordering::SeqCst);
        ToolResult::success("ran")
    }
}

#[derive(Debug)]
struct DenyHook;

#[async_trait]
impl LifecycleHook for DenyHook {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::deny("operator denied counting")
    }
}

fn denying_gate() -> HookExecutionMiddleware {
    let mut hooks = HookRegistry::new();
    hooks.register(HookMatcher::tool("counting"), Arc::new(DenyHook));
    HookExecutionMiddleware::new(hooks)
}

#[tokio::test]
async fn a_denied_call_by_name_does_not_run_the_tool() {
    let tool = Arc::new(CountingTool::default());
    let tools = ToolRegistry::new();
    tools.register(Arc::clone(&tool) as Arc<dyn Tool>);

    let result = denying_gate().run(&tools, "counting", json!({})).await;

    assert!(!result.success);
    assert_eq!(
        result.error.as_deref(),
        Some("Tool 'counting' was refused: operator denied counting")
    );
    assert_eq!(tool.runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_denied_call_on_a_tool_does_not_run_it() {
    let tool = CountingTool::default();

    let result = denying_gate().run_tool(&tool, json!({})).await;

    assert!(!result.success);
    assert_eq!(
        result.error.as_deref(),
        Some("Tool 'counting' was refused: operator denied counting")
    );
    assert_eq!(tool.runs.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_admitted_call_runs_the_tool_once() {
    let tool = CountingTool::default();
    let gate = HookExecutionMiddleware::new(HookRegistry::new());

    let result = gate.run_tool(&tool, json!({})).await;

    assert!(result.success);
    assert_eq!(result.content, "ran");
    assert_eq!(tool.runs.load(Ordering::SeqCst), 1);
}
