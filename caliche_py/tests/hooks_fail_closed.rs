use caliche::agent::hooks::lifecycle::{HookContext, LifecycleHook};
use caliche::agent::hooks::permission::PermissionDecision;
use caliche_py::hooks::PyLifecycleHook;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::json;

#[tokio::test]
async fn pre_tool_use_missing_decision_fails_closed() {
    Python::initialize();

    let hook = Python::attach(|py| -> PyResult<PyLifecycleHook> {
        let module = PyModule::from_code(
            py,
            c"
class MissingDecisionHook:
    def on_pre_tool_use(self, ctx):
        return {}
",
            c"fail_closed_hook.py",
            c"fail_closed_hook",
        )?;
        let hook_obj = module.getattr("MissingDecisionHook")?.call0()?;
        Ok(PyLifecycleHook::new(hook_obj.unbind()))
    })
    .expect("python hook");

    let tool_input = json!({"cmd": "echo hi"});
    let ctx = HookContext::new("bash", &tool_input, "tool-use-1");
    let result = hook.on_pre_tool_use(&ctx).await;

    match result.permission_decision {
        PermissionDecision::Deny { reason } => {
            assert!(reason.contains("missing decision"));
        }
        other => panic!("expected deny, got {other:?}"),
    }
}

#[tokio::test]
async fn pre_tool_use_invalid_decision_fails_closed() {
    Python::initialize();

    let hook = Python::attach(|py| -> PyResult<PyLifecycleHook> {
        let module = PyModule::from_code(
            py,
            c"
class InvalidDecisionHook:
    def on_pre_tool_use(self, ctx):
        return {'decision': 'maybe'}
",
            c"invalid_decision_hook.py",
            c"invalid_decision_hook",
        )?;
        let hook_obj = module.getattr("InvalidDecisionHook")?.call0()?;
        Ok(PyLifecycleHook::new(hook_obj.unbind()))
    })
    .expect("python hook");

    let tool_input = json!({});
    let ctx = HookContext::new("bash", &tool_input, "tool-use-2");
    let result = hook.on_pre_tool_use(&ctx).await;

    match result.permission_decision {
        PermissionDecision::Deny { reason } => {
            assert!(reason.contains("Invalid pre_tool_use hook decision"));
        }
        other => panic!("expected deny, got {other:?}"),
    }
}

#[tokio::test]
async fn pre_tool_use_explicit_allow_permits_tool() {
    Python::initialize();

    let hook = Python::attach(|py| -> PyResult<PyLifecycleHook> {
        let module = PyModule::from_code(
            py,
            c"
class AllowHook:
    def on_pre_tool_use(self, ctx):
        return {'decision': 'allow'}
",
            c"allow_hook.py",
            c"allow_hook",
        )?;
        let hook_obj = module.getattr("AllowHook")?.call0()?;
        Ok(PyLifecycleHook::new(hook_obj.unbind()))
    })
    .expect("python hook");

    let tool_input = json!({});
    let ctx = HookContext::new("bash", &tool_input, "tool-use-3");
    let result = hook.on_pre_tool_use(&ctx).await;
    assert!(matches!(
        result.permission_decision,
        PermissionDecision::Allow
    ));
}
