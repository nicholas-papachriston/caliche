//! Python bindings for agent lifecycle hooks

use crate::python_async::call_python_maybe_async;
use async_trait::async_trait;
use caliche::agent::hooks::lifecycle::{HookContext, HookResult, LifecycleHook};
use caliche::agent::hooks::matcher::HookMatcher;
use caliche::agent::hooks::permission::{PermissionDecision, PermissionMode};
use caliche::agent::tools::tool::ToolResult;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use pythonize::{depythonize, pythonize};
use serde_json::Value;

fn pre_tool_use_fail_closed(reason: impl Into<String>) -> HookResult {
    let reason = reason.into();
    tracing::warn!("Python pre_tool_use hook failed closed: {reason}");
    HookResult::deny(reason)
}

fn parse_permission_decision(value: &Bound<'_, PyAny>) -> PyResult<PermissionDecision> {
    value.extract::<String>().map_or_else(
        |_| {
            let dict = value.cast::<PyDict>().map_err(|_| {
                PyRuntimeError::new_err("Permission decision must be a string or dict")
            })?;
            let decision = dict
                .get_item("decision")?
                .ok_or_else(|| {
                    PyRuntimeError::new_err("Permission decision dict missing 'decision' key")
                })?
                .extract::<String>()
                .map_err(|_| PyRuntimeError::new_err("Invalid permission decision format"))?;
            match decision.to_lowercase().as_str() {
                "allow" => Ok(PermissionDecision::Allow),
                "deny" => {
                    let reason = dict
                        .get_item("reason")?
                        .and_then(|reason| reason.extract::<String>().ok())
                        .unwrap_or_else(|| "Denied by hook".to_string());
                    Ok(PermissionDecision::Deny {
                        reason: reason.into(),
                    })
                }
                _ => Err(PyRuntimeError::new_err(format!(
                    "Invalid permission decision: {decision}"
                ))),
            }
        },
        |decision| match decision.to_lowercase().as_str() {
            "allow" => Ok(PermissionDecision::Allow),
            "deny" => Err(PyRuntimeError::new_err(
                "Deny decision requires a reason. Use HookResult.deny() instead.",
            )),
            _ => Err(PyRuntimeError::new_err(format!(
                "Invalid permission decision: {decision}"
            ))),
        },
    )
}

/// Python wrapper for `PermissionMode` enum
#[pyclass]
#[derive(Clone, Copy)]
pub struct PyPermissionMode {
    inner: PermissionMode,
}

#[pymethods]
impl PyPermissionMode {
    #[new]
    fn new(mode: &str) -> PyResult<Self> {
        let inner = match mode.to_lowercase().as_str() {
            "default" => PermissionMode::Default,
            "accept-edits" => PermissionMode::AcceptEdits,
            "accept-all" => PermissionMode::AcceptAll,
            "deny-all" => PermissionMode::DenyAll,
            "interactive" => PermissionMode::Interactive,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid permission mode: {mode}. Must be one of: default, accept-edits, accept-all, deny-all, interactive"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    const fn default() -> Self {
        Self {
            inner: PermissionMode::Default,
        }
    }

    #[staticmethod]
    const fn accept_edits() -> Self {
        Self {
            inner: PermissionMode::AcceptEdits,
        }
    }

    #[staticmethod]
    const fn accept_all() -> Self {
        Self {
            inner: PermissionMode::AcceptAll,
        }
    }

    #[staticmethod]
    const fn deny_all() -> Self {
        Self {
            inner: PermissionMode::DenyAll,
        }
    }

    #[staticmethod]
    const fn interactive() -> Self {
        Self {
            inner: PermissionMode::Interactive,
        }
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "PyO3 protocol methods require borrowed Python instance receivers"
    )]
    fn __str__(&self) -> String {
        match self.inner {
            PermissionMode::Default => "Default",
            PermissionMode::AcceptEdits => "AcceptEdits",
            PermissionMode::AcceptAll => "AcceptAll",
            PermissionMode::DenyAll => "DenyAll",
            PermissionMode::Interactive => "Interactive",
        }
        .to_string()
    }
}

impl From<PyPermissionMode> for PermissionMode {
    fn from(py_mode: PyPermissionMode) -> Self {
        py_mode.inner
    }
}

/// Python wrapper for `HookMatcher` enum
#[pyclass]
#[derive(Clone)]
pub struct PyHookMatcher {
    inner: HookMatcher,
}

#[pymethods]
impl PyHookMatcher {
    #[new]
    fn new(matcher_type: &str, tool_name: Option<String>) -> PyResult<Self> {
        let inner = match matcher_type.to_lowercase().as_str() {
            "tool" => {
                let name = tool_name.ok_or_else(|| {
                    PyRuntimeError::new_err("Tool matcher requires a tool_name parameter")
                })?;
                HookMatcher::Tool { name: name.into() }
            }
            "all" => HookMatcher::All,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid matcher type: {matcher_type}. Must be 'tool' or 'all'"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    fn tool(tool_name: String) -> Self {
        Self {
            inner: HookMatcher::Tool {
                name: tool_name.into(),
            },
        }
    }

    #[staticmethod]
    const fn all() -> Self {
        Self {
            inner: HookMatcher::All,
        }
    }
}

impl From<PyHookMatcher> for HookMatcher {
    fn from(py_matcher: PyHookMatcher) -> Self {
        py_matcher.inner
    }
}

/// Python wrapper for `HookContext` (owned data for FFI)
#[pyclass]
pub struct PyHookContext {
    name: String,
    input: serde_json::Value,
    use_id: String,
}

#[pymethods]
impl PyHookContext {
    #[new]
    fn new(
        tool_name: String,
        tool_input: &Bound<'_, PyAny>,
        tool_use_id: String,
    ) -> PyResult<Self> {
        let py_input = depythonize(tool_input)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse tool_input: {e}")))?;

        Ok(Self {
            name: tool_name,
            input: py_input,
            use_id: tool_use_id,
        })
    }

    #[getter]
    fn tool_name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn tool_input(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(pythonize(py, &self.input)?.unbind())
    }

    #[getter]
    fn tool_use_id(&self) -> String {
        self.use_id.clone()
    }
}

impl From<&HookContext<'_>> for PyHookContext {
    fn from(ctx: &HookContext<'_>) -> Self {
        let (tool_name, tool_input, tool_use_id) = ctx.to_owned();
        Self {
            name: tool_name,
            input: tool_input,
            use_id: tool_use_id,
        }
    }
}

/// Python wrapper for `HookResult`
#[pyclass]
pub struct PyHookResult {
    inner: HookResult,
}

#[pymethods]
impl PyHookResult {
    #[new]
    fn new(
        permission_decision: &Bound<'_, PyAny>,
        hook_specific_output: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let decision = parse_permission_decision(permission_decision)?;

        let output = hook_specific_output
            .map(|output_obj| {
                depythonize(output_obj).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to parse hook output: {e}"))
                })
            })
            .transpose()?;

        Ok(Self {
            inner: HookResult::with_output(decision, output.unwrap_or(Value::Null)),
        })
    }

    #[staticmethod]
    fn allow() -> Self {
        Self {
            inner: HookResult::allow(),
        }
    }

    #[staticmethod]
    fn deny(reason: String) -> Self {
        Self {
            inner: HookResult::deny(reason),
        }
    }

    #[getter]
    fn permission_decision(&self, py: Python) -> PyResult<Py<PyAny>> {
        let decision_dict = PyDict::new(py);
        match &self.inner.permission_decision {
            PermissionDecision::Allow => {
                decision_dict.set_item("decision", "allow")?;
            }
            PermissionDecision::Deny { reason } => {
                decision_dict.set_item("decision", "deny")?;
                decision_dict.set_item("reason", reason.as_ref())?;
            }
            PermissionDecision::RequireApproval {
                reason,
                proposed_amendment,
            } => {
                decision_dict.set_item("decision", "require_approval")?;
                decision_dict.set_item("reason", reason.as_ref())?;
                if let Some(prefix) = proposed_amendment {
                    let list = PyList::empty(py);
                    for s in prefix {
                        list.append(s.as_str())?;
                    }
                    decision_dict.set_item("proposed_amendment", list)?;
                }
            }
        }
        Ok(decision_dict.into())
    }

    #[getter]
    fn hook_specific_output(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .hook_specific_output
            .as_ref()
            .map(|output| -> PyResult<Py<PyAny>> { Ok(pythonize(py, output)?.unbind()) })
            .transpose()
    }
}

impl From<HookResult> for PyHookResult {
    fn from(result: HookResult) -> Self {
        Self { inner: result }
    }
}

/// Python wrapper for `LifecycleHook` trait
///
/// This allows Python objects to implement the `LifecycleHook` interface
/// by calling Python methods from Rust.
#[derive(Debug)]
pub struct PyLifecycleHook {
    hook_obj: Py<PyAny>,
}

impl PyLifecycleHook {
    #[must_use]
    pub const fn new(hook_obj: Py<PyAny>) -> Self {
        Self { hook_obj }
    }
}

#[async_trait]
impl LifecycleHook for PyLifecycleHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        let py_ctx = PyHookContext::from(ctx);
        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let py_ctx_obj = Py::new(py, py_ctx)?;
            let method = self.hook_obj.getattr(py, "on_pre_tool_use")?;
            method.call1(py, (py_ctx_obj,))
        });

        match result {
            Ok(result_obj) => {
                let final_result = call_python_maybe_async(result_obj).await;
                match final_result {
                    Ok(py_result) => Python::attach(|py| {
                        let bound = py_result.bind(py);
                        if let Ok(py_hook_result) = bound.extract::<PyRef<PyHookResult>>() {
                            py_hook_result.inner.clone()
                        } else if let Ok(dict) = bound.cast::<pyo3::types::PyDict>() {
                            // Try to construct from dict
                            let decision = if let Ok(Some(decision_obj)) = dict.get_item("decision")
                            {
                                if let Ok(decision_str) = decision_obj.extract::<String>() {
                                    match decision_str.to_lowercase().as_str() {
                                        "allow" => PermissionDecision::Allow,
                                        "deny" => {
                                            let reason = dict
                                                .get_item("reason")
                                                .ok()
                                                .flatten()
                                                .and_then(|r| r.extract::<String>().ok())
                                                .unwrap_or_else(|| "Denied by hook".to_string());
                                            PermissionDecision::Deny {
                                                reason: reason.into(),
                                            }
                                        }
                                        other => {
                                            return pre_tool_use_fail_closed(format!(
                                                "Invalid pre_tool_use hook decision: {other}"
                                            ));
                                        }
                                    }
                                } else {
                                    return pre_tool_use_fail_closed(
                                        "pre_tool_use hook decision is not a string",
                                    );
                                }
                            } else {
                                return pre_tool_use_fail_closed(
                                    "pre_tool_use hook result missing decision",
                                );
                            };
                            HookResult {
                                permission_decision: decision,
                                hook_specific_output: dict
                                    .get_item("hook_specific_output")
                                    .ok()
                                    .flatten()
                                    .and_then(|o| depythonize(o.as_ref()).ok()),
                            }
                        } else {
                            pre_tool_use_fail_closed(
                                "pre_tool_use hook returned an unparseable result",
                            )
                        }
                    }),
                    Err(e) => pre_tool_use_fail_closed(format!(
                        "Python pre_tool_use hook async error: {e}"
                    )),
                }
            }
            Err(e) => {
                pre_tool_use_fail_closed(format!("Python pre_tool_use hook invocation error: {e}"))
            }
        }
    }

    async fn on_post_tool_use(&self, _ctx: &HookContext<'_>, _result: &ToolResult) -> HookResult {
        // Try to call the Python method, but default to allow if it doesn't exist
        let py_ctx = PyHookContext::from(_ctx);
        // Convert Rust ToolResult to Python ToolResult
        let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let tool_result = crate::tools::ToolResult::new(
                _result.success,
                _result.content.clone(),
                _result.error.clone(),
            );
            Ok(Py::new(py, tool_result)?.into_any())
        });

        let Ok(py_result_obj) = py_result else {
            return HookResult::allow();
        };

        let result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let py_ctx_obj = Py::new(py, py_ctx)?;
            let method = self.hook_obj.getattr(py, "on_post_tool_use")?;
            method.call1(py, (py_ctx_obj, py_result_obj))
        });

        match result {
            Ok(result_obj) => {
                let final_result = call_python_maybe_async(result_obj).await;
                final_result.map_or_else(
                    |_| HookResult::allow(),
                    |py_result| {
                        Python::attach(|py| {
                            let bound = py_result.bind(py);
                            bound
                                .extract::<PyRef<PyHookResult>>()
                                .map_or_else(|_| HookResult::allow(), |result| result.inner.clone())
                        })
                    },
                )
            }
            Err(_) => HookResult::allow(), // Default to allow if method doesn't exist
        }
    }
}
