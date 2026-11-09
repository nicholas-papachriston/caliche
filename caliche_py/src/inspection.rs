//! Python bindings for Goose-style tool inspection.

use crate::get_runtime;
use crate::hooks::PyPermissionMode;
use caliche::agent::hooks::permission::PermissionMode;
use caliche::agent::hooks::{
    InspectionAction, InspectionHookAdapter, InspectionResult, PathSecurityInspector,
    PermissionModeInspector, ToolInspectionManager, ToolInspector, ToolInspectorRequest,
    default_inspection_manager,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::depythonize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Action recommended by an inspector.
#[pyclass(name = "InspectionAction")]
#[derive(Clone)]
pub struct PyInspectionAction {
    inner: InspectionAction,
}

#[pymethods]
impl PyInspectionAction {
    #[staticmethod]
    const fn allow() -> Self {
        Self {
            inner: InspectionAction::Allow,
        }
    }

    #[staticmethod]
    const fn deny(reason: String) -> Self {
        Self {
            inner: InspectionAction::Deny { reason },
        }
    }

    #[staticmethod]
    const fn require_approval(reason: String) -> Self {
        Self {
            inner: InspectionAction::RequireApproval { reason },
        }
    }

    const fn kind(&self) -> &'static str {
        match self.inner {
            InspectionAction::Allow => "allow",
            InspectionAction::Deny { .. } => "deny",
            InspectionAction::RequireApproval { .. } => "require_approval",
        }
    }

    fn reason(&self) -> Option<String> {
        match &self.inner {
            InspectionAction::Allow => None,
            InspectionAction::Deny { reason } | InspectionAction::RequireApproval { reason } => {
                Some(reason.clone())
            }
        }
    }

    fn __str__(&self) -> String {
        match &self.inner {
            InspectionAction::Allow => "Allow".to_string(),
            InspectionAction::Deny { reason } => format!("Deny({reason})"),
            InspectionAction::RequireApproval { reason } => {
                format!("RequireApproval({reason})")
            }
        }
    }
}

impl From<InspectionAction> for PyInspectionAction {
    fn from(inner: InspectionAction) -> Self {
        Self { inner }
    }
}

/// Result produced by a single inspector.
#[pyclass(name = "InspectionResult")]
#[derive(Clone)]
pub struct PyInspectionResult {
    inner: InspectionResult,
}

#[pymethods]
impl PyInspectionResult {
    #[getter]
    fn call_id(&self) -> String {
        self.inner.call_id.clone()
    }

    #[getter]
    const fn inspector_name(&self) -> &'static str {
        self.inner.inspector_name
    }

    #[getter]
    fn reason(&self) -> String {
        self.inner.reason.clone()
    }

    #[getter]
    const fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    #[getter]
    fn action(&self) -> PyInspectionAction {
        self.inner.action.clone().into()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("call_id", self.call_id())?;
        dict.set_item("inspector_name", self.inspector_name())?;
        dict.set_item("reason", self.reason())?;
        dict.set_item("confidence", self.confidence())?;
        dict.set_item("action", self.action().kind())?;
        if let Some(reason) = self.action().reason() {
            dict.set_item("action_reason", reason)?;
        }
        Ok(dict.into())
    }
}

/// Path security inspector.
#[pyclass(name = "PathSecurityInspector")]
pub struct PyPathSecurityInspector {
    inner: PathSecurityInspector,
}

#[pymethods]
impl PyPathSecurityInspector {
    #[new]
    fn new(workspace_root: String) -> Self {
        Self {
            inner: PathSecurityInspector::new(PathBuf::from(workspace_root)),
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "Python instance method ABI requires a receiver"
    )]
    const fn name(&self) -> &'static str {
        "path_security"
    }

    fn inspect(
        &self,
        py: Python<'_>,
        tool_name: &str,
        arguments: &Bound<'_, PyAny>,
        call_id: &str,
    ) -> PyResult<Vec<PyInspectionResult>> {
        inspect_with(py, &self.inner, tool_name, arguments, call_id)
    }
}

/// Permission-mode inspector.
#[pyclass(name = "PermissionModeInspector")]
pub struct PyPermissionModeInspector {
    inner: PermissionModeInspector,
}

#[pymethods]
impl PyPermissionModeInspector {
    #[new]
    fn new(mode: PyPermissionMode) -> Self {
        Self {
            inner: PermissionModeInspector::new(mode.into()),
        }
    }

    #[expect(
        clippy::unused_self,
        reason = "Python instance method ABI requires a receiver"
    )]
    const fn name(&self) -> &'static str {
        "permission_mode"
    }

    fn inspect(
        &self,
        py: Python<'_>,
        tool_name: &str,
        arguments: &Bound<'_, PyAny>,
        call_id: &str,
    ) -> PyResult<Vec<PyInspectionResult>> {
        inspect_with(py, &self.inner, tool_name, arguments, call_id)
    }
}

/// Coordinates inspectors and merges recommendations.
///
/// Configure with [`Self::add_path_security`] / [`Self::add_permission_mode`], then
/// call [`Self::inspect_one`] or [`Self::as_hook`].
#[pyclass(name = "ToolInspectionManager")]
pub struct PyToolInspectionManager {
    path_root: Option<PathBuf>,
    permission_mode: Option<PermissionMode>,
    materialized: OnceLock<Arc<ToolInspectionManager>>,
}

impl PyToolInspectionManager {
    fn materialize(&self) -> Arc<ToolInspectionManager> {
        self.materialized
            .get_or_init(|| {
                let mut manager = ToolInspectionManager::new();
                if let Some(root) = &self.path_root {
                    manager.add_inspector(Box::new(PathSecurityInspector::new(root.clone())));
                }
                if let Some(mode) = self.permission_mode {
                    manager.add_inspector(Box::new(PermissionModeInspector::new(mode)));
                }
                Arc::new(manager)
            })
            .clone()
    }
}

#[pymethods]
impl PyToolInspectionManager {
    #[new]
    const fn new() -> Self {
        Self {
            path_root: None,
            permission_mode: None,
            materialized: OnceLock::new(),
        }
    }

    fn add_path_security(mut slf: PyRefMut<'_, Self>, workspace_root: String) -> PyResult<()> {
        if slf.materialized.get().is_some() {
            return Err(PyRuntimeError::new_err(
                "ToolInspectionManager is frozen after inspect/as_hook; create a new manager",
            ));
        }
        slf.path_root = Some(PathBuf::from(workspace_root));
        Ok(())
    }

    fn add_permission_mode(mut slf: PyRefMut<'_, Self>, mode: PyPermissionMode) -> PyResult<()> {
        if slf.materialized.get().is_some() {
            return Err(PyRuntimeError::new_err(
                "ToolInspectionManager is frozen after inspect/as_hook; create a new manager",
            ));
        }
        slf.permission_mode = Some(mode.into());
        Ok(())
    }

    fn inspector_names(&self) -> Vec<&'static str> {
        self.materialize().inspector_names()
    }

    fn inspect_one(
        &self,
        py: Python<'_>,
        tool_name: &str,
        arguments: &Bound<'_, PyAny>,
        call_id: &str,
    ) -> PyResult<PyInspectionAction> {
        let args = depythonize_args(arguments)?;
        let request = ToolInspectorRequest {
            tool_name,
            arguments: &args,
            call_id,
        };
        let manager = self.materialize();
        let action = py
            .detach(|| {
                let rt = get_runtime().map_err(|e| e.to_string())?;
                rt.block_on(async { manager.inspect_one(&request).await })
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(action.into())
    }

    /// Wrap this manager as a lifecycle hook for `AgentBuilder.hook(...)`.
    fn as_hook(&self) -> PyInspectionHook {
        PyInspectionHook {
            inner: Arc::new(InspectionHookAdapter::new(self.materialize())),
        }
    }
}

/// Lifecycle-hook wrapper around an inspection manager.
#[pyclass(name = "InspectionHookAdapter")]
#[derive(Clone)]
pub struct PyInspectionHook {
    pub(crate) inner: Arc<InspectionHookAdapter>,
}

#[pymethods]
impl PyInspectionHook {
    #[new]
    fn new(manager: &PyToolInspectionManager) -> Self {
        manager.as_hook()
    }
}

impl PyInspectionHook {
    pub(crate) fn as_lifecycle_hook(&self) -> Arc<dyn caliche::agent::hooks::LifecycleHook> {
        Arc::clone(&self.inner) as Arc<dyn caliche::agent::hooks::LifecycleHook>
    }
}

/// Build the default inspection chain.
#[pyfunction]
#[pyo3(name = "default_inspection_manager", signature = (permission_mode, workspace_root=None))]
pub fn py_default_inspection_manager(
    permission_mode: PyPermissionMode,
    workspace_root: Option<String>,
) -> PyToolInspectionManager {
    let mode: PermissionMode = permission_mode.into();
    let root = workspace_root.map(PathBuf::from);
    let manager = default_inspection_manager(mode, root.clone());
    let py_manager = PyToolInspectionManager {
        path_root: root,
        permission_mode: Some(mode),
        materialized: OnceLock::new(),
    };
    let _ = py_manager.materialized.set(Arc::new(manager));
    py_manager
}

fn depythonize_args(arguments: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    depythonize(arguments)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid tool arguments: {e}")))
}

fn inspect_with(
    py: Python<'_>,
    inspector: &dyn ToolInspector,
    tool_name: &str,
    arguments: &Bound<'_, PyAny>,
    call_id: &str,
) -> PyResult<Vec<PyInspectionResult>> {
    let args = depythonize_args(arguments)?;
    let request = ToolInspectorRequest {
        tool_name,
        arguments: &args,
        call_id,
    };
    let results = py
        .detach(|| {
            let rt = get_runtime().map_err(|e| e.to_string())?;
            rt.block_on(async { inspector.inspect(&request).await })
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
    Ok(results
        .into_iter()
        .map(|inner| PyInspectionResult { inner })
        .collect())
}
