//! JavaScript bindings for Goose-style tool inspection.

use crate::get_runtime;
use crate::hooks::JsPermissionMode;
use caliche::agent::hooks::permission::PermissionMode;
use caliche::agent::hooks::{
    InspectionAction, InspectionHookAdapter, InspectionResult, PathSecurityInspector,
    PermissionModeInspector, ToolInspectionManager, ToolInspector, ToolInspectorRequest,
    default_inspection_manager as rust_default_inspection_manager,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Action recommended by an inspector.
#[napi(js_name = "InspectionAction")]
#[derive(Clone)]
pub struct JsInspectionAction {
    inner: InspectionAction,
}

#[napi]
impl JsInspectionAction {
    #[napi(factory, catch_unwind)]
    pub const fn allow() -> Self {
        Self {
            inner: InspectionAction::Allow,
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn deny(reason: String) -> Self {
        Self {
            inner: InspectionAction::Deny { reason },
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn require_approval(reason: String) -> Self {
        Self {
            inner: InspectionAction::RequireApproval { reason },
        }
    }

    #[napi(catch_unwind)]
    pub const fn kind(&self) -> &'static str {
        match self.inner {
            InspectionAction::Allow => "allow",
            InspectionAction::Deny { .. } => "deny",
            InspectionAction::RequireApproval { .. } => "require_approval",
        }
    }

    #[napi(catch_unwind)]
    pub fn reason(&self) -> Option<String> {
        match &self.inner {
            InspectionAction::Allow => None,
            InspectionAction::Deny { reason } | InspectionAction::RequireApproval { reason } => {
                Some(reason.clone())
            }
        }
    }

    #[napi(catch_unwind)]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        match &self.inner {
            InspectionAction::Allow => "Allow".to_string(),
            InspectionAction::Deny { reason } => format!("Deny({reason})"),
            InspectionAction::RequireApproval { reason } => {
                format!("RequireApproval({reason})")
            }
        }
    }
}

impl From<InspectionAction> for JsInspectionAction {
    fn from(inner: InspectionAction) -> Self {
        Self { inner }
    }
}

/// Result produced by a single inspector.
#[napi(js_name = "InspectionResult")]
#[derive(Clone)]
pub struct JsInspectionResult {
    inner: InspectionResult,
}

#[napi]
impl JsInspectionResult {
    #[napi(getter, catch_unwind)]
    pub fn call_id(&self) -> String {
        self.inner.call_id.clone()
    }

    #[napi(getter, catch_unwind)]
    pub const fn inspector_name(&self) -> &'static str {
        self.inner.inspector_name
    }

    #[napi(getter, catch_unwind)]
    pub fn reason(&self) -> String {
        self.inner.reason.clone()
    }

    #[napi(getter, catch_unwind)]
    pub fn confidence(&self) -> f64 {
        f64::from(self.inner.confidence)
    }

    #[napi(getter, catch_unwind)]
    pub fn action(&self) -> JsInspectionAction {
        self.inner.action.clone().into()
    }

    #[napi(catch_unwind)]
    pub fn to_object(&self) -> Value {
        let mut obj = serde_json::json!({
            "call_id": self.call_id(),
            "inspector_name": self.inspector_name(),
            "reason": self.reason(),
            "confidence": self.confidence(),
            "action": self.action().kind(),
        });
        if let Some(reason) = self.action().reason() {
            obj["action_reason"] = Value::String(reason);
        }
        obj
    }
}

/// Path security inspector.
#[napi(js_name = "PathSecurityInspector")]
pub struct JsPathSecurityInspector {
    inner: PathSecurityInspector,
}

#[napi]
impl JsPathSecurityInspector {
    #[napi(constructor, catch_unwind)]
    pub fn new(workspace_root: String) -> Self {
        Self {
            inner: PathSecurityInspector::new(PathBuf::from(workspace_root)),
        }
    }

    #[napi(catch_unwind)]
    pub const fn name(&self) -> &'static str {
        "path_security"
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn inspect(
        &self,
        tool_name: String,
        arguments: Value,
        call_id: String,
    ) -> napi::Result<Vec<JsInspectionResult>> {
        inspect_with(&self.inner, &tool_name, &arguments, &call_id)
    }
}

/// Permission-mode inspector.
#[napi(js_name = "PermissionModeInspector")]
pub struct JsPermissionModeInspector {
    inner: PermissionModeInspector,
}

#[napi]
impl JsPermissionModeInspector {
    #[napi(constructor, catch_unwind)]
    pub fn new(mode: &JsPermissionMode) -> Self {
        Self {
            inner: PermissionModeInspector::new(PermissionMode::from(*mode)),
        }
    }

    #[napi(catch_unwind)]
    pub const fn name(&self) -> &'static str {
        "permission_mode"
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn inspect(
        &self,
        tool_name: String,
        arguments: Value,
        call_id: String,
    ) -> napi::Result<Vec<JsInspectionResult>> {
        inspect_with(&self.inner, &tool_name, &arguments, &call_id)
    }
}

/// Coordinates inspectors and merges recommendations.
#[napi(js_name = "ToolInspectionManager")]
pub struct JsToolInspectionManager {
    path_root: Option<PathBuf>,
    permission_mode: Option<PermissionMode>,
    materialized: OnceLock<Arc<ToolInspectionManager>>,
}

impl JsToolInspectionManager {
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

#[napi]
impl JsToolInspectionManager {
    #[napi(constructor, catch_unwind)]
    pub const fn new() -> Self {
        Self {
            path_root: None,
            permission_mode: None,
            materialized: OnceLock::new(),
        }
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_path_security(&mut self, workspace_root: String) -> napi::Result<()> {
        if self.materialized.get().is_some() {
            return Err(Error::from_reason(
                "ToolInspectionManager is frozen after inspect/asHook; create a new manager",
            ));
        }
        self.path_root = Some(PathBuf::from(workspace_root));
        Ok(())
    }

    #[napi(catch_unwind)]
    pub fn add_permission_mode(&mut self, mode: &JsPermissionMode) -> napi::Result<()> {
        if self.materialized.get().is_some() {
            return Err(Error::from_reason(
                "ToolInspectionManager is frozen after inspect/asHook; create a new manager",
            ));
        }
        self.permission_mode = Some(PermissionMode::from(*mode));
        Ok(())
    }

    #[napi(catch_unwind)]
    pub fn inspector_names(&self) -> Vec<&'static str> {
        self.materialize().inspector_names()
    }

    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn inspect_one(
        &self,
        tool_name: String,
        arguments: Value,
        call_id: String,
    ) -> napi::Result<JsInspectionAction> {
        let request = ToolInspectorRequest {
            tool_name: &tool_name,
            arguments: &arguments,
            call_id: &call_id,
        };
        let manager = self.materialize();
        let rt = get_runtime()?;
        let action = rt
            .block_on(async { manager.inspect_one(&request).await })
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(action.into())
    }

    /// Wrap this manager as a lifecycle hook for `AgentBuilder.addInspectionHook(...)`.
    #[napi(catch_unwind)]
    pub fn as_hook(&self) -> JsInspectionHook {
        JsInspectionHook {
            inner: Arc::new(InspectionHookAdapter::new(self.materialize())),
        }
    }
}

/// Lifecycle-hook wrapper around an inspection manager.
#[napi(js_name = "InspectionHookAdapter")]
#[derive(Clone)]
pub struct JsInspectionHook {
    pub(crate) inner: Arc<InspectionHookAdapter>,
}

#[napi]
impl JsInspectionHook {
    #[napi(constructor, catch_unwind)]
    pub fn new(manager: &JsToolInspectionManager) -> Self {
        manager.as_hook()
    }
}

impl JsInspectionHook {
    pub(crate) fn as_lifecycle_hook(&self) -> Arc<dyn caliche::agent::hooks::LifecycleHook> {
        Arc::clone(&self.inner) as Arc<dyn caliche::agent::hooks::LifecycleHook>
    }
}

/// Build the default inspection chain.
#[napi(catch_unwind)]
pub fn default_inspection_manager(
    permission_mode: &JsPermissionMode,
    workspace_root: Option<String>,
) -> JsToolInspectionManager {
    let mode: PermissionMode = PermissionMode::from(*permission_mode);
    let root = workspace_root.map(PathBuf::from);
    let manager = rust_default_inspection_manager(mode, root.clone());
    let js_manager = JsToolInspectionManager {
        path_root: root,
        permission_mode: Some(mode),
        materialized: OnceLock::new(),
    };
    let _ = js_manager.materialized.set(Arc::new(manager));
    js_manager
}

fn inspect_with(
    inspector: &dyn ToolInspector,
    tool_name: &str,
    arguments: &Value,
    call_id: &str,
) -> napi::Result<Vec<JsInspectionResult>> {
    let request = ToolInspectorRequest {
        tool_name,
        arguments,
        call_id,
    };
    let rt = get_runtime()?;
    let results = rt
        .block_on(async { inspector.inspect(&request).await })
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(results
        .into_iter()
        .map(|inner| JsInspectionResult { inner })
        .collect())
}
