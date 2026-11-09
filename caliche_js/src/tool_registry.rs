//! JavaScript bindings for deferred/active tool registry controls.
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use crate::builtin_tools::BuiltinTool;
use crate::get_runtime;
use caliche::agent::tools::{Tool, ToolRegistry};
use napi_derive::napi;
use serde_json::Value;
use std::sync::Arc;

/// Thin handle over the shared Rust [`ToolRegistry`].
#[napi(js_name = "ToolRegistry")]
#[derive(Clone)]
pub struct JsToolRegistry {
    pub(crate) inner: Arc<ToolRegistry>,
}

impl JsToolRegistry {
    pub(crate) const fn from_inner(inner: Arc<ToolRegistry>) -> Self {
        Self { inner }
    }
}

#[napi]
impl JsToolRegistry {
    /// Create an empty registry (useful for deferred/active catalog tests).
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ToolRegistry::new()),
        }
    }

    /// Register a tool without exposing it until [`Self::activate`].
    #[napi(catch_unwind)]
    pub fn register_deferred(&self, tool: &BuiltinTool) {
        self.inner.register_deferred(Arc::clone(&tool.inner));
    }

    /// Register a tool and keep current active-set semantics.
    #[napi(catch_unwind)]
    pub fn register(&self, tool: &BuiltinTool) {
        self.inner.register(Arc::clone(&tool.inner));
    }

    /// Activate named tools (creating an active set if unrestricted).
    #[napi(catch_unwind)]
    pub fn activate(&self, names: Vec<String>) {
        self.inner.activate(names);
    }

    /// Deactivate named tools.
    #[napi(catch_unwind)]
    pub fn deactivate(&self, names: Vec<String>) {
        self.inner.deactivate(names);
    }

    /// Replace the active set. Pass `null`/`undefined` to activate all registered tools.
    #[napi(catch_unwind)]
    pub fn set_active_tools(&self, names: Option<Vec<String>>) {
        match names {
            Some(names) => self.inner.set_active_tools(Some(names)),
            None => self.inner.activate_all(),
        }
    }

    /// Clear the active filter so every registered tool is active.
    #[napi(catch_unwind)]
    pub fn activate_all(&self) {
        self.inner.activate_all();
    }

    #[napi(catch_unwind)]
    pub fn is_active(&self, name: String) -> bool {
        self.inner.is_active(&name)
    }

    #[napi(catch_unwind)]
    pub fn active_tool_names(&self) -> Vec<String> {
        self.inner.active_tool_names()
    }

    /// Execute a tool by name (honors active-set filtering).
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self, name: String, arguments: Value) -> napi::Result<Value> {
        let registry = Arc::clone(&self.inner);
        let result = registry.execute(&name, arguments).await;
        Ok(tool_result_to_value(result))
    }

    /// Schemas currently exposed to the model (active tools only).
    #[napi(catch_unwind)]
    pub fn get_tool_schemas(&self) -> Vec<Value> {
        self.inner
            .get_tool_schemas()
            .iter()
            .map(|schema| {
                serde_json::json!({
                    "name": schema.name.as_ref(),
                    "description": schema.description.as_ref(),
                    "parameters": {
                        "type": schema.parameters.schema_type.as_ref(),
                        "properties": schema.parameters.properties,
                        "required": schema.parameters.required.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>(),
                    },
                })
            })
            .collect()
    }

    #[napi(catch_unwind)]
    pub fn list(&self) -> Vec<Value> {
        self.inner
            .list()
            .into_iter()
            .map(|(name, description, parameters)| {
                serde_json::json!({
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                })
            })
            .collect()
    }
}

fn tool_result_to_value(result: caliche::agent::tools::ToolResult) -> Value {
    let mut obj = serde_json::json!({
        "success": result.success,
        "content": result.content,
    });
    if let Some(err) = result.error {
        obj["error"] = Value::String(err);
    }
    if let Some(payload) = result.typed_payload {
        obj["typed_payload"] = payload;
    }
    obj
}

/// Shared helper used by `BuiltinTool::execute`.
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn execute_tool(tool: Arc<dyn Tool>, arguments: Value) -> napi::Result<Value> {
    let rt = get_runtime()?;
    let result = rt.block_on(async move { tool.execute(arguments).await });
    Ok(tool_result_to_value(result))
}
