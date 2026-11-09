//! JavaScript bindings for Caliche Built-in Tools

use crate::mutation::JsMutationStyle;
use crate::tool_registry::execute_tool;
use caliche::agent::tools::{
    ApplyPatchTool, CalculatorTool, CodeExecutionTool, FsTool, HttpTool, MutationStyle, Tool,
};
use napi_derive::napi;
use serde_json::Value;
use std::sync::Arc;

/// Wrapper for built-in tools
#[napi]
pub struct BuiltinTool {
    pub(crate) inner: Arc<dyn Tool>,
}

#[napi]
impl BuiltinTool {
    /// Create a `CalculatorTool` instance
    #[napi(factory, catch_unwind)]
    pub fn calculator() -> Self {
        Self {
            inner: Arc::new(CalculatorTool) as Arc<dyn Tool>,
        }
    }

    /// Create an `FsTool` instance.
    ///
    /// When `mutation_style` is `ApplyPatch`, write/edit ops are disabled on `fs`
    /// (use [`Self::apply_patch`] instead).
    #[napi(factory, catch_unwind)]
    pub fn fs(mutation_style: Option<&JsMutationStyle>) -> Self {
        let style = mutation_style
            .copied()
            .map_or(MutationStyle::EditWrite, MutationStyle::from);
        Self {
            inner: Arc::new(FsTool::with_mutation_style(style)) as Arc<dyn Tool>,
        }
    }

    /// Create an `ApplyPatchTool` instance (Codex-style freeform patch).
    #[napi(factory, catch_unwind)]
    pub fn apply_patch() -> Self {
        Self {
            inner: Arc::new(ApplyPatchTool::new()) as Arc<dyn Tool>,
        }
    }

    /// Create an `HttpTool` instance (unified HTTP tool supporting request, `fetch_html`, and `api_call` operations)
    #[napi(factory, catch_unwind)]
    pub fn http() -> Self {
        Self {
            inner: Arc::new(HttpTool::new()) as Arc<dyn Tool>,
        }
    }

    /// Create a `CodeExecutionTool` instance
    #[napi(factory, catch_unwind)]
    pub fn code_execution() -> Self {
        Self {
            inner: Arc::new(CodeExecutionTool) as Arc<dyn Tool>,
        }
    }

    /// Get the name of a built-in tool
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get the description of a built-in tool
    #[napi(catch_unwind)]
    pub fn description(&self) -> String {
        self.inner.description().to_string()
    }

    /// Get the parameters of a built-in tool
    #[napi(catch_unwind)]
    pub fn parameters(&self) -> serde_json::Value {
        self.inner.parameters()
    }

    /// Execute the tool (async work is driven on the shared runtime).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn execute(&self, arguments: Value) -> napi::Result<Value> {
        execute_tool(Arc::clone(&self.inner), arguments)
    }
}
