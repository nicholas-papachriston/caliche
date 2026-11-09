use crate::agent::tools::names::{TOOL_CONTEXT_GET, TOOL_CONTEXT_SET};
use crate::agent::tools::tool::{Tool, ToolMeta, ToolResult};
use crate::orchestration::state::SharedState;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Tool for reading values from shared state
#[derive(Debug)]
pub struct ContextGetTool {
    shared_state: Arc<SharedState>,
}

impl ContextGetTool {
    /// Create a new context get tool
    #[must_use]
    pub const fn new(shared_state: Arc<SharedState>) -> Self {
        Self { shared_state }
    }
}

const GET_NAME: &str = TOOL_CONTEXT_GET;
const GET_DESCRIPTION: &str = "Read a value from the shared context/state. Use JSON pointer paths like '/order_id' or '/customer/name'.";

impl ToolMeta for ContextGetTool {
    fn name(&self) -> &str {
        GET_NAME
    }

    fn description(&self) -> &str {
        GET_DESCRIPTION
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(GET_NAME)
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(GET_DESCRIPTION)
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "JSON pointer path to the value (e.g., '/order_id' or '/customer/name'). Use '/' for entire state."
                }
            },
            "required": ["path"]
        })
    }
}

#[async_trait]
impl Tool for ContextGetTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path = match arguments.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult::error("Missing required parameter: path");
            }
        };

        match self.shared_state.get(path).await {
            Ok(Some(value)) => {
                let value_str = serde_json::to_string(&value)
                    .unwrap_or_else(|_| format!("<unserializable value at {path}>"));
                ToolResult::success(value_str)
            }
            Ok(None) => ToolResult::success(format!("No value found at path: {path}")),
            Err(e) => ToolResult::error(format!("Failed to get context value: {e}")),
        }
    }
}

/// Tool for writing values to shared state
#[derive(Debug)]
pub struct ContextSetTool {
    shared_state: Arc<SharedState>,
}

impl ContextSetTool {
    /// Create a new context set tool
    #[must_use]
    pub const fn new(shared_state: Arc<SharedState>) -> Self {
        Self { shared_state }
    }
}

const SET_NAME: &str = TOOL_CONTEXT_SET;
const SET_DESCRIPTION: &str = "Write a value to the shared context/state. Use JSON pointer paths like '/order_id' or '/customer/name'.";

impl ToolMeta for ContextSetTool {
    fn name(&self) -> &str {
        SET_NAME
    }

    fn description(&self) -> &str {
        SET_DESCRIPTION
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(SET_NAME)
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(SET_DESCRIPTION)
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "JSON pointer path where to set the value (e.g., '/order_id' or '/customer/name'). Use '/' to replace entire state."
                },
                "value": {
                    "description": "The value to set (can be string, number, boolean, object, or array)"
                }
            },
            "required": ["path", "value"]
        })
    }
}

#[async_trait]
impl Tool for ContextSetTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path = match arguments.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult::error("Missing required parameter: path");
            }
        };

        let value = match arguments.get("value") {
            Some(v) => v.clone(),
            None => {
                return ToolResult::error("Missing required parameter: value");
            }
        };

        match self.shared_state.set(path, value.clone()).await {
            Ok(()) => {
                let value_str = serde_json::to_string(&value)
                    .unwrap_or_else(|_| "<unserializable value>".to_string());
                ToolResult::success(format!("Successfully set {path} = {value_str}"))
            }
            Err(e) => ToolResult::error(format!("Failed to set context value: {e}")),
        }
    }
}
