use serde_json::Value;

pub use caliche::mcp::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

pub const ERR_AUTH_REQUIRED: i32 = -32000;
pub const ERR_NOT_FOUND: i32 = -32001;
pub const ERR_CONFLICT: i32 = -32002;
pub const ERR_VALIDATION: i32 = -32003;
pub const ERR_RATE_LIMITED: i32 = -32004;
pub const ERR_FEATURE_DISABLED: i32 = -32005;
pub const ERR_WS_REQUIRED: i32 = -32006;

pub fn app_error(code: i32, message: impl Into<String>, data: Option<Value>) -> JsonRpcError {
    JsonRpcError::server_error(code, message.into(), data)
}
