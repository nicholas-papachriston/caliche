use axum::Json;
use axum::body::to_bytes;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::middleware::session_auth::maybe_authorize_session_optional;
use crate::rpc::connection::ConnectionCtx;
use crate::rpc::types::JsonRpcError;

pub fn parse_params<T: DeserializeOwned>(params: Value) -> Result<T, JsonRpcError> {
    serde_json::from_value(params).map_err(|e| {
        JsonRpcError::invalid_params(Some(json!({
            "error": e.to_string()
        })))
    })
}

pub fn map_handler_err(status: axum::http::StatusCode, message: String) -> JsonRpcError {
    JsonRpcError::server_error(
        -32000 - i32::from(status.as_u16()),
        format!("HTTP {}: {}", status.as_u16(), message),
        Some(json!({"status": status.as_u16()})),
    )
}

pub fn to_rpc_value<T: Serialize>(response: T) -> Result<Value, JsonRpcError> {
    serde_json::to_value(response).map_err(|e| {
        JsonRpcError::internal_error(Some(json!({
            "error": e.to_string()
        })))
    })
}

pub fn require_param_str(params: &Value, key: &str) -> Result<String, JsonRpcError> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| JsonRpcError::invalid_params(Some(json!({ "missing": key }))))
}

pub fn require_session_id(params: &Value) -> Result<String, JsonRpcError> {
    require_param_str(params, "session_id")
}

pub async fn require_authorized_session(
    ctx: &ConnectionCtx,
    params: &Value,
) -> Result<String, JsonRpcError> {
    let session_id = require_session_id(params)?;
    maybe_authorize_session_optional(&ctx.sessions_state, &session_id, ctx.user_context.as_ref())
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(session_id)
}

pub fn json_handler_to_value<T: Serialize>(
    result: Result<Json<T>, (StatusCode, String)>,
) -> Result<Value, JsonRpcError> {
    match result {
        Ok(Json(response)) => to_rpc_value(response),
        Err((status, message)) => Err(map_handler_err(status, message)),
    }
}

pub async fn into_value_response(resp: impl IntoResponse) -> Result<Value, JsonRpcError> {
    let response = resp.into_response();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;
    if status.is_success() {
        Ok(value)
    } else {
        Err(map_handler_err(status, value.to_string()))
    }
}
