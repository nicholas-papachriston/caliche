//! MCP admission for protocol `2026-07-28`.
//!
//! Every request carries the protocol version in `_meta`, except
//! `server/discover`, which may omit it. This server does not accept
//! `initialize`.

use crate::negotiate::parse_protocol_date;
use crate::protocol::{JsonRpcError, SUPPORTED_PROTOCOL_VERSIONS};
use beef::lean::Cow;
use serde_json::{Value, json};

/// `_meta` key that carries the protocol version on a modern request.
pub const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";

/// JSON-RPC code for `UnsupportedProtocolVersionError`.
pub const UNSUPPORTED_PROTOCOL_VERSION: i32 = -32022;

/// JSON-RPC code for `HeaderMismatchError`.
pub const HEADER_MISMATCH: i32 = -32020;

/// JSON-RPC code when a request was cancelled before it started.
pub const REQUEST_CANCELLED: i32 = -32800;

const MODERN_FLOOR: u32 = 2026_07_28;
const LIST_TTL_MS: u64 = 60_000;
const DISCOVER_TTL_MS: u64 = 3_600_000;

/// Cache fields required on some results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    None,
    PrivateList,
    PublicDiscover,
}

/// Decide whether this request may run.
///
/// `server/discover` may omit `_meta`. Every other method must name
/// [`crate::protocol::LATEST_PROTOCOL_VERSION`]. Any other version returns
/// [`UNSUPPORTED_PROTOCOL_VERSION`].
pub fn admit(method: &str, params: Option<&Value>) -> Result<(), JsonRpcError> {
    if method == "server/discover" {
        return match meta_protocol_version(params) {
            Some(version) => require_version(version),
            None => Ok(()),
        };
    }
    match meta_protocol_version(params) {
        Some(version) => require_version(version),
        None => Err(unsupported_protocol("missing")),
    }
}

/// Cache hint for a successful result.
#[must_use]
pub fn cache_hint(method: &str) -> CacheHint {
    match method {
        "server/discover" => CacheHint::PublicDiscover,
        "tools/list"
        | "prompts/list"
        | "resources/list"
        | "resources/read"
        | "resources/templates/list" => CacheHint::PrivateList,
        _ => CacheHint::None,
    }
}

/// Add `resultType` and, when required, cache fields to a success result.
pub fn shape_result(
    mut response: crate::protocol::JsonRpcResponse,
    hint: CacheHint,
) -> crate::protocol::JsonRpcResponse {
    let Some(Value::Object(map)) = response.result.as_mut() else {
        return response;
    };
    map.insert("resultType".to_string(), json!("complete"));
    match hint {
        CacheHint::None => {}
        CacheHint::PrivateList => insert_cache(map, LIST_TTL_MS, "private"),
        CacheHint::PublicDiscover => insert_cache(map, DISCOVER_TTL_MS, "public"),
    }
    response
}

/// Error body a modern client uses to pick another version.
pub fn unsupported_protocol(requested: &str) -> JsonRpcError {
    JsonRpcError {
        code: UNSUPPORTED_PROTOCOL_VERSION,
        message: Cow::borrowed("Unsupported protocol version"),
        data: Some(json!({
            "supported": SUPPORTED_PROTOCOL_VERSIONS,
            "requested": requested,
        })),
    }
}

/// Protocol version declared in `_meta`, when the field is a string.
#[must_use]
pub fn meta_protocol_version(params: Option<&Value>) -> Option<&str> {
    params?.get("_meta")?.get(META_PROTOCOL_VERSION)?.as_str()
}

/// `true` when a protocol header selects the modern revision.
#[must_use]
pub fn protocol_header_is_modern(header: Option<&str>) -> bool {
    header.is_some_and(is_modern)
}

/// Name a modern client must send in `Mcp-Name`.
#[must_use]
pub fn expected_mcp_name<'a>(method: &'a str, params: Option<&'a Value>) -> &'a str {
    params
        .and_then(|value| value.get("name"))
        .and_then(Value::as_str)
        .or_else(|| {
            params
                .and_then(|value| value.get("uri"))
                .and_then(Value::as_str)
        })
        .unwrap_or(method)
}

/// Stable key for a JSON-RPC id. String and number ids keep their text.
#[must_use]
pub fn request_id_key(id: &Value) -> String {
    match id {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        other => other.to_string(),
    }
}

/// `requestId` from a `notifications/cancelled` payload.
#[must_use]
pub fn cancelled_request_key(params: Option<&Value>) -> Option<String> {
    params
        .and_then(|value| value.get("requestId"))
        .map(request_id_key)
}

/// Error for a request that was cancelled before dispatch.
#[must_use]
pub fn request_cancelled() -> JsonRpcError {
    JsonRpcError::server_error(REQUEST_CANCELLED, "Request cancelled".to_string(), None)
}

/// `subscriptions/listen` result. The server does not open a long-lived stream.
#[must_use]
pub fn listen_response(id: Option<Value>) -> crate::protocol::JsonRpcResponse {
    let body = match id.as_ref() {
        Some(value) => json!({ "subscriptionId": request_id_key(value) }),
        None => json!({ "subscriptionId": "listen" }),
    };
    crate::protocol::JsonRpcResponse::success(id, body)
}

/// Header check for a modern HTTP request.
///
/// Legacy requests return `None`. A modern request must send
/// `MCP-Protocol-Version`, `Mcp-Method`, and `Mcp-Name`.
#[must_use]
pub fn modern_header_error(
    method: &str,
    params: Option<&Value>,
    method_header: Option<&str>,
    protocol_header: Option<&str>,
    name_header: Option<&str>,
) -> Option<JsonRpcError> {
    modern_header_mismatch(method, params, method_header, protocol_header, name_header)
}

fn require_version(version: &str) -> Result<(), JsonRpcError> {
    if SUPPORTED_PROTOCOL_VERSIONS.contains(&version) && is_modern(version) {
        Ok(())
    } else {
        Err(unsupported_protocol(version))
    }
}

fn is_modern(version: &str) -> bool {
    parse_protocol_date(version).is_some_and(|date| date >= MODERN_FLOOR)
}

fn modern_header_mismatch(
    method: &str,
    params: Option<&Value>,
    method_header: Option<&str>,
    protocol_header: Option<&str>,
    name_header: Option<&str>,
) -> Option<JsonRpcError> {
    let Some(version) = protocol_header else {
        return Some(header_mismatch("MCP-Protocol-Version is required"));
    };
    if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
        return Some(unsupported_protocol(version));
    }
    if meta_protocol_version(params).is_some_and(|meta| meta != version) {
        return Some(header_mismatch("MCP-Protocol-Version does not match _meta"));
    }
    if method_header != Some(method) {
        return Some(header_mismatch(
            "Mcp-Method does not match the JSON-RPC method",
        ));
    }
    if name_header != Some(expected_mcp_name(method, params)) {
        return Some(header_mismatch("Mcp-Name does not match the request"));
    }
    None
}

fn header_mismatch(message: &'static str) -> JsonRpcError {
    JsonRpcError {
        code: HEADER_MISMATCH,
        message: Cow::borrowed(message),
        data: None,
    }
}

fn insert_cache(map: &mut serde_json::Map<String, Value>, ttl_ms: u64, scope: &str) {
    map.insert("ttlMs".to_string(), json!(ttl_ms));
    map.insert("cacheScope".to_string(), json!(scope));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_without_meta_is_allowed() {
        admit("server/discover", None).expect("admit");
    }

    #[test]
    fn current_list_is_allowed() {
        let params = json!({
            "_meta": { META_PROTOCOL_VERSION: "2026-07-28" }
        });
        admit("tools/list", Some(&params)).expect("admit");
    }

    #[test]
    fn missing_version_is_unsupported() {
        let err = admit("tools/list", None).expect_err("version");
        assert_eq!(err.code, UNSUPPORTED_PROTOCOL_VERSION);
    }

    #[test]
    fn unknown_version_lists_supported_revisions() {
        let params = json!({
            "_meta": { META_PROTOCOL_VERSION: "1900-01-01" }
        });
        let err = admit("tools/list", Some(&params)).expect_err("version");
        assert_eq!(err.code, UNSUPPORTED_PROTOCOL_VERSION);
        let supported = err
            .data
            .as_ref()
            .and_then(|data| data.get("supported"))
            .and_then(|value| value.as_array())
            .expect("supported");
        assert!(supported.iter().any(|value| value == "2026-07-28"));
        assert!(supported.iter().all(|value| value != "2025-11-25"));
    }

    #[test]
    fn missing_headers_are_a_mismatch() {
        let err = modern_header_error("tools/list", None, None, None, None).expect("mismatch");
        assert_eq!(err.code, HEADER_MISMATCH);
    }

    #[test]
    fn modern_request_without_method_header_is_a_mismatch() {
        let params = json!({
            "_meta": { META_PROTOCOL_VERSION: "2026-07-28" }
        });
        let err = modern_header_error(
            "tools/list",
            Some(&params),
            None,
            Some("2026-07-28"),
            Some("tools/list"),
        )
        .expect("mismatch");
        assert_eq!(err.code, HEADER_MISMATCH);
    }

    #[test]
    fn matching_modern_headers_pass() {
        let params = json!({
            "_meta": { META_PROTOCOL_VERSION: "2026-07-28" },
            "name": "echo"
        });
        assert!(
            modern_header_error(
                "tools/call",
                Some(&params),
                Some("tools/call"),
                Some("2026-07-28"),
                Some("echo"),
            )
            .is_none()
        );
    }

    #[test]
    fn listen_body_uses_the_request_id() {
        let response = listen_response(Some(json!(7)));
        let result = response.result.expect("result");
        assert_eq!(result["subscriptionId"], "7");
    }
}
