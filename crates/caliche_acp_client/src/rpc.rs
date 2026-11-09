use caliche_contracts::acp::{
    RequestPermissionOutcome, RequestPermissionResponse, SessionNotification, methods,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::AcpClientError;

/// Failure delivered to a waiter for an in-flight JSON-RPC id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaiterFail {
    Rpc { code: i32, message: String },
    Protocol(String),
    Closed,
}

impl From<WaiterFail> for AcpClientError {
    fn from(fail: WaiterFail) -> Self {
        match fail {
            WaiterFail::Rpc { code, message } => Self::Rpc { code, message },
            WaiterFail::Protocol(detail) => Self::Protocol { detail },
            WaiterFail::Closed => Self::StdoutClosed,
        }
    }
}

/// One stdout line, already split from the newline.
#[derive(Debug)]
pub enum LineClass {
    Response {
        id: u64,
        body: Result<Value, WaiterFail>,
    },
    Incoming(Incoming),
    Reply(Value),
    ReplyAndIncoming {
        reply: Value,
        incoming: Incoming,
    },
    Fatal(WaiterFail),
    Ignore,
}

/// A `session/update` the prompt loop can store.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Update(SessionNotification),
    Unrecognized(Value),
}

#[derive(Debug, Deserialize)]
struct Envelope {
    #[serde(default)]
    jsonrpc: Option<String>,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    params: Option<Value>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<Value>,
}

/// Encode one JSON-RPC frame. The only newline is the terminator.
///
/// # Errors
///
/// Returns the serde error when `value` cannot be encoded.
pub fn encode_frame(value: &Value) -> Result<Vec<u8>, serde_json::Error> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn classify_line(line: &str) -> LineClass {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return LineClass::Fatal(WaiterFail::Protocol(truncate_line(line)));
    };
    if !value.is_object() {
        return LineClass::Fatal(WaiterFail::Protocol(truncate_line(line)));
    }
    classify_object(value)
}

fn classify_object(value: Value) -> LineClass {
    let Ok(envelope) = serde_json::from_value::<Envelope>(value) else {
        return LineClass::Fatal(WaiterFail::Protocol(
            "ACP frame is not a JSON object".to_owned(),
        ));
    };
    if let Some(version) = envelope.jsonrpc.as_deref() {
        if version != caliche_contracts::rpc::JSONRPC_VERSION {
            return LineClass::Fatal(WaiterFail::Protocol(format!(
                "ACP jsonrpc version {version} is not 2.0"
            )));
        }
    }
    match (envelope.method, envelope.id) {
        (Some(method), Some(id)) => classify_request(&method, &id, envelope.params),
        (Some(method), None) => classify_notification(&method, envelope.params),
        (None, Some(id)) => classify_response(&id, envelope.result, envelope.error),
        (None, None) => LineClass::Fatal(WaiterFail::Protocol(
            "ACP frame has neither method nor id".to_owned(),
        )),
    }
}

fn classify_request(method: &str, id: &Value, params: Option<Value>) -> LineClass {
    if method == methods::SESSION_UPDATE {
        return LineClass::ReplyAndIncoming {
            reply: reply_ok(id, &Value::Null),
            incoming: incoming_from_update(params),
        };
    }
    if method == methods::SESSION_REQUEST_PERMISSION {
        return LineClass::Reply(permission_cancelled(&id));
    }
    LineClass::Reply(reply_err(
        &id,
        -32601,
        &format!("Method not found: {method}"),
    ))
}

fn classify_notification(method: &str, params: Option<Value>) -> LineClass {
    if method == methods::SESSION_UPDATE {
        LineClass::Incoming(incoming_from_update(params))
    } else {
        LineClass::Ignore
    }
}

fn classify_response(id: &Value, result: Option<Value>, error: Option<Value>) -> LineClass {
    let Some(id) = json_id_u64(id) else {
        return LineClass::Ignore;
    };
    if let Some(error) = error {
        return LineClass::Response {
            id,
            body: Err(error_fail(&error)),
        };
    }
    result.map_or_else(
        || LineClass::Response {
            id,
            body: Err(WaiterFail::Protocol(format!(
                "ACP response {id} has no result"
            ))),
        },
        |result| LineClass::Response {
            id,
            body: Ok(result),
        },
    )
}

fn incoming_from_update(params: Option<Value>) -> Incoming {
    let Some(params) = params else {
        return Incoming::Unrecognized(Value::Null);
    };
    serde_json::from_value::<SessionNotification>(params.clone())
        .map_or(Incoming::Unrecognized(params), Incoming::Update)
}

fn permission_cancelled(id: &Value) -> Value {
    let result = RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled);
    match serde_json::to_value(result) {
        Ok(value) => reply_ok(id, &value),
        Err(err) => reply_err(id, -32603, &err.to_string()),
    }
}

fn reply_ok(id: &Value, result: &Value) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "result": result,
    })
}

fn reply_err(id: &Value, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "error": { "code": code, "message": message },
    })
}

fn error_fail(error: &Value) -> WaiterFail {
    let code = error
        .get("code")
        .and_then(Value::as_i64)
        .and_then(|code| i32::try_from(code).ok())
        .unwrap_or(-32603);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("ACP request failed")
        .to_owned();
    WaiterFail::Rpc { code, message }
}

fn json_id_u64(id: &Value) -> Option<u64> {
    id.as_u64()
        .or_else(|| id.as_i64().and_then(|value| u64::try_from(value).ok()))
}

fn truncate_line(line: &str) -> String {
    const MAX_CHARS: usize = 180;
    let trimmed = line.trim();
    let count = trimmed.chars().count();
    if count <= MAX_CHARS {
        return trimmed.to_owned();
    }
    let end = trimmed
        .char_indices()
        .nth(MAX_CHARS)
        .map_or(trimmed.len(), |(index, _)| index);
    format!("{}...", &trimmed[..end])
}

#[cfg(test)]
mod tests {
    use super::{LineClass, classify_line, encode_frame};
    use serde_json::{Value, json};

    #[test]
    fn frame_is_one_json_line() {
        let frame = encode_frame(
            &json!({"jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION, "method": "initialize"}),
        )
        .unwrap();
        assert!(frame.ends_with(b"\n"), "{frame:?}");
        assert!(
            !frame[..frame.len() - 1].contains(&b'\n'),
            "frame contains an embedded newline"
        );
        let text = std::str::from_utf8(&frame).unwrap();
        let parsed: Value = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(parsed["method"], "initialize");
    }

    #[test]
    fn log_line_is_a_protocol_error() {
        let class = classify_line("tracing info: turn started");
        assert!(matches!(class, LineClass::Fatal(_)));
    }

    #[test]
    fn permission_request_is_cancelled() {
        let class = classify_line(
            r#"{"jsonrpc":"2.0","id":7,"method":"session/request_permission","params":{}}"#,
        );
        let LineClass::Reply(value) = class else {
            panic!("expected a reply");
        };
        assert_eq!(
            value["result"]["outcome"]["outcome"].as_str(),
            Some("cancelled")
        );
    }

    #[test]
    fn filesystem_request_is_method_not_found() {
        let class =
            classify_line(r#"{"jsonrpc":"2.0","id":3,"method":"fs/read_text_file","params":{}}"#);
        let LineClass::Reply(value) = class else {
            panic!("expected a reply");
        };
        assert_eq!(value["error"]["code"].as_i64(), Some(-32601));
    }

    #[test]
    fn session_update_notification_is_incoming() {
        let class = classify_line(
            r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hi"}}}}"#,
        );
        assert!(matches!(class, LineClass::Incoming(_)));
    }
}
