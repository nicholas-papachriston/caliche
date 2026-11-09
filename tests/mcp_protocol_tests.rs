//! Tests for MCP protocol implementation
//!
//! Tests JSON-RPC 2.0 compliance, message serialization, error handling,
//! and protocol version negotiation.

#![cfg(feature = "mcp")]

use beef::lean::Cow;
use caliche::mcp::protocol::*;
use serde_json::{Value, json};

#[tokio::test]
async fn test_jsonrpc_request_serialization() {
    let request = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("test_method"),
        params: Some(json!({"key": "value"})),
        id: Some(Value::Number(1.into())),
    };

    let serialized = serde_json::to_string(&request).expect("Request serialization should succeed");
    assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
    assert!(serialized.contains("\"method\":\"test_method\""));
    assert!(serialized.contains("\"id\":1"));

    let deserialized: JsonRpcRequest =
        serde_json::from_str(&serialized).expect("Request deserialization should succeed");
    assert_eq!(&*deserialized.jsonrpc, "2.0");
    assert_eq!(&*deserialized.method, "test_method");
}

#[tokio::test]
async fn test_jsonrpc_response_success() {
    let response =
        JsonRpcResponse::success(Some(Value::Number(1.into())), json!({"result": "success"}));

    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());
    assert!(response.error.is_none());
    assert_eq!(response.id, Some(Value::Number(1.into())));

    let serialized =
        serde_json::to_string(&response).expect("Response serialization should succeed");
    assert!(serialized.contains("\"jsonrpc\":\"2.0\""));
    assert!(serialized.contains("\"result\""));
}

#[tokio::test]
async fn test_jsonrpc_response_error() {
    let error = JsonRpcError::invalid_request(None);
    let response = JsonRpcResponse::error(Some(Value::String("req-1".to_string())), error);

    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_none());
    assert!(response.error.is_some());
    assert_eq!(
        response
            .error
            .as_ref()
            .expect("Response should have error")
            .code,
        -32600
    );
    assert_eq!(
        response
            .error
            .as_ref()
            .expect("Response should have error")
            .message,
        "Invalid Request"
    );
}

#[tokio::test]
async fn test_jsonrpc_error_codes() {
    // Parse error
    let parse_error = JsonRpcError::parse_error(None);
    assert_eq!(parse_error.code, -32700);
    assert_eq!(parse_error.message, "Parse error");

    // Invalid request
    let invalid_request = JsonRpcError::invalid_request(None);
    assert_eq!(invalid_request.code, -32600);

    // Method not found
    let method_not_found = JsonRpcError::method_not_found("unknown_method");
    assert_eq!(method_not_found.code, -32601);
    assert_eq!(
        method_not_found.data,
        Some(Value::String("unknown_method".to_string()))
    );

    // Invalid params
    let invalid_params = JsonRpcError::invalid_params(Some(json!({"field": "error"})));
    assert_eq!(invalid_params.code, -32602);

    // Internal error
    let internal_error = JsonRpcError::internal_error(None);
    assert_eq!(internal_error.code, -32603);

    // Server error
    let server_error = JsonRpcError::server_error(-32000, "Custom error".to_string(), None);
    assert_eq!(server_error.code, -32000);
    assert_eq!(server_error.message, "Custom error");
}

#[tokio::test]
async fn test_jsonrpc_notification() {
    // Notification has null ID
    let notification = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("notify"),
        params: Some(json!({"event": "something"})),
        id: None,
    };

    assert!(notification.id.is_none());
    let serialized =
        serde_json::to_string(&notification).expect("Notification serialization should succeed");
    // Should not contain id field when None
    assert!(!serialized.contains("\"id\""));
}

#[tokio::test]
async fn test_jsonrpc_batch_request() {
    let batch = vec![
        JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("method1"),
            params: None,
            id: Some(Value::Number(1.into())),
        },
        JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("method2"),
            params: None,
            id: Some(Value::Number(2.into())),
        },
    ];

    let serialized =
        serde_json::to_string(&batch).expect("Batch request serialization should succeed");
    assert!(serialized.contains("method1"));
    assert!(serialized.contains("method2"));

    let deserialized: Vec<JsonRpcRequest> =
        serde_json::from_str(&serialized).expect("Batch request deserialization should succeed");
    assert_eq!(deserialized.len(), 2);
}

#[tokio::test]
async fn test_mcp_protocol_version() {
    assert_eq!(LATEST_PROTOCOL_VERSION, "2026-07-28");
    assert_eq!(
        LATEST_PROTOCOL_VERSION,
        *SUPPORTED_PROTOCOL_VERSIONS
            .last()
            .expect("SUPPORTED_PROTOCOL_VERSIONS should be non-empty"),
        "LATEST_PROTOCOL_VERSION must be the last entry in SUPPORTED_PROTOCOL_VERSIONS"
    );
    assert_eq!(SUPPORTED_PROTOCOL_VERSIONS, ["2026-07-28"]);
}

#[tokio::test]
async fn test_initialize_params_serialization() {
    let params = InitializeParams {
        protocol_version: LATEST_PROTOCOL_VERSION.to_string(),
        capabilities: ClientCapabilities::default(),
        client_info: Some(ClientInfo {
            name: Cow::borrowed("test_client"),
            version: Some(Cow::borrowed("1.0.0")),
            title: None,
            icons: None,
        }),
    };

    let serialized = serde_json::to_string(&params).expect("Params serialization should succeed");
    assert!(serialized.contains("protocolVersion"));
    assert!(serialized.contains("test_client"));

    let deserialized: InitializeParams =
        serde_json::from_str(&serialized).expect("InitializeParams deserialization should succeed");
    assert_eq!(deserialized.protocol_version, LATEST_PROTOCOL_VERSION);
}

#[tokio::test]
async fn test_initialize_result_serialization() {
    let result = InitializeResult {
        protocol_version: Cow::borrowed(LATEST_PROTOCOL_VERSION),
        capabilities: ServerCapabilities {
            tools: Some(ToolsCapability {
                list_changed: Some(false),
            }),
            prompts: None,
            resources: None,
            logging: None,
            completions: None,
            tasks: None,
            experimental: None,
        },
        server_info: Some(ServerInfo {
            name: Cow::borrowed("test_server"),
            version: Some(Cow::borrowed("1.0.0")),
            title: None,
            description: None,
            icons: None,
        }),
        instructions: None,
    };

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("protocolVersion"));
    assert!(serialized.contains("test_server"));
    assert!(serialized.contains("capabilities"));
}

#[tokio::test]
async fn test_tools_list_result_serialization() {
    let tool = McpTool::new(
        "test_tool",
        "A test tool",
        json!({
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            }
        }),
    );

    let result = ToolsListResult {
        tools: vec![tool],
        next_cursor: None,
    };

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("test_tool"));
    assert!(serialized.contains("tools"));
}

#[tokio::test]
async fn test_tools_call_params_serialization() {
    let params = ToolsCallParams {
        name: "calculator".to_string(),
        arguments: Some(json!({"operation": "add", "a": 1, "b": 2})),
    };

    let serialized = serde_json::to_string(&params).expect("Params serialization should succeed");
    assert!(serialized.contains("calculator"));
    assert!(serialized.contains("operation"));

    let deserialized: ToolsCallParams =
        serde_json::from_str(&serialized).expect("ToolsCallParams deserialization should succeed");
    assert_eq!(deserialized.name, "calculator");
}

#[tokio::test]
async fn test_tools_call_result_with_text() {
    let result = ToolsCallResult::from_text("Result: 3", None, false);

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("Result: 3"));
    assert!(serialized.contains("\"type\":\"text\""));
}

#[tokio::test]
async fn test_tools_call_result_with_resource() {
    let result = ToolsCallResult {
        content: Some(vec![ContentPart::Resource {
            resource: ResourceReference {
                uri: "file:///path/to/file".to_string(),
                text: Some("File contents".to_string()),
                mime_type: Some("text/plain".to_string()),
            },
        }]),
        structured_content: None,
        is_error: Some(false),
        meta: None,
    };

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("file:///path/to/file"));
    assert!(serialized.contains("\"type\":\"resource\""));
}

#[tokio::test]
async fn test_prompts_list_result_serialization() {
    let prompt = McpPrompt {
        name: "test_prompt".to_string(),
        title: None,
        description: Some("A test prompt".to_string()),
        icons: None,
        arguments: Some(vec![PromptArgument {
            name: "topic".to_string(),
            title: None,
            description: Some("The topic".to_string()),
            required: Some(true),
        }]),
        meta: None,
    };

    let result = PromptsListResult {
        prompts: vec![prompt],
        next_cursor: None,
    };

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("test_prompt"));
    assert!(serialized.contains("prompts"));
}

#[tokio::test]
async fn test_resources_list_result_serialization() {
    let resource = McpResource {
        uri: "file:///test".to_string(),
        name: Some("Test Resource".to_string()),
        title: None,
        description: Some("A test resource".to_string()),
        mime_type: Some("text/plain".to_string()),
        size: None,
        icons: None,
        meta: None,
    };

    let result = ResourcesListResult {
        resources: vec![resource],
        next_cursor: None,
    };

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("file:///test"));
    assert!(serialized.contains("resources"));
}

#[tokio::test]
async fn test_resources_read_result_serialization() {
    let contents = ResourceContents {
        uri: "file:///test".to_string(),
        mime_type: Some("text/plain".to_string()),
        text: Some("File contents".to_string()),
        blob: None,
    };

    let result = ResourcesReadResult {
        contents: vec![contents],
    };

    let serialized = serde_json::to_string(&result).expect("Result serialization should succeed");
    assert!(serialized.contains("file:///test"));
    assert!(serialized.contains("contents"));
}

#[tokio::test]
async fn test_jsonrpc_version_validation() {
    // Valid version
    let valid_request = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("test"),
        params: None,
        id: Some(Value::Number(1.into())),
    };
    assert_eq!(&*valid_request.jsonrpc, "2.0");

    // Invalid version should be caught by handler
    let invalid_request = JsonRpcRequest {
        jsonrpc: Cow::borrowed("1.0"),
        method: Cow::borrowed("test"),
        params: None,
        id: Some(Value::Number(1.into())),
    };
    assert_ne!(&*invalid_request.jsonrpc, "2.0");
}

#[tokio::test]
async fn test_error_response_with_data() {
    let error = JsonRpcError::invalid_params(Some(json!({
        "field": "missing_required",
        "details": "The 'name' field is required"
    })));

    assert_eq!(error.code, -32602);
    assert!(error.data.is_some());
    let data = error.data.as_ref().expect("Error should have data");
    assert!(data.get("field").is_some());
}

#[tokio::test]
async fn test_request_id_types() {
    // String ID
    let req1 = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("test"),
        params: None,
        id: Some(Value::String("req-123".to_string())),
    };
    assert!(matches!(req1.id, Some(Value::String(_))));

    // Number ID
    let req2 = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("test"),
        params: None,
        id: Some(Value::Number(42.into())),
    };
    assert!(matches!(req2.id, Some(Value::Number(_))));

    // Null ID (notification)
    let req3 = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("test"),
        params: None,
        id: None,
    };
    assert!(req3.id.is_none());
}

#[tokio::test]
async fn test_server_capabilities_all_enabled() {
    let capabilities = ServerCapabilities {
        tools: Some(ToolsCapability {
            list_changed: Some(true),
        }),
        prompts: Some(PromptsCapability {
            list_changed: Some(true),
        }),
        resources: Some(ResourcesCapability {
            subscribe: Some(true),
            list_changed: Some(true),
        }),
        logging: None,
        completions: None,
        tasks: None,
        experimental: Some(json!({"feature": "enabled"})),
    };

    let serialized =
        serde_json::to_string(&capabilities).expect("Capabilities serialization should succeed");
    assert!(serialized.contains("tools"));
    assert!(serialized.contains("prompts"));
    assert!(serialized.contains("resources"));
    assert!(serialized.contains("experimental"));
}

#[tokio::test]
async fn test_server_capabilities_none_enabled() {
    let capabilities = ServerCapabilities::default();

    let serialized =
        serde_json::to_string(&capabilities).expect("Capabilities serialization should succeed");
    // Should serialize empty object or skip None fields
    assert!(!serialized.contains("tools") || serialized.contains("null"));
}

#[tokio::test]
async fn structured_content_also_emits_text_json() {
    let structured = json!({"answer": 42});
    let result = ToolsCallResult::from_text("human", Some(structured.clone()), false);
    let content = result.content.expect("content");
    assert_eq!(content.len(), 2);
    assert!(matches!(&content[0], ContentPart::Text { text } if text == "human"));
    assert!(
        matches!(&content[1], ContentPart::Text { text } if text.as_str() == structured.to_string())
    );
    assert_eq!(result.structured_content, Some(structured));
}

#[tokio::test]
async fn icon_theme_roundtrip() {
    let icon = Icon {
        src: "https://example.com/icon.png".to_string(),
        mime_type: Some("image/png".to_string()),
        sizes: Some(vec!["48x48".to_string()]),
        theme: Some("dark".to_string()),
    };
    let encoded = serde_json::to_value(&icon).expect("encode");
    assert_eq!(encoded["theme"], "dark");
    let decoded: Icon = serde_json::from_value(encoded).expect("decode");
    assert_eq!(decoded.theme.as_deref(), Some("dark"));
}
