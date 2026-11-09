//! Tests for agent lifecycle control features:
//! - Lifecycle hooks (sync and async)
//! - Permission modes
//! - Allowed tools filtering
//! - In-process MCP servers
//! - Session forking and merging

use async_trait::async_trait;
use caliche::agent::tools::tool::ToolMeta;
use caliche::prelude::{Message, ProviderType};
use caliche::{
    AgentBuilder, Tool, ToolResult,
    agent::hooks::permission::PermissionMode,
    agent::hooks::{HookContext, HookMatcher, HookResult, LifecycleHook},
    agent::session::AgentSession,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

/// Test hook that tracks calls and can deny based on tool name
#[derive(Debug, Clone)]
struct TestHook {
    pre_calls: Arc<Mutex<Vec<String>>>,
    post_calls: Arc<Mutex<Vec<String>>>,
    deny_tool: Option<String>,
}

impl TestHook {
    fn new() -> Self {
        Self {
            pre_calls: Arc::new(Mutex::new(Vec::new())),
            post_calls: Arc::new(Mutex::new(Vec::new())),
            deny_tool: None,
        }
    }

    fn with_deny_tool(tool_name: String) -> Self {
        Self {
            pre_calls: Arc::new(Mutex::new(Vec::new())),
            post_calls: Arc::new(Mutex::new(Vec::new())),
            deny_tool: Some(tool_name),
        }
    }

    #[expect(
        clippy::unwrap_used,
        reason = "test helper: a failure fails the calling test"
    )]
    fn get_pre_calls(&self) -> Vec<String> {
        self.pre_calls.lock().unwrap().clone()
    }

    #[expect(
        clippy::unwrap_used,
        reason = "test helper: a failure fails the calling test"
    )]
    fn get_post_calls(&self) -> Vec<String> {
        self.post_calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl LifecycleHook for TestHook {
    #[expect(
        clippy::unwrap_used,
        reason = "test helper: a failure fails the calling test"
    )]
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        self.pre_calls
            .lock()
            .unwrap()
            .push(ctx.tool_name.clone().into_owned());

        if let Some(ref deny_tool) = self.deny_tool {
            if ctx.tool_name == *deny_tool {
                return HookResult::deny("Tool denied by test hook");
            }
        }

        HookResult::allow()
    }

    #[expect(
        clippy::unwrap_used,
        reason = "test helper: a failure fails the calling test"
    )]
    async fn on_post_tool_use(&self, ctx: &HookContext<'_>, _result: &ToolResult) -> HookResult {
        self.post_calls
            .lock()
            .unwrap()
            .push(ctx.tool_name.clone().into_owned());
        HookResult::allow()
    }
}

/// Async test hook that simulates async operations
#[derive(Debug, Clone)]
struct AsyncTestHook {
    call_count: Arc<Mutex<usize>>,
}

impl AsyncTestHook {
    fn new() -> Self {
        Self {
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    #[expect(
        clippy::unwrap_used,
        reason = "test helper: a failure fails the calling test"
    )]
    fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

#[async_trait]
impl LifecycleHook for AsyncTestHook {
    #[expect(
        clippy::unwrap_used,
        reason = "test helper: a failure fails the calling test"
    )]
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        // Simulate async operation
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        *self.call_count.lock().unwrap() += 1;
        HookResult::allow()
    }
}

/// Test tool for MCP server
#[derive(Debug)]
#[cfg_attr(
    not(feature = "mcp"),
    expect(dead_code, reason = "only the mcp tests build it")
)]
struct TestTool {
    name: String,
}
impl ToolMeta for TestTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Test tool"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": { "type": "string" }
            }
        })
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::owned(self.name().to_string())
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::owned(self.description().to_string())
    }
}

#[async_trait]
impl Tool for TestTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let input = args
            .get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        ToolResult::success(format!("Result: {}", input))
    }
}

#[tokio::test]
async fn test_permission_modes() {
    // Test that permission modes can be set (builder methods return Self)
    let _builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .permission_mode(PermissionMode::Default);

    let _builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .permission_mode(PermissionMode::AcceptEdits);

    let _builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .permission_mode(PermissionMode::AcceptAll);

    let _builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .permission_mode(PermissionMode::DenyAll);
}

#[tokio::test]
async fn test_allowed_tools() {
    // Test that allowed_tools can be set
    let _builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .allowed_tools(&["calculator", "fs"])
        .enable_all_default_tools();
}

#[tokio::test]
async fn test_hook_registration() {
    let hook = Arc::new(TestHook::new());
    let hook_clone = hook;

    let builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .hook(HookMatcher::tool("calculator"), hook_clone);

    // Hook should be registered (we can't easily verify this without building,
    // but we can check that the builder accepts it)
    let _builder = builder;
}

#[tokio::test]
async fn test_hook_matcher_tool() {
    let matcher = HookMatcher::tool("test_tool");
    match matcher {
        HookMatcher::Tool { name } => assert_eq!(name, "test_tool"),
        _ => panic!("Expected Tool matcher"),
    }
}

#[tokio::test]
async fn test_hook_matcher_all() {
    let matcher = HookMatcher::all();
    match matcher {
        HookMatcher::All => (),
        _ => panic!("Expected All matcher"),
    }
}

#[tokio::test]
async fn test_hook_result_allow() {
    let result = HookResult::allow();
    assert!(result.is_allowed());
    assert!(!result.is_denied());
}

#[tokio::test]
async fn test_hook_result_deny() {
    let result = HookResult::deny("Test reason");
    assert!(!result.is_allowed());
    assert!(result.is_denied());
}

#[tokio::test]
async fn test_hook_context_creation() {
    let args = json!({"arg": "value"});
    let ctx = HookContext::new("test_tool".to_string(), &args, "call_123".to_string());

    assert_eq!(ctx.tool_name, "test_tool");
    assert_eq!(ctx.tool_use_id, "call_123");
    assert_eq!(
        ctx.tool_input.get("arg").and_then(|v| v.as_str()),
        Some("value")
    );
}

#[tokio::test]
async fn test_async_hook() {
    let hook = Arc::new(AsyncTestHook::new());
    let hook_clone = hook.clone();

    // Test that async hook can be called
    let empty = json!({});
    let ctx = HookContext::new("test_tool".to_string(), &empty, "call_1".to_string());

    let result = hook_clone.on_pre_tool_use(&ctx).await;
    assert!(result.is_allowed());
    assert_eq!(hook.get_call_count(), 1);
}

#[tokio::test]
async fn test_hook_pre_and_post_calls() {
    let hook = Arc::new(TestHook::new());
    let hook_clone = hook.clone();

    let empty = json!({});
    let ctx = HookContext::new("test_tool".to_string(), &empty, "call_1".to_string());

    // Call pre hook
    let result = hook_clone.on_pre_tool_use(&ctx).await;
    assert!(result.is_allowed());

    // Call post hook
    let tool_result = ToolResult::success("test".to_string());
    let _ = hook_clone.on_post_tool_use(&ctx, &tool_result).await;

    assert_eq!(hook.get_pre_calls(), vec!["test_tool"]);
    assert_eq!(hook.get_post_calls(), vec!["test_tool"]);
}

#[tokio::test]
async fn test_hook_deny() {
    let hook = Arc::new(TestHook::with_deny_tool("dangerous_tool".to_string()));
    let hook_clone = hook.clone();

    let empty = json!({});
    let ctx_allowed = HookContext::new("safe_tool".to_string(), &empty, "call_1".to_string());

    let ctx_denied = HookContext::new("dangerous_tool".to_string(), &empty, "call_2".to_string());

    // Safe tool should be allowed
    let result1 = hook_clone.on_pre_tool_use(&ctx_allowed).await;
    assert!(result1.is_allowed());

    // Dangerous tool should be denied
    let result2 = hook_clone.on_pre_tool_use(&ctx_denied).await;
    assert!(result2.is_denied());
}

#[tokio::test]
async fn test_session_creation() {
    let session = AgentSession::new("test-session");
    assert_eq!(session.id, "test-session");
}

#[tokio::test]
async fn test_session_with_history() {
    let history = vec![Message::user("Hello"), Message::assistant("Hi there")];

    let session = AgentSession::with_history("test-session", history);
    assert_eq!(session.id, "test-session");

    let retrieved_history = session.get_history().unwrap();
    assert_eq!(retrieved_history.len(), 2);
}

#[tokio::test]
async fn test_session_add_message() {
    let session = AgentSession::new("test-session");

    session.add_message(Message::user("Test message")).unwrap();

    let history = session.get_history().unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].extract_text(), "Test message");
}

#[tokio::test]
async fn test_session_clear_history() {
    let session = AgentSession::new("test-session");

    session.add_message(Message::user("Message 1")).unwrap();
    session.add_message(Message::user("Message 2")).unwrap();

    assert_eq!(session.get_history().unwrap().len(), 2);

    session.clear_history().unwrap();
    assert_eq!(session.get_history().unwrap().len(), 0);
}

#[tokio::test]
async fn test_session_fork() {
    let main_session = AgentSession::new("main");
    main_session
        .add_message(Message::user("Initial message"))
        .unwrap();

    let forked = main_session.fork("forked").await.unwrap();

    assert_eq!(forked.id, "forked");
    assert_ne!(forked.id, main_session.id);

    // Forked session should have copied history
    let forked_history = forked.get_history().unwrap();
    assert_eq!(forked_history.len(), 1);
}

#[tokio::test]
async fn test_session_merge() {
    let mut main_session = AgentSession::new("main");
    main_session
        .add_message(Message::user("Main message 1"))
        .unwrap();

    let forked = main_session.fork("forked").await.unwrap();
    forked.add_message(Message::user("Forked message")).unwrap();
    forked
        .add_message(Message::assistant("Forked response"))
        .unwrap();

    // Merge forked into main
    main_session.merge(&forked).await.unwrap();

    let merged_history = main_session.get_history().unwrap();
    // Forked session has 1 original message + 2 new messages = 3 total
    // Main session has 1 message, merge adds all 3 from fork = 4 total
    assert_eq!(merged_history.len(), 4);
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn test_in_process_mcp_server_creation() {
    use caliche::mcp::InProcessMcpServer;

    let server = InProcessMcpServer::builder("test-server")
        .version("1.0.0")
        .add_tool(TestTool {
            name: "test_tool".to_string(),
        })
        .build()
        .unwrap();

    assert_eq!(server.name(), "test-server");
    assert_eq!(server.version(), "1.0.0");

    let tools = server.list_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "test_tool");
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn test_in_process_mcp_server_tool_execution() {
    use caliche::mcp::InProcessMcpServer;

    let server = InProcessMcpServer::builder("test-server")
        .version("1.0.0")
        .add_tool(TestTool {
            name: "test_tool".to_string(),
        })
        .build()
        .unwrap();

    let result = server
        .call_tool("test_tool", json!({"input": "test_value"}))
        .await
        .unwrap();

    assert!(result.content.is_some());
    if let Some(content) = result.content {
        assert!(!content.is_empty());
    }
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn test_agent_with_mcp_server() {
    use caliche::mcp::InProcessMcpServer;

    let server = InProcessMcpServer::builder("test-tools")
        .version("1.0.0")
        .add_tool(TestTool {
            name: "test_tool".to_string(),
        })
        .build()
        .unwrap();

    let builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .mcp_server("test-tools", server);

    // Builder should accept MCP server
    let _builder = builder;
}

#[tokio::test]
async fn test_combined_features() {
    let hook = Arc::new(TestHook::new());
    let hook_clone = hook;

    // Test that all features can be combined
    let _builder = AgentBuilder::new()
        .provider(ProviderType::LMStudio)
        .allowed_tools(&["calculator"])
        .permission_mode(PermissionMode::AcceptEdits)
        .hook(HookMatcher::all(), hook_clone);
}

#[tokio::test]
async fn test_hook_registry_matching() {
    use caliche::agent::hooks::HookRegistry;

    let mut registry = HookRegistry::new();
    let hook1 = Arc::new(TestHook::new());
    let hook2 = Arc::new(TestHook::new());

    // Register tool-specific hook
    registry.register(HookMatcher::tool("calculator"), hook1);

    // Register "all" hook
    registry.register(HookMatcher::all(), hook2);

    // Get matching hooks for calculator
    let matching = registry.get_matching("calculator");
    assert_eq!(matching.len(), 2); // Both hooks should match

    // Get matching hooks for other tool
    let matching_other = registry.get_matching("fs");
    assert_eq!(matching_other.len(), 1); // Only "all" hook should match
}

#[tokio::test]
async fn test_permission_mode_default() {
    assert_eq!(PermissionMode::default(), PermissionMode::Default);
}

#[tokio::test]
async fn test_hook_result_with_output() {
    use caliche::agent::hooks::permission::PermissionDecision;

    let output = json!({"custom": "data"});
    let result = HookResult::with_output(PermissionDecision::Allow, output.clone());

    assert!(result.is_allowed());
    assert_eq!(result.hook_specific_output, Some(output));
}
