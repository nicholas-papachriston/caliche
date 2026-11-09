use async_trait::async_trait;
/// Tool system tests
use caliche::agent::tools::tool::ToolMeta;
use caliche::{Agent, Tool, ToolRegistry, ToolResult, Toolkit, testing::MockProvider};
use serde_json::json;
use std::sync::Arc;

/// Simple test tool
#[derive(Debug)]
struct AdditionTool;
impl ToolMeta for AdditionTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Adds two numbers together"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["a", "b"]
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
impl Tool for AdditionTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let a = args["a"].as_f64().unwrap_or(0.0);
        let b = args["b"].as_f64().unwrap_or(0.0);
        let result = a + b;

        ToolResult::success(json!({"result": result}).to_string())
    }
}

/// Tool that always fails
#[derive(Debug)]
struct FailingTool;
impl ToolMeta for FailingTool {
    fn name(&self) -> &str {
        "failing_tool"
    }

    fn description(&self) -> &str {
        "A tool that always fails"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
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
impl Tool for FailingTool {
    async fn execute(&self, _args: serde_json::Value) -> ToolResult {
        ToolResult::error("This tool always fails".to_string())
    }
}

#[tokio::test]
async fn test_tool_registry_creation() {
    let registry = ToolRegistry::new();
    assert_eq!(registry.list().len(), 0);
}

#[tokio::test]
async fn test_tool_registry_add_tool() {
    let registry = ToolRegistry::new();
    let tool: Arc<dyn Tool> = Arc::new(AdditionTool);

    registry.register(tool);
    assert_eq!(registry.list().len(), 1);
}

#[tokio::test]
async fn test_tool_registry_get_tool() {
    let registry = ToolRegistry::new();
    let tool: Arc<dyn Tool> = Arc::new(AdditionTool);

    registry.register(tool);

    let retrieved = registry.get("add");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name(), "add");
}

#[tokio::test]
async fn test_tool_registry_get_nonexistent_tool() {
    let registry = ToolRegistry::new();
    let retrieved = registry.get("nonexistent");
    assert!(retrieved.is_none());
}

#[tokio::test]
async fn test_tool_execution_success() {
    let tool = AdditionTool;
    let args = json!({"a": 5, "b": 3});

    let result = tool.execute(args).await;

    assert!(result.success);
    assert!(result.content.contains("8"));
}

#[tokio::test]
async fn test_tool_execution_failure() {
    let tool = FailingTool;
    let args = json!({});

    let result = tool.execute(args).await;

    assert!(!result.success);
    assert!(result.error.is_some());
}

#[tokio::test]
async fn test_tool_result_success() {
    let result = ToolResult::success("Success message".to_string());

    assert!(result.success);
    assert_eq!(result.content, "Success message");
}

#[tokio::test]
async fn test_tool_result_error() {
    let result = ToolResult::error("Error message".to_string());

    assert!(!result.success);
    assert_eq!(result.error, Some("Error message".to_string()));
}

#[tokio::test]
async fn test_tool_metadata() {
    let tool = AdditionTool;

    assert_eq!(tool.name(), "add");
    assert_eq!(tool.description(), "Adds two numbers together");

    let params = tool.parameters();
    assert!(params["properties"]["a"].is_object());
    assert!(params["properties"]["b"].is_object());
    assert_eq!(params["required"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_tool_registry_multiple_tools() {
    let registry = ToolRegistry::new();

    registry.register(Arc::new(AdditionTool));
    registry.register(Arc::new(FailingTool));

    assert_eq!(registry.list().len(), 2);
    assert!(registry.get("add").is_some());
    assert!(registry.get("failing_tool").is_some());
}

#[tokio::test]
async fn test_tool_definitions_for_llm() {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(AdditionTool));

    let tools_list = registry.list();
    assert_eq!(tools_list.len(), 1);

    let (name, description, _parameters) = &tools_list[0];
    assert_eq!(*name, "add");
    assert_eq!(*description, "Adds two numbers together");
}

#[cfg(feature = "tools-calculator")]
#[tokio::test]
async fn test_calculator_tool_basic() {
    use caliche::agent::tools::CalculatorTool;

    let tool = CalculatorTool;

    // Test addition
    let result = tool
        .execute(json!({"operation": "add", "a": 2, "b": 2}))
        .await;
    assert!(result.success);
    assert!(result.content.contains("4"));
}

#[cfg(feature = "tools-calculator")]
#[tokio::test]
async fn test_calculator_tool_multiply() {
    use caliche::agent::tools::CalculatorTool;

    let tool = CalculatorTool;

    // Test multiplication
    let result = tool
        .execute(json!({"operation": "multiply", "a": 15, "b": 2}))
        .await;
    assert!(result.success);
    assert!(result.content.contains("30"));
}

#[cfg(feature = "tools-calculator")]
#[tokio::test]
async fn test_calculator_tool_invalid() {
    use caliche::agent::tools::CalculatorTool;

    let tool = CalculatorTool;

    // Test missing required parameter
    let result = tool.execute(json!({"operation": "add", "a": 5})).await;
    assert!(!result.success);
    assert!(result.error.is_some());
}

#[cfg(feature = "tools-file")]
#[tokio::test]
async fn test_fs_tool_read() -> anyhow::Result<()> {
    use caliche::agent::tools::FsTool;

    // A fixture, not the repository manifest: a file over the auto-outline
    // threshold reads as an outline, so the test must not depend on its size.
    let dir = tempfile::tempdir()?;
    let file_path = dir.path().join("Cargo.toml");
    tokio::fs::write(&file_path, "[package]\nname = \"fixture\"\n").await?;

    let tool = FsTool::new();
    let result = tool.execute(json!({"op": "read", "path": file_path})).await;
    assert!(result.success, "{:?}", result.error);
    assert!(result.content.contains("[package]"));
    assert!(result.content.contains("name = \"fixture\""));
    Ok(())
}

#[cfg(feature = "tools-file")]
#[tokio::test]
async fn test_fs_tool_read_line_range_has_payload() -> anyhow::Result<()> {
    use caliche::agent::tools::FsTool;

    let dir = tempfile::tempdir()?;
    let file_path = dir.path().join("sample.txt");
    tokio::fs::write(&file_path, "one\ntwo\nthree\nfour\n").await?;

    let tool = FsTool::new();
    let result = tool
        .execute(json!({
            "op": "read",
            "path": file_path,
            "start_line": 2,
            "end_line": 3
        }))
        .await;

    assert!(result.success, "{:?}", result.error);
    assert!(result.content.contains("2:"));
    assert!(result.content.contains("two"));
    assert!(result.content.contains("three"));
    assert!(!result.content.contains("four"));
    let payload = result
        .typed_payload
        .ok_or_else(|| std::io::Error::other("typed payload"))?;
    assert_eq!(payload["op"], "read");
    assert_eq!(payload["returned_lines"], 2);
    assert_eq!(payload["cache_hit"], false);
    Ok(())
}

#[cfg(all(feature = "tools-file", feature = "tools-grep"))]
#[tokio::test]
async fn test_fs_tool_glob_respects_pattern_and_payload() -> anyhow::Result<()> {
    use caliche::agent::tools::FsTool;

    let dir = tempfile::tempdir()?;
    tokio::fs::write(dir.path().join("a.rs"), "fn a() {}\n").await?;
    tokio::fs::write(dir.path().join("b.txt"), "text\n").await?;

    let tool = FsTool::new();
    let result = tool
        .execute(json!({
            "op": "glob",
            "path": dir.path(),
            "pattern": "*.rs"
        }))
        .await;

    assert!(result.success, "{:?}", result.error);
    assert!(result.content.contains("a.rs"));
    assert!(!result.content.contains("b.txt"));
    let payload = result
        .typed_payload
        .ok_or_else(|| std::io::Error::other("typed payload"))?;
    assert_eq!(payload["op"], "glob");
    assert_eq!(payload["match_count"], 1);
    Ok(())
}

#[cfg(all(feature = "tools-file", feature = "tools-grep"))]
#[tokio::test]
async fn test_fs_tool_grep_files_with_matches_mode() -> anyhow::Result<()> {
    use caliche::agent::tools::FsTool;

    let dir = tempfile::tempdir()?;
    tokio::fs::write(dir.path().join("hit.rs"), "pub fn target() {}\n").await?;
    tokio::fs::write(dir.path().join("miss.rs"), "pub fn other() {}\n").await?;

    let tool = FsTool::new();
    let result = tool
        .execute(json!({
            "op": "grep",
            "path": dir.path(),
            "pattern": "target",
            "file_pattern": ".rs",
            "mode": "files_with_matches"
        }))
        .await;

    assert!(result.success, "{:?}", result.error);
    assert!(result.content.contains("hit.rs"));
    assert!(!result.content.contains("miss.rs"));
    let payload = result
        .typed_payload
        .ok_or_else(|| std::io::Error::other("typed payload"))?;
    assert_eq!(payload["op"], "grep");
    assert_eq!(payload["mode"], "files_with_matches");
    assert_eq!(payload["file_count"], 1);
    Ok(())
}

#[cfg(feature = "tools-file")]
#[tokio::test]
async fn test_fs_tool_list_extension_summary_and_payload() -> anyhow::Result<()> {
    use caliche::agent::tools::FsTool;

    let dir = tempfile::tempdir()?;
    tokio::fs::create_dir(dir.path().join("subdir")).await?;
    tokio::fs::write(dir.path().join("a.rs"), "fn a() {}\n").await?;
    tokio::fs::write(dir.path().join("b.txt"), "text\n").await?;

    let tool = FsTool::new();
    let result = tool
        .execute(json!({
            "op": "list",
            "path": dir.path(),
            "extensions_summary": true
        }))
        .await;

    assert!(result.success, "{:?}", result.error);
    assert!(result.content.contains("Extension summary"));
    assert!(result.content.contains(".rs: 1"));
    assert!(result.content.contains(".txt: 1"));
    let payload = result
        .typed_payload
        .ok_or_else(|| std::io::Error::other("typed payload"))?;
    assert_eq!(payload["op"], "list");
    assert_eq!(payload["entry_count"], 3);
    Ok(())
}

#[cfg(feature = "tools-file")]
#[tokio::test]
async fn test_fs_tool_read_query_cache_hit_payload() -> anyhow::Result<()> {
    use caliche::agent::tools::FsTool;

    let dir = tempfile::tempdir()?;
    let file_path = dir.path().join("cache.txt");
    tokio::fs::write(
        &file_path,
        "alpha target line\nbeta filler\ngamma target line\n",
    )
    .await?;

    let tool = FsTool::new();
    let args = json!({
        "op": "read",
        "path": file_path,
        "query": "target",
        "num_chunks": 1
    });
    let first = tool.execute(args.clone()).await;
    let second = tool.execute(args).await;

    assert!(first.success, "{:?}", first.error);
    assert!(second.success, "{:?}", second.error);
    let first_payload = first
        .typed_payload
        .as_ref()
        .ok_or_else(|| std::io::Error::other("first payload"))?;
    let second_payload = second
        .typed_payload
        .as_ref()
        .ok_or_else(|| std::io::Error::other("second payload"))?;
    assert_eq!(first_payload["cache_hit"], false);
    assert_eq!(second_payload["cache_hit"], true);
    Ok(())
}

// ============================================================================
// Toolkit Tests
// ============================================================================

#[tokio::test]
async fn test_toolkit_trait() {
    use async_trait::async_trait;
    use caliche::{Tool, Toolkit};
    use serde_json::json;
    use std::sync::Arc;

    struct TestToolkit;

    impl Toolkit for TestToolkit {
        fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
            #[derive(Debug)]
            struct TestTool;
            impl ToolMeta for TestTool {
                fn name(&self) -> &str {
                    "test_tool"
                }

                fn description(&self) -> &str {
                    "A test tool"
                }

                fn parameters(&self) -> serde_json::Value {
                    json!({})
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
                async fn execute(&self, _args: serde_json::Value) -> ToolResult {
                    ToolResult::success("test".to_string())
                }
            }
            vec![Arc::new(TestTool)]
        }

        fn name(&self) -> &str {
            "test"
        }
    }

    let toolkit = TestToolkit;
    assert_eq!(toolkit.name(), "test");
    assert_eq!(toolkit.get_tools().len(), 1);
}

#[tokio::test]
async fn test_filesystem_toolkit() {
    use caliche::agent::tools::FileSystemToolkit;

    let toolkit = FileSystemToolkit;
    assert_eq!(toolkit.name(), "filesystem");

    let tools: Vec<Arc<dyn Tool>> = toolkit.get_tools();
    assert!(!tools.is_empty());

    // Verify it includes expected tools
    assert!(tools.iter().any(|t| t.name() == "fs"));
}

#[tokio::test]
async fn test_toolkit_registration_with_builder() {
    use async_trait::async_trait;
    use caliche::{Tool, Toolkit};
    use serde_json::json;
    use std::sync::Arc;

    struct TestToolkit;

    impl Toolkit for TestToolkit {
        fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
            #[derive(Debug)]
            struct TestTool;
            impl ToolMeta for TestTool {
                fn name(&self) -> &str {
                    "test_tool"
                }

                fn description(&self) -> &str {
                    "A test tool"
                }

                fn parameters(&self) -> serde_json::Value {
                    json!({})
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
                async fn execute(&self, _args: serde_json::Value) -> ToolResult {
                    ToolResult::success("test".to_string())
                }
            }
            vec![Arc::new(TestTool)]
        }

        fn name(&self) -> &str {
            "test"
        }
    }

    let provider: Arc<dyn caliche::llm::Provider> = Arc::new(MockProvider::new(vec![]));
    let tools = ToolRegistry::new();

    let toolkit = TestToolkit;
    for tool in toolkit.get_tools() {
        tools.register(tool);
    }

    let tools = Arc::new(tools);
    let agent = Agent::new(provider, tools);

    // Verify tool was registered
    let tool_list = agent.list_tools();
    assert!(tool_list.iter().any(|(name, _, _)| *name == "test_tool"));
}

// ============================================================================
// Context Tools Tests
// ============================================================================

#[tokio::test]
async fn test_context_tools() {
    use caliche::SharedState;
    use caliche::agent::tools::{ContextGetTool, ContextSetTool};
    use serde_json::json;
    use std::sync::Arc;

    let shared_state = Arc::new(SharedState::new());
    shared_state
        .set("/order_id", json!("ORD-123"))
        .await
        .unwrap();

    let get_tool = ContextGetTool::new(Arc::clone(&shared_state));
    let set_tool = ContextSetTool::new(Arc::clone(&shared_state));

    // Test get tool
    let get_args = json!({ "path": "/order_id" });
    let get_result = get_tool.execute(get_args).await;
    assert!(get_result.success);
    assert!(get_result.content.contains("ORD-123"));

    // Test set tool
    let set_args = json!({
        "path": "/status",
        "value": "processing"
    });
    let set_result = set_tool.execute(set_args).await;
    assert!(set_result.success);

    // Verify the value was set
    let status = shared_state.get("/status").await.unwrap();
    assert_eq!(status, Some(json!("processing")));
}

#[tokio::test]
async fn test_context_tools_with_nested_paths() {
    use caliche::SharedState;
    use caliche::agent::tools::{ContextGetTool, ContextSetTool};
    use serde_json::json;
    use std::sync::Arc;

    let shared_state = Arc::new(SharedState::new());

    let get_tool = ContextGetTool::new(Arc::clone(&shared_state));
    let set_tool = ContextSetTool::new(Arc::clone(&shared_state));

    // Set nested value
    let set_args = json!({
        "path": "/order/items/0/name",
        "value": "Widget"
    });
    let set_result = set_tool.execute(set_args).await;
    assert!(set_result.success);

    // Get nested value
    let get_args = json!({ "path": "/order/items/0/name" });
    let get_result = get_tool.execute(get_args).await;
    assert!(get_result.success);
    assert!(get_result.content.contains("Widget"));
}

#[tokio::test]
async fn test_context_get_missing_path() {
    use caliche::SharedState;
    use caliche::agent::tools::ContextGetTool;
    use serde_json::json;
    use std::sync::Arc;

    let shared_state = Arc::new(SharedState::new());
    let get_tool = ContextGetTool::new(shared_state);

    // Get non-existent path should return a message indicating no value found
    let get_args = json!({ "path": "/nonexistent" });
    let get_result = get_tool.execute(get_args).await;
    assert!(get_result.success);
    // Should return a message indicating no value was found
    assert!(
        get_result.content.contains("No value found")
            || get_result.content.contains("nonexistent")
            || get_result.content == "null"
    );
}

#[cfg(feature = "tools-file")]
mod fs_edit_search_read {
    use caliche::Tool;
    use caliche::agent::tools::FsTool;
    use caliche::agent::tools::common::set_tool_workspace_root;
    use caliche::agent::tools::hashline::compute_line_hash;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    struct WorkspaceGuard;

    impl Drop for WorkspaceGuard {
        fn drop(&mut self) {
            set_tool_workspace_root(None);
        }
    }

    fn lock_workspace(dir: &Path) -> WorkspaceGuard {
        set_tool_workspace_root(Some(dir.to_path_buf()));
        WorkspaceGuard
    }

    #[expect(
        clippy::expect_used,
        reason = "test helper: a failure fails the calling test"
    )]
    async fn write_file(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        tokio::fs::write(&path, body).await.expect("write fixture");
        path
    }

    #[tokio::test]
    async fn hashline_delete_and_replace_accept_anchors_crlf_and_leading_space() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let path = write_file(
            dir.path(),
            "sample.txt",
            "keep_head\r\n    delete_me  \r\nreplace_me\r\nkeep_tail\r\n",
        )
        .await;
        let delete_hash = compute_line_hash("delete_me");
        let replace_hash = compute_line_hash("replace_me");
        let tool = FsTool::new();

        let deleted = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": [{
                    "type": "hashline_delete",
                    "start_line": "2",
                    "end_line": "2",
                    "start_hash": format!("2:{delete_hash}| delete_me"),
                    "end_hash": delete_hash
                }]
            }))
            .await;
        assert!(deleted.success, "delete: {:?}", deleted.error);

        let replaced = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": format!(
                    "[{{\"type\":\"hashline_replace\",\"start_line\":2,\"end_line\":2,\"start_hash\":\"{replace_hash}\",\"end_hash\":\"{replace_hash}\",\"content\":\"replaced\"}}]"
                )
            }))
            .await;
        assert!(replaced.success, "replace: {:?}", replaced.error);

        let text = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(text.contains("keep_head"), "{text}");
        assert!(text.contains("replaced"), "{text}");
        assert!(text.contains("keep_tail"), "{text}");
        assert!(!text.contains("delete_me"), "{text}");
        assert!(!text.contains("replace_me"), "{text}");
    }

    #[tokio::test]
    async fn hashline_delete_relocates_a_stale_line_and_reports_a_real_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let path = write_file(
            dir.path(),
            "shifted.txt",
            "inserted\nalpha_unique\nbeta_unique\n",
        )
        .await;
        let beta = compute_line_hash("beta_unique");
        let tool = FsTool::new();
        let relocated = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": [{
                    "type": "hashline_delete",
                    "start_line": 2,
                    "end_line": 2,
                    "start_hash": beta,
                    "end_hash": beta
                }]
            }))
            .await;
        assert!(relocated.success, "relocate: {:?}", relocated.error);
        let text = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(!text.contains("beta_unique"), "{text}");
        assert!(text.contains("alpha_unique"), "{text}");

        let inserted_hash = compute_line_hash("inserted");
        let alpha_hash = compute_line_hash("alpha_unique");
        let wrong = ["0000", "0001", "0002"]
            .into_iter()
            .find(|candidate| *candidate != inserted_hash && *candidate != alpha_hash)
            .expect("unused hash");
        let missed = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": [{
                    "type": "hashline_replace",
                    "start_line": 1,
                    "end_line": 1,
                    "start_hash": wrong,
                    "end_hash": wrong,
                    "content": "nope"
                }]
            }))
            .await;
        assert!(!missed.success, "mismatch must fail");
        let error = missed.error.unwrap_or_default();
        assert!(error.contains(wrong), "{error}");
        assert!(error.contains("Re-read"), "{error}");
        assert!(
            error.contains(&inserted_hash),
            "error should show the current hash: {error}"
        );
        let after = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(after.contains("inserted"), "{after}");
        assert!(!after.contains("nope"), "{after}");
    }

    #[tokio::test]
    async fn ranged_read_uses_offset_limit_and_string_line_numbers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let mut body = String::new();
        for line in 1..=30 {
            body.push_str(&format!("line-{line:02} token\n"));
        }
        let path = write_file(dir.path(), "lines.txt", &body).await;
        let tool = FsTool::new();

        let sliced = tool
            .execute(json!({
                "op": "read",
                "path": &path,
                "offset": 10,
                "limit": 3
            }))
            .await;
        assert!(sliced.success, "{:?}", sliced.error);
        assert!(sliced.content.contains("line-10"), "{}", sliced.content);
        assert!(sliced.content.contains("line-11"), "{}", sliced.content);
        assert!(sliced.content.contains("line-12"), "{}", sliced.content);
        assert!(
            !sliced.content.contains("line-01"),
            "offset read stuck on the file head: {}",
            sliced.content
        );
        let payload = sliced.typed_payload.expect("payload");
        assert_eq!(payload["start_line"], 10);
        assert_eq!(payload["end_line"], 12);

        let strings = tool
            .execute(json!({
                "op": "read",
                "path": &path,
                "start_line": "15",
                "end_line": "16"
            }))
            .await;
        assert!(strings.success, "{:?}", strings.error);
        assert!(strings.content.contains("line-15"), "{}", strings.content);
        assert!(strings.content.contains("line-16"), "{}", strings.content);
        assert!(
            !strings.content.contains("line-01"),
            "string line range stuck on the file head: {}",
            strings.content
        );

        let bad = tool
            .execute(json!({
                "op": "read",
                "path": &path,
                "start_line": "head"
            }))
            .await;
        assert!(
            !bad.success,
            "unparseable start_line must not return the file"
        );
        let error = bad.error.unwrap_or_default();
        assert!(error.contains("start_line"), "{error}");
        assert!(!bad.content.contains("line-01"), "{}", bad.content);
    }

    #[cfg(feature = "tools-grep")]
    #[tokio::test]
    async fn quoted_and_invalid_regex_patterns_search_as_literals() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let path = write_file(
            dir.path(),
            "code.rs",
            "fn foo(bar) {}\nlet literal = foo(;\n",
        )
        .await;
        let tool = FsTool::new();

        let quoted = tool
            .execute(json!({
                "op": "grep",
                "path": &path,
                "pattern": "\"fn foo(bar)\"",
                "context_lines": 0
            }))
            .await;
        assert!(quoted.success, "{:?}", quoted.error);
        assert!(
            quoted.content.contains("fn foo(bar)"),
            "quoted pattern was dropped: {}",
            quoted.content
        );
        assert!(
            quoted.content.contains("quotes"),
            "expected the quote-stripping note: {}",
            quoted.content
        );

        let invalid = tool
            .execute(json!({
                "op": "grep",
                "path": &path,
                "pattern": "foo(",
                "context_lines": 0
            }))
            .await;
        assert!(invalid.success, "{:?}", invalid.error);
        assert!(
            invalid.content.contains("foo("),
            "invalid regex was not searched as a literal: {}",
            invalid.content
        );
        assert!(
            invalid.content.contains("not valid regex"),
            "host grep dropped the invalid-regex notice: {}",
            invalid.content
        );
        assert!(
            invalid.content.contains("searched once as a literal"),
            "host grep dropped the literal-search notice: {}",
            invalid.content
        );
    }

    #[expect(
        clippy::panic,
        reason = "test helper: a failure fails the calling test"
    )]
    fn line_with_low_byte(low: &str, avoid: &[String]) -> String {
        assert!(
            avoid
                .iter()
                .all(|line| &compute_line_hash(line)[2..] != low),
            "fixture already contains low byte {low}"
        );
        for index in 0..50_000 {
            let text = format!("low-{low}-{index}");
            if &compute_line_hash(&text)[2..] == low {
                return text;
            }
        }
        panic!("no fixture line with low byte {low}");
    }

    #[expect(
        clippy::panic,
        reason = "test helper: a failure fails the calling test"
    )]
    fn filler_without_low_byte(slot: usize, low: &str) -> String {
        for salt in 0..1_000 {
            let text = format!("pad-{slot}-{salt}");
            if &compute_line_hash(&text)[2..] != low {
                return text;
            }
        }
        panic!("no filler for slot {slot}");
    }

    #[tokio::test]
    async fn malformed_hash_does_not_delete_a_colliding_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let victim = line_with_low_byte("12", &[]);
        let mut lines = Vec::with_capacity(20);
        for slot in 1..=20 {
            if slot == 18 {
                lines.push(victim.clone());
            } else {
                lines.push(filler_without_low_byte(slot, "12"));
            }
        }
        let kept_twelve = lines[11].clone();
        let body = lines.join("\n") + "\n";
        let path = write_file(dir.path(), "malformed.txt", &body).await;
        let tool = FsTool::new();

        for raw in ["#12", "L42:ab", "line 42", "42-ab"] {
            let deleted = tool
                .execute(json!({
                    "op": "edit",
                    "path": &path,
                    "operations": [{
                        "type": "hashline_delete",
                        "start_line": 12,
                        "end_line": 12,
                        "start_hash": raw,
                        "end_hash": raw
                    }]
                }))
                .await;
            assert!(
                !deleted.success,
                "{raw} edited the file: {:?}",
                deleted.error
            );
            let error = deleted.error.unwrap_or_default();
            assert!(
                error.contains("Invalid hash"),
                "{raw} was scraped into a hash: {error}"
            );
            assert!(error.contains(raw), "{error}");
            let text = tokio::fs::read_to_string(&path).await.expect("read");
            assert!(
                text.contains(&victim),
                "{raw} deleted the colliding line: {text}"
            );
            assert!(
                text.contains(&kept_twelve),
                "{raw} deleted the hinted line: {text}"
            );
        }
    }

    #[tokio::test]
    async fn legacy_two_hex_does_not_overwrite_another_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let hint = filler_without_low_byte(1, "ab");
        let second = filler_without_low_byte(2, "ab");
        let third = filler_without_low_byte(3, "ab");
        let victim = line_with_low_byte("ab", &[hint.clone(), second.clone(), third.clone()]);
        let hint_text = hint.clone();
        let lines = vec![hint, second, third, victim.clone()];
        let full = compute_line_hash(&victim);
        let legacy = full[2..].to_string();
        assert_eq!(legacy, "ab");
        for (index, line) in lines.iter().enumerate() {
            if index == 3 {
                continue;
            }
            assert_ne!(
                &compute_line_hash(line)[2..],
                legacy,
                "fixture low-byte collision on line {}",
                index + 1
            );
        }
        let body = lines.join("\n") + "\n";
        let path = write_file(dir.path(), "legacy.txt", &body).await;
        let tool = FsTool::new();

        let missed = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": [{
                    "type": "hashline_replace",
                    "start_line": 1,
                    "end_line": 1,
                    "start_hash": legacy,
                    "end_hash": legacy,
                    "content": "OVERWROTE"
                }]
            }))
            .await;
        assert!(!missed.success, "2-hex relocated: {:?}", missed.error);
        let error = missed.error.unwrap_or_default();
        assert!(error.contains("not relocated"), "{error}");
        let text = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(text.contains(&victim), "{text}");
        assert!(text.contains(&hint_text), "{text}");
        assert!(!text.contains("OVERWROTE"), "{text}");

        let relocated = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": [{
                    "type": "hashline_replace",
                    "start_line": 1,
                    "end_line": 1,
                    "start_hash": full,
                    "end_hash": full,
                    "content": "moved"
                }]
            }))
            .await;
        assert!(
            relocated.success,
            "4-hex unique match should still relocate: {:?}",
            relocated.error
        );
        let after = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(after.contains("moved"), "{after}");
        assert!(!after.contains(&victim), "{after}");
        assert!(after.contains(&hint_text), "{after}");
    }

    #[tokio::test]
    async fn loose_whitespace_retry_refuses_two_candidates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let body = "keep\n    alpha\n  alpha\n";
        let path = write_file(dir.path(), "space.txt", &body).await;
        let hash = compute_line_hash("alpha");
        let tool = FsTool::new();
        let missed = tool
            .execute(json!({
                "op": "edit",
                "path": &path,
                "operations": [{
                    "type": "hashline_replace",
                    "start_line": 1,
                    "end_line": 1,
                    "start_hash": hash,
                    "end_hash": hash,
                    "content": "OVERWROTE"
                }]
            }))
            .await;
        assert!(
            !missed.success,
            "ambiguous whitespace edit applied: {:?}",
            missed.error
        );
        let error = missed.error.unwrap_or_default();
        assert!(error.contains("multiple lines"), "{error}");
        let text = tokio::fs::read_to_string(&path).await.expect("read");
        assert_eq!(text, body);
    }

    #[tokio::test]
    async fn offset_past_eof_does_not_return_the_file_head() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _guard = lock_workspace(dir.path());
        let mut body = String::new();
        for line in 1..=30 {
            body.push_str(&format!("line-{line:02} token\n"));
        }
        let path = write_file(dir.path(), "tail.txt", &body).await;
        let tool = FsTool::new();
        let past = tool
            .execute(json!({
                "op": "read",
                "path": &path,
                "offset": 500,
                "limit": 10
            }))
            .await;
        assert!(
            !past.success,
            "offset past EOF returned the file: {}",
            past.content
        );
        assert!(
            !past.content.contains("line-01"),
            "offset past EOF returned the file head: {}",
            past.content
        );
        let error = past.error.unwrap_or_default();
        assert!(
            error.contains("beyond"),
            "expected an end-of-file error, got: {error}"
        );
    }
}
