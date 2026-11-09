//! Harness module integration tests (events, state, planning, persistence, etc.).

use anyhow::{Result, anyhow};
use caliche::agent::tools::ToolRegistry;
use caliche::agent::tools::{CalculatorTool, ToolCall};
use caliche::harness::coding::{
    CodingContextPolicy, CodingMode, CodingTaskRequest, VerificationPolicy,
};
use caliche::harness::config::PermissionModeConfig;
use caliche::harness::{
    AgentEvent, ContextManager, ConversationExport, DiffTracker, EventStream, ParallelToolExecutor,
    PlanStatus, ProjectType, SessionPersistence, SnapshotChangeKind, StateManager, StepStatus,
    StreamingResponseHandler, TaskPlan, TaskPlanner, ToolCallAnalysis, detect_language,
    detect_thinking_blocks, extract_code_blocks,
};
use caliche::llm::ProviderType;
use caliche::messages::{Message, TokenUsage};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::{TempDir, tempdir};
use tokio::sync::broadcast;
use uuid::Uuid;

#[tokio::test]
async fn test_event_stream_broadcast() {
    let stream = EventStream::new();
    let mut rx1 = stream.subscribe();
    let mut rx2 = stream.subscribe();

    stream.emit(AgentEvent::Info {
        message: "test".to_string(),
    });

    let event1 = rx1.recv().await.unwrap();
    let event2 = rx2.recv().await.unwrap();

    assert!(matches!(event1, AgentEvent::Info { .. }));
    assert!(matches!(event2, AgentEvent::Info { .. }));
}

#[test]
fn test_event_serialization() {
    let event = AgentEvent::tool_queued(
        "tool-1",
        "bash",
        &serde_json::json!({"command": "ls"}),
        "act-1",
    );
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("tool_queued"));
    assert!(json.contains("tool-1"));
}

#[tokio::test]
async fn test_session_end_event_broadcast_and_serialization() {
    let usage = TokenUsage {
        prompt_tokens: Some(10),
        completion_tokens: Some(20),
        total_tokens: Some(30),
        ..TokenUsage::default()
    };
    let event = AgentEvent::session_end("sess-end-1", 4, usage);

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("session_end"));
    assert!(json.contains("sess-end-1"));

    let stream = EventStream::new();
    let mut rx = stream.subscribe();
    stream.emit(event);

    let received = rx.recv().await.unwrap();
    match received {
        AgentEvent::SessionEnd {
            session_id,
            total_turns,
            total_usage,
            ..
        } => {
            assert_eq!(session_id, "sess-end-1");
            assert_eq!(total_turns, 4);
            assert_eq!(total_usage.total_tokens, Some(30));
        }
        other => panic!("expected SessionEnd, got {other:?}"),
    }
}

#[tokio::test]
async fn test_parallel_executor() {
    let (sender, mut receiver) = broadcast::channel(100);
    let executor = ParallelToolExecutor::new(sender, 5);

    let registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool));

    let calls = vec![
        ToolCall {
            name: beef::lean::Cow::borrowed("calculator"),
            arguments: serde_json::json!({"operation": "add", "a": 1, "b": 1}),
        },
        ToolCall {
            name: beef::lean::Cow::borrowed("calculator"),
            arguments: serde_json::json!({"operation": "add", "a": 2, "b": 2}),
        },
    ];

    let results = executor.execute_batch(&registry, calls, None, None).await;

    assert_eq!(results.len(), 2);
    for (name, result) in &results {
        assert_eq!(name, "calculator");
        assert!(result.success);
    }

    let mut event_count = 0;
    while let Ok(event) = receiver.try_recv() {
        event_count += 1;
        match event {
            AgentEvent::ToolQueued { .. } => {}
            AgentEvent::ToolExecuting { .. } => {}
            AgentEvent::ToolComplete { .. } => {}
            _ => panic!("Unexpected event type"),
        }
    }
    assert!(event_count >= 4);
}

#[test]
fn test_tool_call_analysis() {
    let calls = vec![
        ToolCall {
            name: beef::lean::Cow::borrowed("fs"),
            arguments: serde_json::json!({"op":"write","path": "/tmp/test.txt", "content": "hello"}),
        },
        ToolCall {
            name: beef::lean::Cow::borrowed("bash"),
            arguments: serde_json::json!({"command": "ls"}),
        },
    ];

    let analysis = ToolCallAnalysis::analyze(&calls);
    assert!(analysis.has_file_modifications);
    assert!(analysis.has_dangerous_ops);
    assert_eq!(analysis.files_to_modify.len(), 1);
    assert!(analysis.needs_permission());
}

#[test]
fn test_safe_tool_analysis() {
    let calls = vec![
        ToolCall {
            name: beef::lean::Cow::borrowed("fs"),
            arguments: serde_json::json!({"op":"read","path": "/tmp/test.txt"}),
        },
        ToolCall {
            name: beef::lean::Cow::borrowed("calculator"),
            arguments: serde_json::json!({"expression": "1+1"}),
        },
    ];

    let analysis = ToolCallAnalysis::analyze(&calls);
    assert!(!analysis.has_file_modifications);
    assert!(!analysis.has_dangerous_ops);
    assert!(!analysis.needs_permission());
}

#[tokio::test]
async fn test_context_manager_creation() {
    let tmp_dir = TempDir::new().unwrap();
    let manager = ContextManager::new(tmp_dir.path().to_path_buf());
    assert_eq!(manager.workspace_root(), tmp_dir.path());
}

#[test]
fn test_project_detection_rust() {
    let tmp_dir = TempDir::new().unwrap();
    fs::write(tmp_dir.path().join("Cargo.toml"), "[package]").unwrap();

    let mut manager = ContextManager::new(tmp_dir.path().to_path_buf());
    let project_type = manager.detect_project().unwrap();
    assert_eq!(project_type, ProjectType::Rust);
}

#[test]
fn test_language_detection() {
    assert_eq!(detect_language("rs"), Some("rust"));
    assert_eq!(detect_language("py"), Some("python"));
    assert_eq!(detect_language("ts"), Some("typescript"));
    assert_eq!(detect_language("unknown"), None);
}

#[test]
fn test_checkpoint_creation() {
    let mut manager = StateManager::new();
    let id = manager.checkpoint("test");
    let checkpoints = manager.list_checkpoints();
    assert_eq!(checkpoints.len(), 1);
    assert_eq!(checkpoints[0].label, "test");
    assert_eq!(checkpoints[0].id, id);
}

#[test]
fn test_file_version_tracking() {
    let mut manager = StateManager::new();
    let path = PathBuf::from("/tmp/test.txt");

    manager.record_file_write(&path, Some("old content"), "new content");

    let versions = manager.file_history(&path).unwrap();
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[0].content, Some("old content".to_string()));
}

#[tokio::test]
async fn test_undo_last() {
    let tmp_dir = TempDir::new().unwrap();
    let file_path = tmp_dir.path().join("test.txt");

    fs::write(&file_path, "original").unwrap();

    let mut manager = StateManager::new();
    manager.record_file_write(&file_path, Some("original"), "modified");
    fs::write(&file_path, "modified").unwrap();

    let action = manager.undo_last().unwrap();
    assert_eq!(action.path, file_path);
    assert_eq!(action.content, Some("original".to_string()));

    let content = fs::read_to_string(&file_path).unwrap();
    assert_eq!(content, "original");
}

#[test]
fn test_conversation_export_json() {
    let export = ConversationExport::new("test-session").with_messages(&[
        Message::user("Hello".to_string()),
        Message::assistant("Hi there!".to_string()),
    ]);

    let json = export.to_json().unwrap();
    assert!(json.contains("test-session"));
    assert!(json.contains("Hello"));

    let parsed = ConversationExport::from_json(&json).unwrap();
    assert_eq!(parsed.session_id, "test-session");
    assert_eq!(parsed.messages.len(), 2);
}

#[test]
fn test_conversation_export_yaml() {
    let export = ConversationExport::new("test-session")
        .with_messages(&[Message::user("Hello".to_string())]);

    let yaml = export.to_yaml().unwrap();
    assert!(yaml.contains("test-session"));

    let parsed = ConversationExport::from_yaml(&yaml).unwrap();
    assert_eq!(parsed.session_id, "test-session");
}

#[test]
fn test_session_persistence() {
    let tmp_dir = TempDir::new().unwrap();
    let persistence = SessionPersistence::new(tmp_dir.path().to_path_buf());

    let export =
        ConversationExport::new("session-1").with_messages(&[Message::user("Test".to_string())]);

    let path = persistence.save(&export).unwrap();
    assert!(path.exists());

    let loaded = persistence.load("session-1").unwrap();
    assert_eq!(loaded.session_id, "session-1");

    let sessions = persistence.list_sessions().unwrap();
    assert_eq!(sessions.len(), 1);

    persistence.delete("session-1").unwrap();
    assert!(!path.exists());
}

#[test]
fn test_task_plan_creation() {
    let mut plan = TaskPlan::new("Implement feature X");
    plan.add_step("Create the file");
    plan.add_step("Write the code");
    plan.add_step("Add tests");
    plan.finalize();

    assert_eq!(plan.steps.len(), 3);
    assert_eq!(plan.status, PlanStatus::Ready);
    assert_eq!(plan.progress(), 0);
}

#[test]
fn test_task_plan_progress() {
    let mut plan = TaskPlan::new("Test task");
    plan.add_step("Step 1");
    plan.add_step("Step 2");
    plan.finalize();

    plan.status = PlanStatus::Executing;
    plan.complete_current_step("Done step 1");

    assert_eq!(plan.progress(), 50);
    assert_eq!(plan.steps[0].status, StepStatus::Completed);
}

#[test]
fn test_plan_parsing() {
    let planner = TaskPlanner::new();

    let output = r#"1. Read the existing file
2. Modify the code
3. Write the updated file
4. Run tests"#;

    let plan = planner.parse_plan("Update file", output);

    assert_eq!(plan.steps.len(), 4);
    assert_eq!(plan.steps[0].description, "Read the existing file");
    assert!(plan.steps[0].suggested_tools.contains(&"fs".to_string()));
}

#[test]
fn test_complexity_estimation() {
    assert_eq!(TaskPlanner::estimate_complexity("read a file"), 1);
    assert_eq!(
        TaskPlanner::estimate_complexity("implement a new feature"),
        3
    );
    assert_eq!(
        TaskPlanner::estimate_complexity("refactor all the tests"),
        5
    );
}

#[test]
fn preserves_original_before_across_multiple_edits_in_one_message() {
    let mut tracker = DiffTracker::new();
    let message_id = Uuid::now_v7();
    tracker.start_message(message_id);

    let path = std::path::Path::new("/tmp/example.txt");
    tracker.record_before(path, Some(b"first"));
    tracker.record_after(path, Some(b"second"));
    tracker.record_before(path, Some(b"second"));
    tracker.record_after(path, Some(b"third"));
    tracker.finish_message();

    let diffs = tracker.diffs_for_session();
    assert_eq!(diffs.len(), 1);
    assert_eq!(diffs[0].snapshots.len(), 1);
    let snapshot = &diffs[0].snapshots[0];
    assert_eq!(snapshot.before.as_deref(), Some(&b"first"[..]));
    assert_eq!(snapshot.after.as_deref(), Some(&b"third"[..]));
    assert_eq!(snapshot.change_kind(), SnapshotChangeKind::Modified);
}

#[test]
fn clears_redo_history_after_new_changes_are_committed() -> Result<()> {
    let temp = tempdir()?;
    let path = temp.path().join("redo.txt");
    fs::write(&path, "v1")?;

    let mut tracker = DiffTracker::new();
    tracker.start_message(Uuid::now_v7());
    tracker.record_before(&path, Some(b"v1"));
    fs::write(&path, "v2")?;
    tracker.record_after(&path, Some(b"v2"));
    tracker.finish_message();

    let _ = tracker.undo_last()?;
    assert!(tracker.can_redo());

    tracker.start_message(Uuid::now_v7());
    tracker.record_before(&path, Some(b"v1"));
    fs::write(&path, "v3")?;
    tracker.record_after(&path, Some(b"v3"));
    tracker.finish_message();

    assert!(!tracker.can_redo());
    Ok(())
}

#[test]
fn undo_to_message_reverts_target_and_newer() -> Result<()> {
    let temp = tempdir()?;
    let path_a = temp.path().join("a.txt");
    let path_b = temp.path().join("b.txt");
    fs::write(&path_a, "a1")?;
    fs::write(&path_b, "b1")?;

    let mut tracker = DiffTracker::new();
    let id_a = Uuid::now_v7();
    tracker.start_message(id_a);
    tracker.record_before(&path_a, Some(b"a1"));
    fs::write(&path_a, "a2")?;
    tracker.record_after(&path_a, Some(b"a2"));
    tracker.finish_message();

    let id_b = Uuid::now_v7();
    tracker.start_message(id_b);
    tracker.record_before(&path_b, Some(b"b1"));
    fs::write(&path_b, "b2")?;
    tracker.record_after(&path_b, Some(b"b2"));
    tracker.finish_message();

    let reverted = tracker.undo_to_message(id_a)?;
    assert_eq!(reverted.len(), 2);
    assert_eq!(fs::read_to_string(&path_a)?, "a1");
    assert_eq!(fs::read_to_string(&path_b)?, "b1");
    assert!(tracker.diffs_for_session().is_empty());
    Ok(())
}

#[test]
fn undo_and_redo_support_deleted_files() -> Result<()> {
    let temp = tempdir()?;
    let path = temp.path().join("delete.txt");
    fs::write(&path, "keep me")?;

    let mut tracker = DiffTracker::new();
    tracker.start_message(Uuid::now_v7());
    tracker.record_before(&path, Some(b"keep me"));
    fs::remove_file(&path)?;
    tracker.record_after(&path, None);
    tracker.finish_message();

    let diff = match tracker.undo_last()? {
        Some(diff) => diff,
        None => return Err(anyhow!("expected an undo diff")),
    };
    assert_eq!(diff.snapshots[0].change_kind(), SnapshotChangeKind::Deleted);
    assert_eq!(fs::read_to_string(&path)?, "keep me");

    let _ = tracker.redo_last()?;
    assert!(!path.exists());
    Ok(())
}

#[test]
fn test_extract_code_blocks() {
    let content = r#"
Here's some code:

```rust
fn main() {
    println!("Hello");
}
```

And more text.

```python
print("world")
```
"#;

    let blocks = extract_code_blocks(content);
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].0, Some("rust".to_string()));
    assert!(blocks[0].1.contains("println"));
    assert_eq!(blocks[1].0, Some("python".to_string()));
    assert!(blocks[1].1.contains("print"));
}

#[test]
fn test_detect_thinking_blocks() {
    let content = r#"
<thinking>
Let me think about this...
The solution involves X and Y.
</thinking>

Here's my answer.
"#;

    let blocks = detect_thinking_blocks(content);
    assert_eq!(blocks.len(), 1);
    assert!(blocks[0].contains("think about this"));
}

#[tokio::test]
async fn test_streaming_handler() {
    let event_stream = EventStream::new();
    let mut rx = event_stream.subscribe();

    let handler = StreamingResponseHandler::new(event_stream.clone());

    handler.process_chunk("Hello ").await;
    handler.process_chunk("world!").await;

    let content = handler.full_content().await;
    assert_eq!(content, "Hello world!");

    let event1 = rx.try_recv().unwrap();
    assert!(matches!(event1, AgentEvent::TextChunk { content } if content == "Hello "));
}

#[test]
fn coding_request_maps_to_harness_config() {
    let request = CodingTaskRequest {
        task: "task".to_string(),
        workspace_root: Some(PathBuf::from("/tmp/workspace")),
        provider: Some(ProviderType::OpenAI),
        model: Some("gpt-test".to_string()),
        mode: CodingMode::Edit,
        permission_mode: PermissionModeConfig::AcceptEdits,
        max_turns: 7,
        tools: Some(vec!["bash".to_string()]),
        verify: VerificationPolicy::Auto,
        context: CodingContextPolicy::default(),
    };

    let config = request.to_harness_config();
    assert_eq!(config.provider, Some(ProviderType::OpenAI));
    assert_eq!(config.model.as_deref(), Some("gpt-test"));
    assert_eq!(config.workspace_root, Some(PathBuf::from("/tmp/workspace")));
    assert_eq!(config.max_turns, 7);
    assert_eq!(config.permission_mode, PermissionModeConfig::AcceptEdits);
    assert_eq!(config.tools, Some(vec!["bash".to_string()]));
}
