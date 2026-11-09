use anyhow::Result;
use async_trait::async_trait;
/// Workflow engine tests
use caliche::orchestration::{
    ExecutionContext, Workflow, WorkflowBuilder, WorkflowEngine, WorkflowNodeExecutor,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn test_workflow_builder_creation() {
    let builder = WorkflowBuilder::new("test_workflow");
    let workflow = builder.build();

    // Workflow ID is auto-generated UUID, not the name
    assert!(!workflow.id().is_empty());
}

#[tokio::test]
async fn test_workflow_builder_with_description() {
    let builder =
        WorkflowBuilder::new("test_workflow").with_description("Test workflow description");

    let workflow = builder.build();
    assert!(!workflow.description().is_empty());
}

#[tokio::test]
async fn test_workflow_builder_add_agent() {
    let builder = WorkflowBuilder::new("test").add_agent("agent1", "Do something");

    let workflow = builder.build();
    assert_eq!(workflow.nodes().len(), 1);
    // Node IDs are also auto-generated UUIDs
    assert!(!workflow.nodes()[0].id.is_empty());
}

#[tokio::test]
async fn test_workflow_builder_sequential() {
    let builder = WorkflowBuilder::new("sequential")
        .add_agent("agent1", "Step 1")
        .then("agent2", "Step 2")
        .then("agent3", "Step 3");

    let workflow = builder.build();

    assert_eq!(workflow.nodes().len(), 3);
    assert_eq!(workflow.edges().len(), 2); // 1->2, 2->3
}

#[tokio::test]
async fn test_workflow_builder_parallel() {
    let builder = WorkflowBuilder::new("parallel")
        .add_agent("starter", "Start")
        .parallel(vec![
            ("worker1".to_string(), "Task 1".to_string()),
            ("worker2".to_string(), "Task 2".to_string()),
            ("worker3".to_string(), "Task 3".to_string()),
        ]);

    let workflow = builder.build();

    // starter + parallel_node + 3 workers + merge_node
    assert!(workflow.nodes().len() >= 5);
}

#[tokio::test]
async fn test_workflow_engine_creation() {
    let engine = WorkflowEngine::new();

    // Engine should be created successfully
    assert!(engine.list_workflows().is_empty());
}

#[tokio::test]
async fn test_workflow_engine_register() {
    let engine = WorkflowEngine::new();

    let workflow = WorkflowBuilder::new("test")
        .add_agent("agent1", "Do something")
        .build();

    let workflow_id = workflow.id().to_string();

    engine.register_workflow(Arc::new(workflow)).await.unwrap();

    let workflows = engine.list_workflows();
    assert_eq!(workflows.len(), 1);
    assert_eq!(workflows[0], workflow_id);
}

#[tokio::test]
async fn test_workflow_validation() {
    let workflow = WorkflowBuilder::new("test")
        .add_agent("agent1", "Step 1")
        .then("agent2", "Step 2")
        .build();

    let result = workflow.validate();
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_execution_context_creation() {
    let context = ExecutionContext::new("workflow123".to_string());

    assert_eq!(context.workflow_id, "workflow123");
    assert!(context.start_time <= chrono::Utc::now());
    assert!(context.end_time.is_none());
    assert!(!context.execution_id.is_empty());
}

#[tokio::test]
async fn test_execution_context_variables() {
    let mut context = ExecutionContext::new("test".to_string());

    // Set variable
    context.set_variable("key1".to_string(), serde_json::json!("value1"));

    // Get variable
    let value = context.get_variable("key1");
    assert!(value.is_some());
    assert_eq!(value.unwrap(), &serde_json::json!("value1"));

    // Get non-existent variable
    let missing = context.get_variable("nonexistent");
    assert!(missing.is_none());
}

#[tokio::test]
async fn test_execution_context_multiple_variables() {
    let mut context = ExecutionContext::new("test".to_string());

    context.set_variable("str".to_string(), serde_json::json!("hello"));
    context.set_variable("num".to_string(), serde_json::json!(42));
    context.set_variable("bool".to_string(), serde_json::json!(true));
    context.set_variable("obj".to_string(), serde_json::json!({"nested": "value"}));

    assert_eq!(context.variables.len(), 4);
}

#[tokio::test]
async fn test_workflow_multiple_register() {
    let engine = WorkflowEngine::new();

    let workflow1 = Arc::new(WorkflowBuilder::new("workflow1").build());
    let workflow2 = Arc::new(WorkflowBuilder::new("workflow2").build());
    let workflow3 = Arc::new(WorkflowBuilder::new("workflow3").build());

    engine.register_workflow(workflow1).await.unwrap();
    engine.register_workflow(workflow2).await.unwrap();
    engine.register_workflow(workflow3).await.unwrap();

    let workflows = engine.list_workflows();
    assert_eq!(workflows.len(), 3);
}

#[tokio::test]
async fn test_workflow_get_by_id() {
    let engine = WorkflowEngine::new();

    let workflow = WorkflowBuilder::new("test123").build();
    let workflow_id = workflow.id().to_string();

    engine.register_workflow(Arc::new(workflow)).await.unwrap();

    let retrieved = engine.get_workflow(&workflow_id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id(), workflow_id);
}

#[derive(Default)]
struct EchoPromptExecutor;

#[async_trait]
impl WorkflowNodeExecutor for EchoPromptExecutor {
    async fn execute_agent(
        &self,
        prompt: &str,
        _agent_id: Option<&str>,
        _capability: Option<&str>,
        _context: &ExecutionContext,
    ) -> Result<(Value, HashMap<String, Value>)> {
        Ok((Value::String(prompt.to_string()), HashMap::new()))
    }

    async fn transform(
        &self,
        _function: &str,
        _context: &ExecutionContext,
        last_output: Option<&Value>,
    ) -> Result<Value> {
        Ok(last_output.cloned().unwrap_or(Value::Null))
    }

    async fn merge_custom(
        &self,
        _name: &str,
        inputs: &Value,
        _context: &ExecutionContext,
    ) -> Result<Value> {
        Ok(inputs.clone())
    }
}

#[tokio::test]
async fn sequential_agent_nodes_resolve_mustache_node_name() {
    let engine = WorkflowEngine::with_executor(Arc::new(EchoPromptExecutor));
    let workflow = WorkflowBuilder::new("named_outputs")
        .add_agent("research", "find facts about rust")
        .then("writeup", "summarize: {{research}}")
        .build();

    let mut context = ExecutionContext::new(workflow.id());
    let result = workflow
        .execute(&engine, &mut context)
        .await
        .expect("named template workflow should execute");

    assert_eq!(
        result,
        Value::String("summarize: find facts about rust".into())
    );
    assert_eq!(
        context.get_variable("research"),
        Some(&Value::String("find facts about rust".into()))
    );
    assert_eq!(
        context.get_variable("last_output"),
        Some(&Value::String("summarize: find facts about rust".into()))
    );
}

#[tokio::test]
async fn json_workflow_with_stable_ids_resolves_mustache_name() {
    let json = r#"{
      "id": "named-outputs",
      "name": "named-outputs",
      "description": "",
      "nodes": [
        {
          "id": "research",
          "name": "research",
          "node_type": {
            "Agent": {
              "agent_id": null,
              "capability": null,
              "prompt_template": "find facts about rust"
            }
          },
          "metadata": {}
        },
        {
          "id": "writeup",
          "name": "writeup",
          "node_type": {
            "Agent": {
              "agent_id": null,
              "capability": null,
              "prompt_template": "summarize: {{research}}"
            }
          },
          "metadata": {}
        }
      ],
      "edges": [
        {
          "from_node": "research",
          "to_node": "writeup",
          "condition": null,
          "label": null
        }
      ],
      "entry_point": null
    }"#;
    let workflow = caliche::orchestration::BuiltWorkflow::from_json(json)
        .expect("CLI-style workflow JSON should parse");
    let engine = WorkflowEngine::with_executor(Arc::new(EchoPromptExecutor));
    let mut context = ExecutionContext::new(workflow.id());
    let result = workflow
        .execute(&engine, &mut context)
        .await
        .expect("named JSON workflow should execute");
    assert_eq!(
        result,
        Value::String("summarize: find facts about rust".into())
    );
}

#[tokio::test]
async fn sequential_agent_nodes_keep_last_output_alias() {
    let engine = WorkflowEngine::with_executor(Arc::new(EchoPromptExecutor));
    let workflow = WorkflowBuilder::new("last_output_alias")
        .add_agent("research", "find facts about rust")
        .then("writeup", "echo {{__last_output}}")
        .build();

    let mut context = ExecutionContext::new(workflow.id());
    let result = workflow
        .execute(&engine, &mut context)
        .await
        .expect("__last_output should still resolve");

    assert_eq!(result, Value::String("echo find facts about rust".into()));
}
