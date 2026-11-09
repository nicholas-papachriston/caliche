//! CLI handlers for orchestration and multi-agent patterns.

use anyhow::{Context, Result};
use async_trait::async_trait;
use caliche::agent::service::AgentService;
use caliche::llm::ProviderType;
use caliche::orchestration::workflow::MergeStrategy;
use caliche::orchestration::{
    BuiltWorkflow, DebatePattern, ExecutionContext, FanOutCollectPattern, HierarchicalPattern,
    PipelinePattern, SupervisorPattern, Workflow, WorkflowEngine, WorkflowNodeExecutor,
    builtin_workflow_merge_custom, builtin_workflow_transform,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

fn parse_workflow_definition(content: &str, path: &str) -> Result<BuiltWorkflow> {
    let trimmed = content.trim_start();
    let likely_yaml = path.ends_with(".yaml") || path.ends_with(".yml");
    let likely_json = path.ends_with(".json") || trimmed.starts_with('{');

    if likely_json && let Ok(workflow) = serde_json::from_str::<BuiltWorkflow>(content) {
        return Ok(workflow);
    }
    if likely_yaml && let Ok(workflow) = serde_norway::from_str::<BuiltWorkflow>(content) {
        return Ok(workflow);
    }

    serde_json::from_str::<BuiltWorkflow>(content)
        .or_else(|_| serde_norway::from_str::<BuiltWorkflow>(content))
        .with_context(|| {
            format!(
                "Invalid workflow format in '{}'. Expected BuiltWorkflow JSON or YAML.",
                path
            )
        })
}

/// Config for CLI workflow execution (provider, model, tools).
/// Uses env-based defaults when not set.
#[derive(Clone, Default)]
pub struct OrchestrationConfig {
    pub provider: Option<ProviderType>,
    pub model: Option<String>,
    pub tools: Option<Vec<String>>,
}

/// Build a WorkflowEngine with CliWorkflowNodeExecutor from config.
fn build_engine(config: &OrchestrationConfig) -> WorkflowEngine {
    let executor = CliWorkflowNodeExecutor::new()
        .with_provider(config.provider)
        .with_model(config.model.clone())
        .with_tools(config.tools.clone());
    WorkflowEngine::with_executor(Arc::new(executor))
}

/// Run a workflow synchronously and return the final output.
async fn run_workflow_sync(
    engine: &WorkflowEngine,
    workflow: Arc<dyn Workflow>,
    mut context: ExecutionContext,
) -> Result<Value> {
    workflow.execute(engine, &mut context).await
}

/// Execute a multi-agent pattern using real orchestration.
///
/// Provider/model defaults are copied from `service` so orchestration nodes
/// inherit the same configuration the REPL or one-shot agent is using.
pub async fn execute_multi_agent_pattern(
    pattern: &str,
    task: &str,
    num_agents: usize,
    service: AgentService,
) -> anyhow::Result<Vec<String>> {
    let config = OrchestrationConfig {
        provider: Some(service.provider()),
        model: Some(service.model().to_string()),
        tools: None,
    };
    let engine = build_engine(&config);

    let workflow: Arc<dyn Workflow> = match pattern.to_lowercase().as_str() {
        "debate" => {
            let p = DebatePattern::new("cli_debate", task, num_agents, 2);
            Arc::new(p)
        }
        "supervisor" => {
            let p = SupervisorPattern::new(
                "cli_supervisor",
                format!("Coordinate workers on: {}", task),
                num_agents,
            );
            Arc::new(p)
        }
        "pipeline" => {
            let p = PipelinePattern::new("cli_pipeline")
                .add_stage("stage_0", format!("Process the following task: {}", task));
            Arc::new(p)
        }
        "hierarchical" => {
            let p = HierarchicalPattern::new("cli_hierarchical").add_layer(
                "layer_0",
                num_agents,
                format!("Contribute to the task: {}", task),
            );
            Arc::new(p)
        }
        "fan_out_collect" | "fanout_collect" | "fanout" => {
            let branch_count = num_agents.max(2);
            let mut pattern = FanOutCollectPattern::new("cli_fan_out_collect")
                .with_description("Fan-out workers and collect results")
                .with_collect_strategy(MergeStrategy::Concatenate);
            for idx in 0..branch_count {
                pattern = pattern.add_branch(
                    format!("worker_{}", idx + 1),
                    format!("Analyze this task from perspective {}: {}", idx + 1, task),
                );
            }
            pattern = pattern
                .with_final_aggregator("Synthesize collected branch outputs into one response");
            Arc::new(pattern)
        }
        _ => {
            anyhow::bail!(
                "Unknown pattern '{}'. Supported: pipeline, debate, supervisor, hierarchical, fan_out_collect",
                pattern
            );
        }
    };

    let workflow_id = workflow.id().to_string();
    let ctx = ExecutionContext::new(workflow_id);
    let result = run_workflow_sync(&engine, workflow, ctx).await?;

    let out = match &result {
        Value::String(s) => s.clone(),
        _ => match serde_json::to_string_pretty(&result) {
            Ok(s) => s,
            Err(_) => result.to_string(),
        },
    };
    Ok(vec![out])
}

/// Load workflow from JSON file, execute it, and print the result.
pub async fn load_workflow(path: &str, config: OrchestrationConfig) -> anyhow::Result<()> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read workflow file: {}", path))?;

    let workflow_def = parse_workflow_definition(&content, path)?;

    workflow_def.validate()?;

    let engine = build_engine(&config);
    let workflow: Arc<dyn Workflow> = Arc::new(workflow_def);
    let workflow_id = workflow.id().to_string();
    let ctx = ExecutionContext::new(workflow_id.clone());

    println!("Executing workflow: {} ({})", workflow.name(), workflow_id);
    let result = run_workflow_sync(&engine, workflow, ctx).await?;

    println!("\n=== Workflow result ===");
    match &result {
        Value::String(s) => println!("{}", s),
        _ => {
            let s = match serde_json::to_string_pretty(&result) {
                Ok(x) => x,
                Err(_) => result.to_string(),
            };
            println!("{}", s);
        }
    }
    Ok(())
}

/// Display multi-agent results
pub fn display_multi_agent_results(pattern: &str, results: &[String]) {
    println!("\n=== Multi-Agent Pattern: {} ===", pattern);
    for (idx, result) in results.iter().enumerate() {
        println!("\nStep {}:", idx + 1);
        println!("{}", result);
    }
}

/// CLI workflow node executor
///
/// Implements WorkflowNodeExecutor for CLI workflow execution.
/// Creates AgentService instances on-demand for agent nodes.
#[derive(Clone)]
pub struct CliWorkflowNodeExecutor {
    /// Default provider to use for agent nodes
    pub default_provider: Option<ProviderType>,
    /// Default model to use for agent nodes
    pub default_model: Option<String>,
    /// Default tools to enable for agent nodes
    pub default_tools: Option<Vec<String>>,
}

impl CliWorkflowNodeExecutor {
    /// Create a new CLI workflow executor with defaults
    pub fn new() -> Self {
        Self {
            default_provider: None,
            default_model: None,
            default_tools: None,
        }
    }

    /// Set the default provider
    pub fn with_provider(mut self, provider: Option<ProviderType>) -> Self {
        self.default_provider = provider;
        self
    }

    /// Set the default model
    pub fn with_model(mut self, model: Option<String>) -> Self {
        self.default_model = model;
        self
    }

    /// Set the default tools
    pub fn with_tools(mut self, tools: Option<Vec<String>>) -> Self {
        self.default_tools = tools;
        self
    }

    /// Extract tools from workflow context
    fn tools_from_context(&self, context: &ExecutionContext) -> Option<Vec<String>> {
        // First check if tools are specified in context
        if let Some(tools) = context
            .get_variable("__tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
        {
            if !tools.is_empty() {
                return Some(tools);
            }
        }
        // Fall back to default tools
        self.default_tools.clone()
    }
}

impl Default for CliWorkflowNodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorkflowNodeExecutor for CliWorkflowNodeExecutor {
    async fn execute_agent(
        &self,
        prompt: &str,
        _agent_id: Option<&str>,
        _capability: Option<&str>,
        context: &ExecutionContext,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let provider = self.default_provider;
        let model = self.default_model.clone();
        let tools = Some(self.tools_from_context(context).unwrap_or_default());

        // Create agent service
        let mut service = AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(provider, model.clone())
                .with_tool_names_or_defaults(tools),
        )
        .await?;
        service.agent_mut().set_system_prompt(
            "You are executing a single workflow step. Follow the user prompt exactly. \
             Do not add extra commentary, lists, or offers of further help. \
             If the prompt asks you to repeat or return text, return only that text.",
        );

        // Execute the prompt
        let response = service.chat(prompt).await?;
        let text = response.text().to_string();

        // Build metadata
        let mut meta = HashMap::new();
        if let Some(ref provider) = provider {
            meta.insert(
                "provider".to_string(),
                Value::String(format!("{:?}", provider)),
            );
        }
        if let Some(ref model) = model {
            meta.insert("model".to_string(), Value::String(model.clone()));
        }

        Ok((Value::String(text), meta))
    }

    async fn transform(
        &self,
        function: &str,
        context: &ExecutionContext,
        last_output: Option<&Value>,
    ) -> Result<Value> {
        builtin_workflow_transform(function, context, last_output)
    }

    async fn merge_custom(
        &self,
        name: &str,
        inputs: &Value,
        context: &ExecutionContext,
    ) -> Result<Value> {
        builtin_workflow_merge_custom(name, inputs, context)
    }
}

#[cfg(test)]
mod parse_workflow_tests {
    use super::parse_workflow_definition;
    use caliche::Workflow;
    use caliche::orchestration::WorkflowBuilder;

    #[test]
    fn parse_json_by_path_hint() {
        let built = WorkflowBuilder::new("cli_wf")
            .add_agent("step1", "hello")
            .with_entry_point("step1")
            .build();
        let Ok(json) = built.to_json() else {
            panic!("workflow should serialize to json");
        };
        let Ok(parsed) = parse_workflow_definition(&json, "def.json") else {
            panic!("workflow json should parse");
        };
        assert_eq!(parsed.name(), "cli_wf");
    }

    #[test]
    fn parse_yaml_by_extension() {
        let built = WorkflowBuilder::new("yaml_wf")
            .add_agent("n1", "prompt")
            .with_entry_point("n1")
            .build();
        let Ok(yaml) = serde_norway::to_string(&built) else {
            panic!("workflow should serialize to yaml");
        };
        let Ok(parsed) = parse_workflow_definition(&yaml, "flow.yaml") else {
            panic!("workflow yaml should parse");
        };
        assert_eq!(parsed.name(), "yaml_wf");
    }

    #[test]
    fn parse_invalid_reports_path() {
        let Err(err) = parse_workflow_definition("%%%", "my/workflow.json") else {
            panic!("invalid workflow source should not parse");
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("my/workflow.json"));
    }
}
