use crate::orchestration::workflow::{
    ExecutionContext, MergeStrategy, NodeType, Workflow, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanOutCollectPattern {
    id: String,
    name: String,
    description: String,
    branches: Vec<FanOutBranch>,
    collect_strategy: MergeStrategy,
    final_aggregator_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanOutBranch {
    pub name: String,
    pub prompt_template: String,
}

impl FanOutCollectPattern {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            description: "Fan-out branch execution followed by collection".to_owned(),
            branches: Vec::new(),
            collect_strategy: MergeStrategy::AllComplete,
            final_aggregator_prompt: None,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn add_branch(
        mut self,
        name: impl Into<String>,
        prompt_template: impl Into<String>,
    ) -> Self {
        self.branches.push(FanOutBranch {
            name: name.into(),
            prompt_template: prompt_template.into(),
        });
        self
    }

    pub fn with_collect_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.collect_strategy = strategy;
        self
    }

    pub fn with_final_aggregator(mut self, prompt_template: impl Into<String>) -> Self {
        self.final_aggregator_prompt = Some(prompt_template.into());
        self
    }

    fn generate_nodes(&self) -> Vec<WorkflowNode> {
        let mut nodes = Vec::new();
        nodes.push(WorkflowNode::new("fan_out", NodeType::Parallel));
        for branch in &self.branches {
            nodes.push(WorkflowNode::new(
                branch.name.clone(),
                NodeType::Agent {
                    agent_id: None,
                    capability: None,
                    prompt_template: branch.prompt_template.clone(),
                },
            ));
        }
        nodes.push(WorkflowNode::new(
            "collect",
            NodeType::Merge {
                strategy: self.collect_strategy.clone(),
            },
        ));
        if let Some(prompt_template) = &self.final_aggregator_prompt {
            nodes.push(WorkflowNode::new(
                "aggregate",
                NodeType::Agent {
                    agent_id: None,
                    capability: None,
                    prompt_template: prompt_template.clone(),
                },
            ));
        }
        nodes
    }

    fn generate_edges(&self, nodes: &[WorkflowNode]) -> Vec<WorkflowEdge> {
        let mut edges = Vec::new();
        if self.branches.is_empty() {
            return edges;
        }

        let parallel_id = nodes[0].id.clone();
        let merge_index = 1 + self.branches.len();
        let merge_id = nodes[merge_index].id.clone();

        for branch_node in nodes.iter().skip(1).take(self.branches.len()) {
            edges.push(WorkflowEdge::new(
                parallel_id.clone(),
                branch_node.id.clone(),
            ));
            edges.push(WorkflowEdge::new(branch_node.id.clone(), merge_id.clone()));
        }

        if self.final_aggregator_prompt.is_some() {
            let aggregate_id = nodes[merge_index + 1].id.clone();
            edges.push(WorkflowEdge::new(merge_id, aggregate_id));
        }

        edges
    }
}

#[async_trait]
impl Workflow for FanOutCollectPattern {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn nodes(&self) -> &[WorkflowNode] {
        &[]
    }

    fn edges(&self) -> &[WorkflowEdge] {
        &[]
    }

    async fn execute(
        &self,
        engine: &WorkflowEngine,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value> {
        let nodes = self.generate_nodes();
        let edges = self.generate_edges(&nodes);
        // This graph starts with a `NodeType::Parallel` fan-out node, which forces control
        // flow directly to its `Merge` node once branches complete. `execute_dag`'s pure
        // topological scheduler does not support that (it independently re-visits branch
        // nodes and rejects forced control flow), so this must run via the state-machine
        // executor, which honors forced transitions and never double-executes branch nodes.
        let Some(entry_point) = nodes.first().map(|n| n.id.clone()) else {
            return Ok(serde_json::json!(null));
        };
        let results = engine
            .execute_state_machine(&nodes, &edges, &entry_point, context)
            .await?;
        if let Some(last_result) = results.last() {
            Ok(last_result
                .output
                .clone()
                .unwrap_or(serde_json::json!(null)))
        } else {
            Ok(serde_json::json!(null))
        }
    }
}
