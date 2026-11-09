use crate::orchestration::workflow::{
    ExecutionContext, MergeStrategy, NodeType, Workflow, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Hierarchical pattern: multi-level agent teams with layers of coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalPattern {
    id: String,
    name: String,
    description: String,
    layers: Vec<HierarchicalLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalLayer {
    pub name: String,
    pub agent_count: usize,
    pub capability: Option<String>,
    pub prompt_template: String,
}

impl HierarchicalPattern {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            description: "Hierarchical multi-agent workflow".to_string(),
            layers: Vec::new(),
        }
    }

    pub fn add_layer(
        mut self,
        name: impl Into<String>,
        agent_count: usize,
        prompt_template: impl Into<String>,
    ) -> Self {
        self.layers.push(HierarchicalLayer {
            name: name.into(),
            agent_count,
            capability: None,
            prompt_template: prompt_template.into(),
        });
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    fn generate_nodes(&self) -> Vec<WorkflowNode> {
        let mut nodes = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Add parallel node for layer
            let parallel_node =
                WorkflowNode::new(format!("layer_{}_parallel", layer_idx), NodeType::Parallel);
            nodes.push(parallel_node);

            // Add agent nodes for layer
            for agent_idx in 0..layer.agent_count {
                let agent_node = WorkflowNode::new(
                    format!("layer_{}_agent_{}", layer_idx, agent_idx),
                    NodeType::Agent {
                        agent_id: None,
                        capability: layer.capability.clone(),
                        prompt_template: layer.prompt_template.clone(),
                    },
                );
                nodes.push(agent_node);
            }

            // Add merge node for layer
            let merge_node = WorkflowNode::new(
                format!("layer_{}_merge", layer_idx),
                NodeType::Merge {
                    strategy: MergeStrategy::AllComplete,
                },
            );
            nodes.push(merge_node);
        }

        nodes
    }

    fn generate_edges(&self, nodes: &[WorkflowNode]) -> Vec<WorkflowEdge> {
        let mut edges = Vec::new();
        let mut node_idx = 0;

        for layer in &self.layers {
            let parallel_id = &nodes[node_idx].id;
            node_idx += 1;

            let agent_ids: Vec<_> = (0..layer.agent_count)
                .map(|_| {
                    let id = nodes[node_idx].id.clone();
                    node_idx += 1;
                    id
                })
                .collect();

            let merge_id = &nodes[node_idx].id;
            node_idx += 1;

            // Connect parallel to agents
            for agent_id in &agent_ids {
                edges.push(WorkflowEdge::new(parallel_id.clone(), agent_id.clone()));
            }

            // Connect agents to merge
            for agent_id in &agent_ids {
                edges.push(WorkflowEdge::new(agent_id.clone(), merge_id.clone()));
            }

            // Connect merge to next layer's parallel (if exists)
            if node_idx < nodes.len() {
                edges.push(WorkflowEdge::new(
                    merge_id.clone(),
                    nodes[node_idx].id.clone(),
                ));
            }
        }

        edges
    }
}

#[async_trait]
impl Workflow for HierarchicalPattern {
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

        // Each layer starts with a `NodeType::Parallel` fan-out node that forces control flow
        // to its `Merge` node once branches complete; `execute_dag`'s topological scheduler
        // does not support forced control flow (see `fanout_collect.rs` for the full
        // rationale), so this must run via the state-machine executor.
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
