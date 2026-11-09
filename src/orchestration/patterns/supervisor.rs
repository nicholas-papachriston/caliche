use crate::orchestration::workflow::{
    ExecutionContext, NodeType, Workflow, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Supervisor pattern: one coordinator agent assigns tasks to multiple worker agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorPattern {
    id: String,
    name: String,
    description: String,
    supervisor_prompt: String,
    worker_count: usize,
    worker_capability: Option<String>,
}

impl SupervisorPattern {
    pub fn new(
        name: impl Into<String>,
        supervisor_prompt: impl Into<String>,
        worker_count: usize,
    ) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            description: "Supervisor pattern workflow".to_owned(),
            supervisor_prompt: supervisor_prompt.into(),
            worker_count,
            worker_capability: None,
        }
    }

    pub fn with_worker_capability(mut self, capability: impl Into<String>) -> Self {
        self.worker_capability = Some(capability.into());
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Generate workflow nodes
    fn generate_nodes(&self) -> Vec<WorkflowNode> {
        let mut nodes = Vec::new();

        // Supervisor node
        let supervisor_node = WorkflowNode::new(
            "supervisor",
            NodeType::Agent {
                agent_id: None,
                capability: Some("Coordination".to_owned()),
                prompt_template: self.supervisor_prompt.clone(),
            },
        );
        nodes.push(supervisor_node);

        // Parallel node for worker distribution
        let parallel_node = WorkflowNode::new("parallel_workers", NodeType::Parallel);
        nodes.push(parallel_node);

        // Worker nodes
        for i in 0..self.worker_count {
            let worker_node = WorkflowNode::new(
                format!("worker_{}", i),
                NodeType::Agent {
                    agent_id: None,
                    capability: self.worker_capability.clone(),
                    prompt_template: format!("Execute subtask {}", i),
                },
            );
            nodes.push(worker_node);
        }

        // Merge node
        let merge_node = WorkflowNode::new(
            "merge_results",
            NodeType::Merge {
                strategy: crate::orchestration::workflow::MergeStrategy::AllComplete,
            },
        );
        nodes.push(merge_node);

        // Final aggregation by supervisor
        let final_node = WorkflowNode::new(
            "final_review",
            NodeType::Agent {
                agent_id: None,
                capability: Some("Coordination".to_owned()),
                prompt_template: "Aggregate and review all worker results".to_owned(),
            },
        );
        nodes.push(final_node);

        nodes
    }

    /// Generate workflow edges
    fn generate_edges(&self, nodes: &[WorkflowNode]) -> Vec<WorkflowEdge> {
        let mut edges = Vec::new();

        let supervisor_id = &nodes[0].id;
        let parallel_id = &nodes[1].id;
        let merge_id = &nodes[nodes.len() - 2].id;
        let final_id = &nodes[nodes.len() - 1].id;

        // Supervisor -> Parallel
        edges.push(WorkflowEdge::new(
            supervisor_id.clone(),
            parallel_id.clone(),
        ));

        // Parallel -> Workers -> Merge
        for node in nodes.iter().skip(2).take(self.worker_count) {
            let worker_id = &node.id;
            edges.push(WorkflowEdge::new(parallel_id.clone(), worker_id.clone()));
            edges.push(WorkflowEdge::new(worker_id.clone(), merge_id.clone()));
        }

        // Merge -> Final review
        edges.push(WorkflowEdge::new(merge_id.clone(), final_id.clone()));

        edges
    }
}

#[async_trait]
impl Workflow for SupervisorPattern {
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
        // Note: In a real implementation, we'd cache these
        // For now, regenerate on each call
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

        // The worker fan-out uses a `NodeType::Parallel` node, which forces control flow to
        // its `Merge` node once branches complete; `execute_dag`'s topological scheduler does
        // not support forced control flow (see `fanout_collect.rs` for the full rationale), so
        // this must run via the state-machine executor.
        let Some(entry_point) = nodes.first().map(|n| n.id.clone()) else {
            return Ok(serde_json::json!(null));
        };
        let results = engine
            .execute_state_machine(&nodes, &edges, &entry_point, context)
            .await?;

        // Return final supervisor review
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supervisor_pattern_creation() {
        let pattern = SupervisorPattern::new("test_supervisor", "Coordinate workers", 3);

        assert_eq!(pattern.worker_count, 3);
        assert_eq!(pattern.supervisor_prompt, "Coordinate workers");

        let nodes = pattern.generate_nodes();
        // 1 supervisor + 1 parallel + 3 workers + 1 merge + 1 final = 7 nodes
        assert_eq!(nodes.len(), 7);
    }
}
