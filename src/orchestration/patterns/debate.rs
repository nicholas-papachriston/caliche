use crate::orchestration::workflow::{
    ExecutionContext, NodeType, Workflow, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Debate pattern: multiple agents discuss and reach consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebatePattern {
    id: String,
    name: String,
    description: String,
    topic: String,
    debater_count: usize,
    max_rounds: usize,
    moderator_prompt: String,
}

impl DebatePattern {
    pub fn new(
        name: impl Into<String>,
        topic: impl Into<String>,
        debater_count: usize,
        max_rounds: usize,
    ) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            description: "Multi-agent debate pattern".to_owned(),
            topic: topic.into(),
            debater_count,
            max_rounds,
            moderator_prompt: "Moderate the debate and synthesize final conclusion".to_owned(),
        }
    }

    pub fn with_moderator_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.moderator_prompt = prompt.into();
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    fn generate_nodes(&self) -> Vec<WorkflowNode> {
        let mut nodes = Vec::new();

        // Loop node for debate rounds
        let loop_node = WorkflowNode::new(
            "debate_rounds",
            NodeType::Loop {
                max_iterations: self.max_rounds,
                condition: "consensus_not_reached".to_owned(),
            },
        );
        nodes.push(loop_node);

        // Parallel node for debaters
        let parallel_node = WorkflowNode::new("parallel_debaters", NodeType::Parallel);
        nodes.push(parallel_node);

        // Debater nodes
        for i in 0..self.debater_count {
            let debater_node = WorkflowNode::new(
                format!("debater_{}", i),
                NodeType::Agent {
                    agent_id: None,
                    capability: None,
                    prompt_template: format!("Argue position {} on topic: {}", i, self.topic),
                },
            );
            nodes.push(debater_node);
        }

        // Merge debater arguments
        let merge_node = WorkflowNode::new(
            "merge_arguments",
            NodeType::Merge {
                strategy: crate::orchestration::workflow::MergeStrategy::Concatenate,
            },
        );
        nodes.push(merge_node);

        // Moderator evaluates consensus
        let eval_node = WorkflowNode::new(
            "evaluate_consensus",
            NodeType::Agent {
                agent_id: None,
                capability: Some("Coordination".to_owned()),
                prompt_template: "Evaluate if consensus is reached".to_owned(),
            },
        );
        nodes.push(eval_node);

        // Final moderator synthesis
        let final_node = WorkflowNode::new(
            "final_synthesis",
            NodeType::Agent {
                agent_id: None,
                capability: Some("Coordination".to_owned()),
                prompt_template: self.moderator_prompt.clone(),
            },
        );
        nodes.push(final_node);

        nodes
    }

    fn generate_edges(&self, nodes: &[WorkflowNode]) -> Vec<WorkflowEdge> {
        let mut edges = Vec::new();

        let loop_id = &nodes[0].id;
        let parallel_id = &nodes[1].id;
        let merge_id = &nodes[nodes.len() - 3].id;
        let eval_id = &nodes[nodes.len() - 2].id;
        let final_id = &nodes[nodes.len() - 1].id;

        // Loop -> Parallel
        edges.push(WorkflowEdge::new(loop_id.clone(), parallel_id.clone()));

        // Parallel -> Debaters -> Merge
        for node in nodes.iter().skip(2).take(self.debater_count) {
            let debater_id = &node.id;
            edges.push(WorkflowEdge::new(parallel_id.clone(), debater_id.clone()));
            edges.push(WorkflowEdge::new(debater_id.clone(), merge_id.clone()));
        }

        // Merge -> Evaluate
        edges.push(WorkflowEdge::new(merge_id.clone(), eval_id.clone()));

        // Evaluate -> Final (the loop node handles iteration internally)
        // Note: We don't create a back-edge to avoid cycles in the DAG
        // The Loop node's max_iterations will control the debate rounds
        edges.push(WorkflowEdge::new(eval_id.clone(), final_id.clone()));

        edges
    }
}

#[async_trait]
impl Workflow for DebatePattern {
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

        // The graph starts with a `NodeType::Loop` node and fans debaters out through a
        // `NodeType::Parallel` node; both force control flow (loop continuation/exit, and
        // parallel-to-merge), which `execute_dag`'s topological scheduler does not support
        // (see `fanout_collect.rs` for the full rationale), so this must run via the
        // state-machine executor.
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
