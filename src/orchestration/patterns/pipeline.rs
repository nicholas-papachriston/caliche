use crate::orchestration::workflow::{
    ExecutionContext, NodeType, Workflow, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Pipeline pattern: sequential data transformation through multiple stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePattern {
    id: String,
    name: String,
    description: String,
    stages: Vec<PipelineStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub name: String,
    pub capability: Option<String>,
    pub prompt_template: String,
    pub transform: Option<String>,
}

impl PipelinePattern {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            description: "Sequential pipeline workflow".to_owned(),
            stages: Vec::new(),
        }
    }

    pub fn add_stage(
        mut self,
        name: impl Into<String>,
        prompt_template: impl Into<String>,
    ) -> Self {
        self.stages.push(PipelineStage {
            name: name.into(),
            capability: None,
            prompt_template: prompt_template.into(),
            transform: None,
        });
        self
    }

    pub fn add_stage_with_capability(
        mut self,
        name: impl Into<String>,
        capability: impl Into<String>,
        prompt_template: impl Into<String>,
    ) -> Self {
        self.stages.push(PipelineStage {
            name: name.into(),
            capability: Some(capability.into()),
            prompt_template: prompt_template.into(),
            transform: None,
        });
        self
    }

    pub fn add_transform_stage(
        mut self,
        name: impl Into<String>,
        transform: impl Into<String>,
    ) -> Self {
        self.stages.push(PipelineStage {
            name: name.into(),
            capability: None,
            prompt_template: String::new(),
            transform: Some(transform.into()),
        });
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    fn generate_nodes(&self) -> Vec<WorkflowNode> {
        self.stages
            .iter()
            .map(|stage| {
                if let Some(transform) = &stage.transform {
                    WorkflowNode::new(
                        stage.name.clone(),
                        NodeType::Transform {
                            function: transform.clone(),
                        },
                    )
                } else {
                    WorkflowNode::new(
                        stage.name.clone(),
                        NodeType::Agent {
                            agent_id: None,
                            capability: stage.capability.clone(),
                            prompt_template: stage.prompt_template.clone(),
                        },
                    )
                }
            })
            .collect()
    }

    fn generate_edges(&self, nodes: &[WorkflowNode]) -> Vec<WorkflowEdge> {
        nodes
            .windows(2)
            .map(|window| {
                WorkflowEdge::new(window[0].id.clone(), window[1].id.clone())
                    .with_label("sequential")
            })
            .collect()
    }
}

#[async_trait]
impl Workflow for PipelinePattern {
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

        let results = engine.execute_dag(&nodes, &edges, context).await?;

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
    fn test_pipeline_pattern() {
        let pattern = PipelinePattern::new("data_pipeline")
            .add_stage("extract", "Extract data from source")
            .add_stage("transform", "Transform data format")
            .add_stage("load", "Load into destination");

        assert_eq!(pattern.stages.len(), 3);

        let nodes = pattern.generate_nodes();
        assert_eq!(nodes.len(), 3);

        let edges = pattern.generate_edges(&nodes);
        assert_eq!(edges.len(), 2); // n-1 edges for n nodes
    }
}
