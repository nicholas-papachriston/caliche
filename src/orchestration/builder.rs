use super::workflow::{
    ExecutionContext, MergeStrategy, NodeType, Workflow, WorkflowEdge, WorkflowEngine, WorkflowNode,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Fluent builder for constructing workflows
pub struct WorkflowBuilder {
    id: String,
    name: String,
    description: String,
    nodes: Vec<WorkflowNode>,
    edges: Vec<WorkflowEdge>,
    current_node_id: Option<String>,
    entry_point: Option<String>,
    pending_fan_out: Option<PendingFanOut>,
}

#[derive(Debug, Clone)]
struct PendingFanOut {
    branch_node_ids: Vec<String>,
}

/// Parse a merge strategy label from JS/Python bindings.
///
/// Accepts: `concatenate`, `vote`, `first_complete` (or `first-complete`), `all_complete` (or
/// `all-complete`), and `custom:<name>` (prefix is case-insensitive).
pub fn merge_strategy_from_str(s: &str) -> Result<MergeStrategy> {
    let t = s.trim();
    let lower = t.to_ascii_lowercase();
    if let Some(pos) = lower.find("custom:") {
        let rest = t[pos + "custom:".len()..].trim();
        if rest.is_empty() {
            anyhow::bail!("custom merge strategy requires a name after 'custom:'");
        }
        return Ok(MergeStrategy::Custom(rest.to_string()));
    }
    match lower.as_str() {
        "concatenate" => Ok(MergeStrategy::Concatenate),
        "vote" => Ok(MergeStrategy::Vote),
        "first_complete" | "first-complete" | "firstcomplete" => Ok(MergeStrategy::FirstComplete),
        "all_complete" | "all-complete" | "allcomplete" => Ok(MergeStrategy::AllComplete),
        _ => anyhow::bail!(
            "unknown merge strategy '{t}': expected concatenate, vote, first_complete, all_complete, or custom:<name>"
        ),
    }
}

impl WorkflowBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7().to_string(),
            name: name.into(),
            description: String::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            current_node_id: None,
            entry_point: None,
            pending_fan_out: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add an agent node
    pub fn add_agent(self, name: impl Into<String>, prompt_template: impl Into<String>) -> Self {
        self.add_agent_with_connection(name, prompt_template, true)
    }

    /// Add an agent node with optional automatic connection
    pub fn add_agent_with_connection(
        mut self,
        name: impl Into<String>,
        prompt_template: impl Into<String>,
        auto_connect: bool,
    ) -> Self {
        if auto_connect {
            self = self.collect(MergeStrategy::AllComplete);
        }

        let node = WorkflowNode::new(
            name.into(),
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: prompt_template.into(),
            },
        );

        // Connect to previous node if exists and auto_connect is true
        if auto_connect {
            // Clone node.id once and reuse
            let node_id = node.id.clone();
            if let Some(prev_id) = &self.current_node_id {
                self.edges
                    .push(WorkflowEdge::new(prev_id.clone(), node_id.clone()));
            }
            self.current_node_id = Some(node_id);
        } else {
            // Don't update current_node_id if not auto-connecting
            // This allows manual connection later
        }

        self.nodes.push(node);
        self
    }

    /// Add an agent node with specific capability requirement
    pub fn add_agent_with_capability(
        mut self,
        name: impl Into<String>,
        capability: impl Into<String>,
        prompt_template: impl Into<String>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let node = WorkflowNode::new(
            name.into(),
            NodeType::Agent {
                agent_id: None,
                capability: Some(capability.into()),
                prompt_template: prompt_template.into(),
            },
        );

        // Clone node.id once and reuse
        let node_id = node.id.clone();
        if let Some(prev_id) = &self.current_node_id {
            self.edges
                .push(WorkflowEdge::new(prev_id.clone(), node_id.clone()));
        }

        self.current_node_id = Some(node_id);
        self.nodes.push(node);
        self
    }

    /// Add an agent node assigned to specific agent ID
    pub fn add_agent_by_id(
        mut self,
        name: impl Into<String>,
        agent_id: impl Into<String>,
        prompt_template: impl Into<String>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let node = WorkflowNode::new(
            name.into(),
            NodeType::Agent {
                agent_id: Some(agent_id.into()),
                capability: None,
                prompt_template: prompt_template.into(),
            },
        );

        if let Some(prev_id) = &self.current_node_id {
            self.edges
                .push(WorkflowEdge::new(prev_id.clone(), node.id.clone()));
        }

        self.current_node_id = Some(node.id.clone());
        self.nodes.push(node);
        self
    }

    /// Continue sequentially with another node
    pub fn then(self, name: impl Into<String>, prompt_template: impl Into<String>) -> Self {
        self.add_agent(name, prompt_template)
    }

    /// Add parallel execution node
    pub fn parallel(self, branches: Vec<(String, String)>) -> Self {
        self.fan_out(branches).collect(MergeStrategy::AllComplete)
    }

    /// Like [`Self::parallel`], but uses the given merge strategy for the implicit collect node.
    pub fn parallel_with_strategy(
        self,
        branches: Vec<(String, String)>,
        strategy: MergeStrategy,
    ) -> Self {
        self.fan_out(branches).collect(strategy)
    }

    /// `true` when a [`Self::fan_out`] is waiting for [`Self::collect`] / [`Self::collect_named`].
    pub fn has_pending_fan_out(&self) -> bool {
        self.pending_fan_out.is_some()
    }

    /// Add a fan-out node with branch tasks. Call `collect(...)` to re-join.
    pub fn fan_out(mut self, branches: Vec<(String, String)>) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let parallel_node = WorkflowNode::new("parallel", NodeType::Parallel);
        if let Some(prev_id) = &self.current_node_id {
            self.edges
                .push(WorkflowEdge::new(prev_id.clone(), parallel_node.id.clone()));
        }
        let parallel_id = parallel_node.id.clone();
        self.nodes.push(parallel_node);

        let mut branch_node_ids = Vec::new();
        for (name, prompt_template) in branches {
            let branch_node = WorkflowNode::new(
                name,
                NodeType::Agent {
                    agent_id: None,
                    capability: None,
                    prompt_template,
                },
            );
            self.edges.push(WorkflowEdge::new(
                parallel_id.clone(),
                branch_node.id.clone(),
            ));
            branch_node_ids.push(branch_node.id.clone());
            self.nodes.push(branch_node);
        }

        self.pending_fan_out = Some(PendingFanOut { branch_node_ids });
        self.current_node_id = None;
        self
    }

    /// Add a merge/collect node for a pending fan-out (merge node display name `"collect"`).
    pub fn collect(self, strategy: MergeStrategy) -> Self {
        self.collect_named("collect", strategy)
    }

    /// Add a merge/collect node for a pending fan-out with a custom merge node name.
    pub fn collect_named(
        mut self,
        merge_node_name: impl Into<String>,
        strategy: MergeStrategy,
    ) -> Self {
        let Some(pending) = self.pending_fan_out.take() else {
            return self;
        };

        let merge_node = WorkflowNode::new(merge_node_name.into(), NodeType::Merge { strategy });
        let merge_id = merge_node.id.clone();
        for branch_id in pending.branch_node_ids {
            self.edges
                .push(WorkflowEdge::new(branch_id, merge_id.clone()));
        }
        self.nodes.push(merge_node);
        self.current_node_id = Some(merge_id);
        self
    }

    /// Add a condition node (without branch nodes - for manual connection)
    pub fn add_condition(mut self, name: impl Into<String>, predicate: impl Into<String>) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let condition_node = WorkflowNode::new(
            name.into(),
            NodeType::Condition {
                predicate: predicate.into(),
            },
        );

        // Don't auto-connect condition nodes - they should be manually connected
        // Don't update current_node_id either

        self.nodes.push(condition_node);
        self
    }

    /// Add conditional branching
    pub fn branch(
        self,
        predicate: impl Into<String>,
        true_branch: impl Into<String>,
        false_branch: impl Into<String>,
    ) -> Self {
        self.branch_with_name("condition", predicate, true_branch, false_branch)
    }

    /// Add conditional branching with a specific condition node name
    pub fn branch_with_name(
        mut self,
        condition_name: impl Into<String>,
        predicate: impl Into<String>,
        true_branch: impl Into<String>,
        false_branch: impl Into<String>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let condition_node = WorkflowNode::new(
            condition_name.into(),
            NodeType::Condition {
                predicate: predicate.into(),
            },
        );

        if let Some(prev_id) = &self.current_node_id {
            self.edges.push(WorkflowEdge::new(
                prev_id.clone(),
                condition_node.id.clone(),
            ));
        }

        let condition_id = condition_node.id.clone();
        self.nodes.push(condition_node);

        // True branch
        let true_node = WorkflowNode::new(
            "true_branch",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: true_branch.into(),
            },
        );
        self.edges.push(
            WorkflowEdge::new(condition_id.clone(), true_node.id.clone())
                .with_label("true")
                .with_condition(crate::orchestration::workflow::transition::COND_BRANCH_TRUE),
        );
        let true_id = true_node.id.clone();
        self.nodes.push(true_node);

        // False branch
        let false_node = WorkflowNode::new(
            "false_branch",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: false_branch.into(),
            },
        );
        self.edges.push(
            WorkflowEdge::new(condition_id, false_node.id.clone())
                .with_label("false")
                .with_condition(crate::orchestration::workflow::transition::COND_BRANCH_FALSE),
        );
        let _false_id = false_node.id.clone();
        self.nodes.push(false_node);

        // Set current to both branches (next node will connect to both)
        // For simplicity, just use true branch
        self.current_node_id = Some(true_id);
        self
    }

    /// Add an if/then/else block and automatically re-join both branches.
    pub fn if_then_else(
        self,
        predicate: impl Into<String>,
        true_branch: impl Into<String>,
        false_branch: impl Into<String>,
    ) -> Self {
        self.if_then_else_with_name("if_condition", predicate, true_branch, false_branch)
    }

    /// Add a named if/then/else block and automatically re-join both branches.
    pub fn if_then_else_with_name(
        mut self,
        condition_name: impl Into<String>,
        predicate: impl Into<String>,
        true_branch: impl Into<String>,
        false_branch: impl Into<String>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let condition_node = WorkflowNode::new(
            condition_name.into(),
            NodeType::Condition {
                predicate: predicate.into(),
            },
        );

        if let Some(prev_id) = &self.current_node_id {
            self.edges.push(WorkflowEdge::new(
                prev_id.clone(),
                condition_node.id.clone(),
            ));
        }
        let condition_id = condition_node.id.clone();
        self.nodes.push(condition_node);

        let true_node = WorkflowNode::new(
            "if_true",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: true_branch.into(),
            },
        );
        let true_id = true_node.id.clone();
        self.nodes.push(true_node);
        self.edges.push(
            WorkflowEdge::new(condition_id.clone(), true_id.clone())
                .with_label("true")
                .with_condition(crate::orchestration::workflow::transition::COND_BRANCH_TRUE),
        );

        let false_node = WorkflowNode::new(
            "if_false",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: false_branch.into(),
            },
        );
        let false_id = false_node.id.clone();
        self.nodes.push(false_node);
        self.edges.push(
            WorkflowEdge::new(condition_id, false_id.clone())
                .with_label("false")
                .with_condition(crate::orchestration::workflow::transition::COND_BRANCH_FALSE),
        );

        let join_node = WorkflowNode::new(
            "if_join",
            NodeType::Transform {
                function: "identity".to_owned(),
            },
        );
        let join_id = join_node.id.clone();
        self.nodes.push(join_node);

        self.edges.push(WorkflowEdge::new(true_id, join_id.clone()));
        self.edges
            .push(WorkflowEdge::new(false_id, join_id.clone()));
        self.current_node_id = Some(join_id);
        self
    }

    /// Add loop node
    pub fn loop_while(
        self,
        condition: impl Into<String>,
        max_iterations: usize,
        body_prompt: impl Into<String>,
    ) -> Self {
        self.loop_while_with_name("loop", condition, max_iterations, body_prompt)
    }

    /// Add loop node with a specific name
    pub fn loop_while_with_name(
        mut self,
        loop_name: impl Into<String>,
        condition: impl Into<String>,
        max_iterations: usize,
        body_prompt: impl Into<String>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let loop_name_str = loop_name.into();
        let loop_node = WorkflowNode::new(
            loop_name_str.clone(),
            NodeType::Loop {
                max_iterations,
                condition: condition.into(),
            },
        );

        if let Some(prev_id) = &self.current_node_id {
            self.edges
                .push(WorkflowEdge::new(prev_id.clone(), loop_node.id.clone()));
        }
        let loop_id = loop_node.id.clone();
        self.nodes.push(loop_node);

        // Loop body - unique name based on loop name to avoid conflicts
        let body_name = format!("{}_body", loop_name_str);
        let body_node = WorkflowNode::new(
            body_name,
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: body_prompt.into(),
            },
        );

        let body_id = body_node.id.clone();

        self.nodes.push(body_node);
        let exit_name = format!("{}_exit", loop_name_str);
        let exit_node = WorkflowNode::new(
            exit_name,
            NodeType::Transform {
                function: "identity".to_owned(),
            },
        );
        let exit_id = exit_node.id.clone();
        self.nodes.push(exit_node);

        self.edges
            .push(WorkflowEdge::new(loop_id.clone(), body_id.clone()).with_label("body"));
        self.edges
            .push(WorkflowEdge::new(loop_id.clone(), exit_id.clone()).with_label("exit"));

        // Loop back
        self.edges
            .push(WorkflowEdge::new(body_id, loop_id).with_label("repeat"));

        self.current_node_id = Some(exit_id);
        if self.entry_point.is_none()
            && let Some(first_node) = self.nodes.first()
        {
            self.entry_point = Some(first_node.id.clone());
        }
        self
    }

    /// Add custom transformation node
    pub fn transform(mut self, name: impl Into<String>, function: impl Into<String>) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let node = WorkflowNode::new(
            name.into(),
            NodeType::Transform {
                function: function.into(),
            },
        );

        if let Some(prev_id) = &self.current_node_id {
            self.edges
                .push(WorkflowEdge::new(prev_id.clone(), node.id.clone()));
        }

        self.current_node_id = Some(node.id.clone());
        self.nodes.push(node);
        self
    }

    /// Add a human-in-the-loop interrupt node (pauses workflow for approval)
    pub fn human_interrupt(
        mut self,
        name: impl Into<String>,
        message: impl Into<String>,
        required_fields: Option<Vec<String>>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let node = WorkflowNode::new(
            name.into(),
            NodeType::HumanInterrupt {
                message: message.into(),
                required_fields,
            },
        );

        // Don't auto-connect human interrupt nodes - they should be manually connected
        // Don't update current_node_id either

        self.nodes.push(node);
        self
    }

    /// Set the entry point for state machine execution (enables cyclic graphs)
    /// Accepts either a node ID or node name (will find first matching node by name)
    pub fn with_entry_point(mut self, node_identifier: impl Into<String>) -> Self {
        self = self.collect(MergeStrategy::AllComplete);

        let identifier = node_identifier.into();

        // Check if it's a node ID (UUID format) or name
        // Try to find by name first, then fall back to treating as ID
        if let Some(node) = self.nodes.iter().find(|n| n.name == identifier) {
            self.entry_point = Some(node.id.clone());
        } else {
            // Treat as node ID
            self.entry_point = Some(identifier);
        }
        self
    }

    /// Add an edge with a condition for conditional transitions
    /// Accepts node names or IDs (will find first matching node by name)
    pub fn add_edge_with_condition(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: impl Into<String>,
    ) -> Self {
        self = self.collect(MergeStrategy::AllComplete);
        let from_str = from.into();
        let to_str = to.into();
        let condition_str = condition.into();
        if let Ok((from_id, to_id)) = self.resolve_edge_endpoints(&from_str, &to_str) {
            self.edges
                .push(WorkflowEdge::new(from_id, to_id).with_condition(condition_str));
        }
        self
    }

    /// Add an edge with a condition and return detailed errors if validation fails.
    pub fn try_add_edge_with_condition(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: impl Into<String>,
    ) -> Result<Self> {
        self = self.collect(MergeStrategy::AllComplete);
        let from_str = from.into();
        let to_str = to.into();
        let condition_str = condition.into();
        let (from_id, to_id) = self.resolve_edge_endpoints(&from_str, &to_str)?;
        self.edges
            .push(WorkflowEdge::new(from_id, to_id).with_condition(condition_str));
        Ok(self)
    }

    fn resolve_edge_endpoints(&self, from: &str, to: &str) -> Result<(String, String)> {
        let from_id = self.find_node_id(from);
        let to_id = self.find_node_id(to);

        let from_exists = self.nodes.iter().any(|n| n.id == from_id);
        let to_exists = self.nodes.iter().any(|n| n.id == to_id);

        if !from_exists {
            return Err(anyhow::anyhow!(
                "Cannot create edge: 'from' node '{}' (resolved to ID '{}') does not exist",
                from,
                from_id
            ));
        }
        if !to_exists {
            return Err(anyhow::anyhow!(
                "Cannot create edge: 'to' node '{}' (resolved to ID '{}') does not exist",
                to,
                to_id
            ));
        }
        Ok((from_id, to_id))
    }

    /// Helper to find node ID by name or return the identifier as-is (assuming it's an ID)
    fn find_node_id(&self, identifier: &str) -> String {
        if let Some(node) = self.nodes.iter().find(|n| n.name == identifier) {
            node.id.clone()
        } else {
            // Check if it's already a valid node ID
            if self.nodes.iter().any(|n| n.id == identifier) {
                identifier.to_string()
            } else {
                // Return as-is, but this might cause validation errors later
                identifier.to_string()
            }
        }
    }

    /// Build the workflow
    pub fn build(mut self) -> BuiltWorkflow {
        self = self.collect(MergeStrategy::AllComplete);

        BuiltWorkflow {
            id: self.id,
            name: self.name,
            description: self.description,
            nodes: self.nodes,
            edges: self.edges,
            entry_point: self.entry_point,
        }
    }
}

/// A concrete workflow built from the builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltWorkflow {
    id: String,
    name: String,
    description: String,
    nodes: Vec<WorkflowNode>,
    edges: Vec<WorkflowEdge>,
    entry_point: Option<String>,
}

#[async_trait]
impl Workflow for BuiltWorkflow {
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
        &self.nodes
    }

    fn edges(&self) -> &[WorkflowEdge] {
        &self.edges
    }

    fn entry_point(&self) -> Option<&str> {
        // Return the node name instead of ID for better usability
        self.entry_point.as_ref().and_then(|id| {
            self.nodes
                .iter()
                .find(|n| n.id == *id)
                .map(|n| n.name.as_str())
        })
    }

    fn validate(&self) -> Result<()> {
        let entry_point = match self.entry_point.as_deref() {
            Some(id) => crate::orchestration::workflow::validation::EntryPointRef::NodeId(id),
            None => crate::orchestration::workflow::validation::EntryPointRef::None,
        };
        crate::orchestration::workflow::validation::validate_workflow(
            &self.nodes,
            &self.edges,
            entry_point,
        )
    }

    async fn execute(
        &self,
        engine: &WorkflowEngine,
        context: &mut ExecutionContext,
    ) -> Result<serde_json::Value> {
        // Choose execution strategy based on entry_point
        let results = if let Some(entry_name) = self.entry_point() {
            // State machine mode (supports cycles)
            // Resolve entry point name to node ID
            let entry_id = self
                .nodes
                .iter()
                .find(|n| n.name == entry_name)
                .map(|n| n.id.clone())
                .context(format!("Entry point node '{}' not found", entry_name))?;

            engine
                .execute_state_machine(&self.nodes, &self.edges, &entry_id, context)
                .await?
        } else {
            // DAG mode (topological sort)
            engine
                .execute_dag(&self.nodes, &self.edges, context)
                .await?
        };

        // Return last result
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

impl BuiltWorkflow {
    /// Serialize workflow to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Deserialize workflow from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }
}
