use crate::agent::Agent;
use crate::agent::kernel::turn::{Handoff, TurnResult};
use crate::config::{DEFAULT_MAX_FLOW_HOPS, OrchestrationConfig};
use crate::orchestration::acl::TaskInvocationPolicy;
use crate::orchestration::coordinator::DelegationOptions;
use crate::orchestration::privilege::{effective_tools, rejected_tools, tool_ceiling};
use crate::orchestration::quarantine::quarantine_once;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Represents a node in the agent flow graph
struct FlowNode {
    name: String,
    agent: Arc<RwLock<Agent>>,
    /// Which other nodes this node may hand off to. Empty means every node.
    task_policy: TaskInvocationPolicy,
    /// Maximum tool set this node may ever grant to a handoff target.
    max_tools: Option<Vec<String>>,
}

/// Builder for constructing agent flows
pub struct FlowBuilder {
    nodes: HashMap<String, FlowNode>,
    entry_point: Option<String>,
    edges: Vec<(String, String)>, // (from, to) pairs for explicit edges
    max_hops: usize,
    quarantine_handoff_output: bool,
}

impl FlowBuilder {
    /// Create a new flow builder
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            edges: Vec::new(),
            max_hops: DEFAULT_MAX_FLOW_HOPS,
            quarantine_handoff_output: true,
        }
    }

    /// Add an agent node to the flow
    pub fn add_node(self, name: impl Into<String>, agent: Agent) -> Self {
        self.add_node_with_policy(name, agent, TaskInvocationPolicy::allow_all())
    }

    /// Add an agent node with a handoff ACL.
    ///
    /// The policy is evaluated against the requested `target_agent` name, so a node can be
    /// limited to a named subset of the flow, for example
    /// `{"*": "deny", "reviewer": "allow"}`.
    pub fn add_node_with_policy(
        mut self,
        name: impl Into<String>,
        agent: Agent,
        task_policy: TaskInvocationPolicy,
    ) -> Self {
        let name = name.into();
        if self.entry_point.is_none() {
            self.entry_point = Some(name.clone());
        }
        self.nodes.insert(
            name.clone(),
            FlowNode {
                name,
                agent: Arc::new(RwLock::new(agent)),
                task_policy,
                max_tools: None,
            },
        );
        self
    }

    /// Set the maximum tool set a node may grant to a handoff target.
    ///
    /// Applies to the named node; a handoff from that node can narrow the set further but
    /// never widen it.
    pub fn set_node_max_tools(mut self, name: &str, tools: Vec<String>) -> Self {
        if let Some(node) = self.nodes.get_mut(name) {
            node.max_tools = Some(tools);
        }
        self
    }

    /// Add an explicit edge between two nodes (optional, for strict graphs)
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push((from.into(), to.into()));
        self
    }

    /// Set the entry point for the flow
    pub fn set_entry_point(mut self, name: impl Into<String>) -> Self {
        self.entry_point = Some(name.into());
        self
    }

    /// Cap how many handoff hops one run may take. Zero falls back to the default.
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = if max_hops == 0 {
            DEFAULT_MAX_FLOW_HOPS
        } else {
            max_hops
        };
        self
    }

    /// Choose whether a handoff wraps the previous agent's output in an untrusted block.
    pub fn with_quarantine_handoff_output(mut self, quarantine: bool) -> Self {
        self.quarantine_handoff_output = quarantine;
        self
    }

    /// Apply hop limit and quarantine settings from the orchestration config.
    pub fn with_orchestration_config(mut self, config: &OrchestrationConfig) -> Self {
        self.max_hops = config.max_flow_hops_or_default();
        self.quarantine_handoff_output = config.quarantine_subagent_output_or_default();
        self
    }

    /// Build the AgentFlow
    pub fn build(self) -> Result<AgentFlow> {
        let entry_point = self
            .entry_point
            .context("No entry point specified. Add at least one node or call set_entry_point()")?;

        if !self.nodes.contains_key(&entry_point) {
            anyhow::bail!("Entry point '{}' does not exist in the flow", entry_point);
        }

        Ok(AgentFlow {
            nodes: self.nodes,
            entry_point,
            edges: self.edges,
            max_hops: self.max_hops,
            quarantine_handoff_output: self.quarantine_handoff_output,
        })
    }
}

impl Default for FlowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

enum SavedTools {
    Unchanged,
    Restore(Option<Vec<String>>),
}

struct SavedDelegation {
    persona: Option<String>,
    tools: SavedTools,
}

fn apply_delegation(
    agent: &mut Agent,
    input: &mut String,
    options: Option<&DelegationOptions>,
) -> SavedDelegation {
    let Some(options) = options else {
        return SavedDelegation {
            persona: None,
            tools: SavedTools::Unchanged,
        };
    };
    let saved = SavedDelegation {
        persona: options
            .persona
            .as_ref()
            .map(|_| agent.system_prompt().to_owned()),
        tools: if options.tools.is_some() {
            SavedTools::Restore(agent.allowed_tools())
        } else {
            SavedTools::Unchanged
        },
    };
    if let Some(persona) = options.persona.as_deref() {
        agent.set_system_prompt(persona);
    }
    if let Some(tools) = options.tools.as_ref() {
        agent.set_allowed_tools(Some(tools.clone()));
    }
    if let Some(instructions) = options.instructions.as_deref() {
        *input = format!("Instructions:\n{}\n\nTask:\n{}", instructions.trim(), input);
    }
    saved
}

fn restore_delegation(agent: &mut Agent, saved: SavedDelegation) {
    if let Some(persona) = saved.persona {
        agent.set_system_prompt(persona);
    }
    match saved.tools {
        SavedTools::Unchanged => {}
        SavedTools::Restore(tools) => agent.set_allowed_tools(tools),
    }
}

fn log_turn_input(agent_name: &str, current_node_name: &str, input: &str) {
    let preview = if input.len() > 100 {
        format!("{}...", caliche_contracts::text::truncate_utf8(input, 100))
    } else {
        input.to_owned()
    };
    tracing::info!(
        agent = %agent_name,
        "Executing agent '{}' with input: {}",
        current_node_name,
        preview
    );
}

enum FlowAdvance {
    Finish(Option<String>),
    Handoff {
        target: String,
        options: DelegationOptions,
    },
}

/// Manages a flow of agents with handoff support
pub struct AgentFlow {
    nodes: HashMap<String, FlowNode>,
    entry_point: String,
    edges: Vec<(String, String)>,
    max_hops: usize,
    quarantine_handoff_output: bool,
}

impl AgentFlow {
    /// Create a new flow builder
    pub fn builder() -> FlowBuilder {
        FlowBuilder::new()
    }

    /// Authorize a handoff and narrow its delegation options to the allowed privileges.
    ///
    /// Three checks apply, in order: the source node's ACL must permit the target name, the
    /// target must exist in the flow, and the requested tools are intersected with the source
    /// and target ceilings so a handoff can never escalate privilege.
    fn authorize_handoff(&self, from: &str, handoff: &Handoff) -> Result<DelegationOptions> {
        let source = self
            .nodes
            .get(from)
            .with_context(|| format!("Handoff source '{from}' not found in the flow"))?;

        source.task_policy.authorize(from, &handoff.target_agent)?;

        let target = self.nodes.get(&handoff.target_agent).with_context(|| {
            format!(
                "Handoff target '{}' does not exist in the flow",
                handoff.target_agent
            )
        })?;

        let mut options = handoff.to_delegation_options();
        let ceiling = tool_ceiling(source.max_tools.as_deref(), target.max_tools.as_deref());
        let refused = options
            .tools
            .as_deref()
            .map(|requested| rejected_tools(ceiling.as_deref(), requested))
            .unwrap_or_default();
        options.tools = effective_tools(ceiling.as_deref(), options.tools.as_deref());

        if !refused.is_empty() {
            tracing::warn!(
                from = %from,
                to = %handoff.target_agent,
                refused = ?refused,
                "Handoff requested tools outside the privilege ceiling; they were dropped"
            );
        }
        Ok(options)
    }

    fn record_turn(
        &self,
        current_node_name: &str,
        input: &str,
        turn_result: TurnResult,
        execution_log: &mut Vec<ExecutionStep>,
    ) -> Result<FlowAdvance> {
        let (response, handoff, usage) = match turn_result {
            TurnResult::Reply(response, usage) => (Some(response), None, usage),
            TurnResult::Handoff(handoff, usage) => (None, Some(handoff), usage),
            TurnResult::Stop(usage) => (None, None, usage),
        };
        execution_log.push(ExecutionStep {
            agent_name: current_node_name.to_owned(),
            input: input.to_owned(),
            response: response.clone(),
            handoff: handoff.clone(),
            usage,
        });
        let Some(handoff) = handoff else {
            return Ok(FlowAdvance::Finish(response));
        };
        let options = self.authorize_handoff(current_node_name, &handoff)?;
        tracing::info!(
            "Handing off from '{}' to '{}': {}",
            current_node_name,
            handoff.target_agent,
            handoff.reason
        );
        Ok(FlowAdvance::Handoff {
            target: handoff.target_agent,
            options,
        })
    }

    fn input_for_turn(&self, initial_input: &str, execution_log: &[ExecutionStep]) -> String {
        let Some(last_log) = execution_log.last() else {
            return initial_input.to_owned();
        };
        match last_log.response.as_deref() {
            Some(response) if self.quarantine_handoff_output => {
                quarantine_once(&last_log.agent_name, response)
            }
            Some(response) => response.to_owned(),
            None => initial_input.to_owned(),
        }
    }

    /// Run the flow with an initial input message
    pub async fn run(&self, initial_input: &str) -> Result<FlowExecutionResult> {
        let mut current_node_name = self.entry_point.clone();
        let mut execution_log: Vec<ExecutionStep> = Vec::new();
        let mut max_iterations = self.max_hops;
        let mut final_response: Option<String> = None;
        let mut pending_options: Option<DelegationOptions> = None;

        while max_iterations > 0 {
            max_iterations -= 1;

            let node = self
                .nodes
                .get(&current_node_name)
                .context(format!("Node '{}' not found", current_node_name))?;

            let mut input = self.input_for_turn(initial_input, &execution_log);
            let mut agent = node.agent.write().await;
            let saved = apply_delegation(&mut agent, &mut input, pending_options.as_ref());
            // Clear so we do not re-apply on the same iteration; Handoff branch sets it for next iteration
            drop(pending_options.take());
            log_turn_input(&node.name, &current_node_name, &input);
            let turn_result = agent.execute_turn(&input).await?;
            restore_delegation(&mut agent, saved);

            match self.record_turn(&current_node_name, &input, turn_result, &mut execution_log)? {
                FlowAdvance::Finish(response) => {
                    if let Some(response) = response {
                        final_response = Some(response);
                    }
                    break;
                }
                FlowAdvance::Handoff { target, options } => {
                    pending_options = Some(options);
                    current_node_name = target;
                }
            }
        }

        if max_iterations == 0 {
            tracing::warn!(
                max_hops = self.max_hops,
                "Reached the maximum flow hop limit; stopping"
            );
        }

        Ok(FlowExecutionResult {
            final_response: final_response.unwrap_or_else(|| "Flow completed".to_string()),
            execution_log,
        })
    }

    /// Generate a Mermaid diagram representation of the flow
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::from("graph TD\n");
        mermaid.push_str("    style Entry fill:#90EE90\n");
        mermaid.push_str("    style Exit fill:#FFB6C1\n\n");

        // Add entry point
        mermaid.push_str(&format!("    Entry([Start]) --> {}\n", self.entry_point));

        // Add all nodes
        for name in self.nodes.keys() {
            mermaid.push_str(&format!("    {}[{}]\n", name, name));
        }

        // Add explicit edges
        for (from, to) in &self.edges {
            mermaid.push_str(&format!("    {} --> {}\n", from, to));
        }

        // Add handoff edges (all nodes can potentially handoff to any other)
        // We'll mark these as dashed to indicate they're dynamic
        for from in self.nodes.keys() {
            for to in self.nodes.keys() {
                if from != to {
                    mermaid.push_str(&format!("    {} -.->|handoff| {}\n", from, to));
                }
            }
        }

        mermaid.push_str("\n    Exit([End])\n");

        mermaid
    }

    /// Get the entry point node name
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Get all node names
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    /// Maximum handoff hops allowed in one run.
    pub fn max_hops(&self) -> usize {
        self.max_hops
    }
}

/// Result of executing a flow
#[derive(Debug, Clone)]
pub struct FlowExecutionResult {
    pub final_response: String,
    pub execution_log: Vec<ExecutionStep>,
}

/// A single step in flow execution
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    pub agent_name: String,
    pub input: String,
    pub response: Option<String>,
    pub handoff: Option<Handoff>,
    pub usage: crate::messages::TokenUsage,
}
