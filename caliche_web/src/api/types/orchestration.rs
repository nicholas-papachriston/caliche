//! Chain, multi-agent, workflow, and reasoning execution types.

use serde::{Deserialize, Serialize};

/// A single reasoning step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_type: String,
    pub content: String,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_input: Option<String>,
    #[serde(default)]
    pub tool_output: Option<String>,
}

/// Request for reasoning patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPatternRequest {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Response from reasoning pattern execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPatternResponse {
    pub final_answer: String,
    pub steps: Vec<ReasoningStep>,
    pub success: bool,
    #[serde(default)]
    pub error: Option<String>,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMChainRequest {
    pub prompt_template: String,
    pub variables: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapReduceRequest {
    pub map_template: String,
    pub reduce_template: String,
    pub items: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalChainRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMapReduceRequest {
    pub document: String,
    pub map_template: String,
    pub reduce_template: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainResponse {
    pub output: serde_json::Value,
    pub intermediate_steps: Vec<(String, serde_json::Value)>,
    pub request_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MultiAgentPattern {
    Pipeline,
    Debate,
    Hierarchical,
    Supervisor,
    #[serde(rename = "fan_out_collect")]
    FanOutCollect,
}

impl MultiAgentPattern {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pipeline => "Pipeline",
            Self::Debate => "Debate",
            Self::Hierarchical => "Hierarchical",
            Self::Supervisor => "Supervisor",
            Self::FanOutCollect => "Fan-Out/Collect",
        }
    }
    pub fn description(&self) -> &'static str {
        match self {
            Self::Pipeline => {
                "Sequential handoff: output of each agent becomes input for the next."
            }
            Self::Debate => {
                "Agents generate competing perspectives, then a synthesizer produces the final answer."
            }
            Self::Hierarchical => {
                "A manager agent delegates subtasks to specialized worker agents."
            }
            Self::Supervisor => "A coordinator routes tasks to the most capable agent.",
            Self::FanOutCollect => {
                "All agents run in parallel on the same task; results are aggregated."
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentRequest {
    pub pattern: MultiAgentPattern,
    pub agents: Vec<AgentConfig>,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub agent_id: String,
    pub agent_name: String,
    pub output: String,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentResponse {
    pub pattern: String,
    pub final_result: String,
    pub agent_results: Vec<AgentResult>,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    pub id: String,
    pub node_type: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEdge {
    pub from: String,
    pub to: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRequest {
    pub nodes: Vec<WorkflowNode>,
    pub edges: Vec<WorkflowEdge>,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResponse {
    pub workflow_id: String,
    #[serde(default)]
    pub results: Vec<serde_json::Value>,
    pub state: String,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkflowStatusResponse {
    pub workflow_id: String,
    pub state: String,
    #[serde(default)]
    pub current_node: Option<String>,
    #[serde(default)]
    pub completed_nodes: Vec<String>,
    #[serde(default)]
    pub results: Vec<serde_json::Value>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorRequest {
    pub task: String,
    #[serde(default)]
    pub required_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorResponse {
    pub selected_agent: String,
    pub result: String,
    #[serde(default)]
    pub capabilities_matched: Vec<String>,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingInterrupt {
    pub id: String,
    pub workflow_id: String,
    pub node_id: String,
    pub prompt: String,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterruptsListResponse {
    #[serde(default)]
    pub interrupts: Vec<PendingInterrupt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeInterruptRequest {
    pub feedback: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeInterruptResponse {
    pub success: bool,
    pub message: String,
}
