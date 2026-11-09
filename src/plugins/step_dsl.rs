//! Step DSL types for plugin execution pipelines.
//!
//! Steps are composable primitives (http, transform, shell, etc.) that execute
//! sequentially. Each step's output is stored by name for reference in later steps
//! via `{{steps.step_name.field}}` interpolation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single step in an execution pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// Unique name within the pipeline; used for output references.
    pub name: String,
    /// The step kind and its parameters.
    #[serde(rename = "type")]
    pub step_type: StepType,
    /// If present, this step is skipped when the condition evaluates to false.
    pub condition: Option<Condition>,
}

/// Condition for conditional step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Path to a value (e.g. `steps.foo.success` or `args.enabled`).
    pub path: String,
    /// Optional operator: eq, ne, exists, not_exists, etc.
    #[serde(default)]
    pub op: ConditionOp,
    /// Optional value for comparison (for eq/ne).
    pub value: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionOp {
    #[default]
    Exists,
    NotExists,
    Eq,
    Ne,
    Truthy,
    Falsy,
}

/// All step kinds with their type-specific parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StepType {
    Http(HttpStep),
    Transform(TransformStep),
    Shell(ShellStep),
    DbQuery(DbQueryStep),
    VectorSearch(VectorSearchStep),
    LlmCall(LlmCallStep),
    Conditional(ConditionalStep),
    Loop(LoopStep),
    Return(ReturnStep),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpStep {
    pub method: String,
    pub url: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub body: Option<serde_json::Value>,
    pub auth_ref: Option<String>,
    pub extract: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformStep {
    pub input: String,
    pub operations: Vec<TransformOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TransformOperation {
    MapFields {
        mapping: HashMap<String, String>,
    },
    Filter {
        path: String,
        value: serde_json::Value,
    },
    Flatten {
        path: String,
    },
    JsonPath {
        path: String,
    },
    StringFormat {
        template: String,
    },
    RegexExtract {
        pattern: String,
        group: Option<usize>,
    },
    Merge {
        with: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellStep {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub parse_output: ShellParseOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellParseOutput {
    #[default]
    Raw,
    Lines,
    Json,
    Regex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbQueryStep {
    pub connection_ref: String,
    pub query: String,
    #[serde(default)]
    pub params: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchStep {
    pub backend_ref: Option<String>,
    pub query_text: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub namespace: Option<String>,
    pub filter: Option<serde_json::Value>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCallStep {
    pub prompt_template: String,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalStep {
    pub condition: Condition,
    pub then_steps: Vec<Step>,
    #[serde(default)]
    pub else_steps: Vec<Step>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopStep {
    pub over: String,
    pub as_var: String,
    pub body_steps: Vec<Step>,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnStep {
    pub value: String,
}

/// Runtime context for step execution: args and accumulated step outputs.
#[derive(Debug, Clone, Default)]
pub struct StepContext {
    pub args: serde_json::Value,
    pub step_outputs: HashMap<String, serde_json::Value>,
    pub loop_vars: HashMap<String, serde_json::Value>,
}
