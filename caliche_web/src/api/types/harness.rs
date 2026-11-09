//! Generic CLI harness registry types.

use serde::{Deserialize, Serialize};

use crate::harness::PromptPlacementKind;

/// Request to register an external CLI harness.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreateHarnessRequest {
    pub name: String,
    #[serde(default)]
    pub preset_id: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default)]
    pub prompt_placement: PromptPlacementKind,
    /// Capabilities used by coordinator routing.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Registered harness returned by list/create.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HarnessInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub preset_id: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub prompt_placement: PromptPlacementKind,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub last_output: Option<String>,
    #[serde(default)]
    pub last_dispatched_at: Option<String>,
}

/// Response from GET list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarnessListResponse {
    pub harnesses: Vec<HarnessInfo>,
}

/// Request to run one task on a registered harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchHarnessRequest {
    pub message: String,
}

/// Result of one harness dispatch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DispatchHarnessResponse {
    pub harness_id: String,
    pub name: String,
    pub ok: bool,
    pub text: String,
}
