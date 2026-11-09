//! Session lifecycle, workspace context, memory, and file browser types.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
pub use caliche_contracts::sessions::{
    CheckpointResponseDto as CheckpointResponse, CreateSessionRequestDto as CreateSessionRequest,
    CreateSessionResponseDto as CreateSessionResponse, DiffChangeKindDto as DiffChangeKind,
    DiffFileSummaryDto as DiffFileSummary, DiffMessageSummaryDto as DiffMessageSummary,
    DiffResponseDto as DiffResponse, ModeResponseDto as ModeResponse,
    RedoResponseDto as RedoResponse, RollbackResponseDto as RollbackResponse,
    UndoResponseDto as UndoResponse,
};

/// Response for [`super::super::client::ApiClient::spawn_coding_run`]. The UI uses this
/// to attach the returned `session_id` to a [`crate::state::SubAgent`] so
/// follow-up cancel/list operations can target the right server session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingRunResponse {
    pub session_id: String,
    pub response: String,
}

/// Session info
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(
    dead_code,
    reason = "nothing builds it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
)]
pub struct SessionInfo {
    pub session_id: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<String>,
    pub turn_count: usize,
    pub is_active: bool,
}

/// One row from `sessions.get` / `GET /api/agent/sessions`.
///
/// `title` is the optional human title. It is absent when create omitted it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiveSessionInfo {
    pub session_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub name_slug: Option<String>,
    pub created_at: String,
    pub last_activity: String,
    pub turn_count: usize,
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

/// Tracked file info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedFileInfo {
    pub path: String,
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub modified: bool,
    pub read_count: usize,
}

/// Response for checking if file browsing is enabled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileBrowsingStatus {
    pub enabled: bool,
    pub workspace: Option<String>,
}

/// A file or directory entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub is_file: bool,
    pub size: Option<u64>,
    pub extension: Option<String>,
}

/// Response for directory listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDirResponse {
    pub path: String,
    pub parent: Option<String>,
    pub entries: Vec<FileEntry>,
}

/// Response for file reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileResponse {
    pub path: String,
    pub content: String,
    pub size: u64,
    pub truncated: bool,
    pub line_count: usize,
}

/// Plan document response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanResponse {
    pub content: Option<String>,
}

/// Memory stats response
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStatsResponse {
    #[serde(default)]
    pub entry_count: usize,
    pub total_tokens: Option<usize>,
    pub summary: Option<String>,
}

/// A single memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub content: String,
    pub relevance: Option<f32>,
    pub timestamp: Option<String>,
}

/// Memory search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResponse {
    #[serde(default)]
    pub results: Vec<MemoryEntry>,
    #[serde(default)]
    pub total: usize,
}

/// Reasoning pattern info
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningPatternInfo {
    pub name: String,
    pub description: String,
}

/// Available reasoning patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPatternsResponse {
    pub patterns: Vec<ReasoningPatternInfo>,
}

/// Git context info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitContextInfo {
    pub branch: String,
    pub has_changes: bool,
    pub staged_count: usize,
    pub modified_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_url: Option<String>,
}

/// Workspace context info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceContextInfo {
    pub workspace_root: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_context: Option<GitContextInfo>,
    pub tracked_files: Vec<TrackedFileInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub messages: Vec<serde_json::Value>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItemInfo {
    pub item_type: String,
    pub content: String,
    pub relevance: f32,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextItemsResponse {
    #[serde(default)]
    pub items: Vec<ContextItemInfo>,
}

/// `metrics.getCache` snapshot. Hit rates are already percentages (0–100).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetricsResponse {
    #[serde(default)]
    pub provider_hits: u64,
    #[serde(default)]
    pub provider_misses: u64,
    #[serde(default)]
    pub provider_hit_rate: f64,
    #[serde(default)]
    pub registry_hits: u64,
    #[serde(default)]
    pub registry_misses: u64,
    #[serde(default)]
    pub registry_hit_rate: f64,
}
