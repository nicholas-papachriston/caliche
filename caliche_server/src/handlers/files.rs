//! File browser handlers for exploring local files.
//!
//! These endpoints allow the UI to browse and read local files for attaching
//! context to chat conversations.

use anyhow::{Context, Result, bail};
use axum::{
    Json,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::AppState;

/// Query parameters for listing a directory
#[derive(Debug, Deserialize)]
pub struct ListDirQuery {
    /// Path to list (defaults to current working directory)
    pub path: Option<String>,
}

/// Query parameters for reading a file
#[derive(Debug, Deserialize)]
pub struct ReadFileQuery {
    /// Path to the file to read
    pub path: String,
    /// Maximum number of lines to read (optional)
    pub max_lines: Option<usize>,
}

/// A file or directory entry
#[derive(Debug, Serialize)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub is_file: bool,
    pub size: Option<u64>,
    pub extension: Option<String>,
}

/// Response for directory listing
#[derive(Debug, Serialize)]
pub struct ListDirResponse {
    pub path: String,
    pub parent: Option<String>,
    pub entries: Vec<FileEntry>,
}

/// Response for file reading
#[derive(Debug, Serialize)]
pub struct ReadFileResponse {
    pub path: String,
    pub content: String,
    pub size: u64,
    pub truncated: bool,
    pub line_count: usize,
}

/// Error response
#[derive(Debug, Serialize)]
pub struct FileError {
    pub error: String,
}

/// Maximum file size to read (1MB)
const MAX_FILE_SIZE: u64 = 1024 * 1024;

/// Maximum lines to read by default
const DEFAULT_MAX_LINES: usize = 1000;

/// Check if the API should allow file browsing
fn is_file_browsing_enabled() -> bool {
    std::env::var("CALICHE_FILE_BROWSING")
        .map(|v| v.to_lowercase() == "true" || v == "1")
        .unwrap_or(false)
}

/// Get the base path for file browsing (workspace root)
fn get_workspace_root() -> PathBuf {
    std::env::var("CALICHE_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Validate that a path is within the workspace.
pub(crate) fn validate_workspace_path(path: &Path) -> Result<PathBuf> {
    let workspace = get_workspace_root();

    // Canonicalize both paths for comparison
    let canonical_workspace = workspace
        .canonicalize()
        .context("Failed to resolve workspace")?;

    let target = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };

    let canonical_target = target.canonicalize().context("Failed to resolve path")?;

    // Ensure the target is within the workspace
    if !canonical_target.starts_with(&canonical_workspace) {
        bail!("Access denied: path is outside workspace");
    }

    Ok(canonical_target)
}

/// Handler to check if file browsing is enabled
pub async fn files_enabled_handler() -> impl IntoResponse {
    let enabled = is_file_browsing_enabled();
    let workspace = if enabled {
        Some(get_workspace_root().to_string_lossy().to_string())
    } else {
        None
    };

    Json(serde_json::json!({
        "enabled": enabled,
        "workspace": workspace
    }))
}

/// Handler for listing directory contents
pub async fn list_files_handler(
    State(_state): State<AppState>,
    Query(query): Query<ListDirQuery>,
) -> impl IntoResponse {
    // Check if file browsing is enabled
    if !is_file_browsing_enabled() {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "File browsing is not enabled. Set CALICHE_FILE_BROWSING=true to enable."
            })),
        )
            .into_response();
    }

    let workspace = get_workspace_root();
    let target_path = query
        .path
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.clone());

    // Validate the path
    let canonical_path = match validate_workspace_path(&target_path) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    // Read directory entries
    let entries = match std::fs::read_dir(&canonical_path) {
        Ok(read_dir) => {
            let mut entries: Vec<FileEntry> = read_dir
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    let metadata = entry.metadata().ok()?;
                    let name = entry.file_name().to_string_lossy().to_string();

                    // Skip hidden files (starting with .)
                    if name.starts_with('.') {
                        return None;
                    }

                    let path = entry.path();
                    let extension = path.extension().map(|e| e.to_string_lossy().to_string());

                    Some(FileEntry {
                        name,
                        path: path.to_string_lossy().to_string(),
                        is_dir: metadata.is_dir(),
                        is_file: metadata.is_file(),
                        size: if metadata.is_file() {
                            Some(metadata.len())
                        } else {
                            None
                        },
                        extension,
                    })
                })
                .collect();

            // Sort: directories first, then files, alphabetically
            entries.sort_by(|a, b| match (a.is_dir, b.is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            });

            entries
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to read directory: {}", e) })),
            )
                .into_response();
        }
    };

    // Compute the canonical parent path and only expose it if it still lives
    // inside the canonical workspace root. Silently falling back to the un-
    // canonicalised `workspace` would let a symlink escape the sandbox, so we
    // explicitly require the workspace to be resolvable and fail closed.
    let parent = match workspace.canonicalize() {
        Ok(canonical_workspace) => canonical_path
            .parent()
            .filter(|p| p.starts_with(&canonical_workspace))
            .map(|p| p.to_string_lossy().to_string()),
        Err(err) => {
            tracing::warn!(
                error = %err,
                workspace = %workspace.display(),
                "failed to canonicalize workspace while computing parent; suppressing parent link"
            );
            None
        }
    };

    Json(ListDirResponse {
        path: canonical_path.to_string_lossy().to_string(),
        parent,
        entries,
    })
    .into_response()
}

/// Handler for reading file contents
pub async fn read_file_handler(
    State(_state): State<AppState>,
    Query(query): Query<ReadFileQuery>,
) -> impl IntoResponse {
    // Check if file browsing is enabled
    if !is_file_browsing_enabled() {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "File browsing is not enabled. Set CALICHE_FILE_BROWSING=true to enable."
            })),
        )
            .into_response();
    }

    let path = PathBuf::from(&query.path);

    // Validate the path
    let canonical_path = match validate_workspace_path(&path) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    // Check if it's a file
    if !canonical_path.is_file() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Path is not a file" })),
        )
            .into_response();
    }

    // Get file size
    let metadata = match std::fs::metadata(&canonical_path) {
        Ok(m) => m,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({ "error": format!("Failed to read file metadata: {}", e) }),
                ),
            )
                .into_response();
        }
    };

    let size = metadata.len();

    // Check file size
    if size > MAX_FILE_SIZE {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("File is too large ({} bytes). Maximum is {} bytes.", size, MAX_FILE_SIZE)
            })),
        ).into_response();
    }

    // Read file contents
    let content = match std::fs::read_to_string(&canonical_path) {
        Ok(c) => c,
        Err(e) => {
            // Try reading as binary and show error
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to read file (may be binary): {}", e) })),
            ).into_response();
        }
    };

    // Apply line limit
    let max_lines = query.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let lines: Vec<&str> = content.lines().collect();
    let line_count = lines.len();
    let truncated = line_count > max_lines;

    let final_content = if truncated {
        lines[..max_lines].join("\n")
    } else {
        content
    };

    Json(ReadFileResponse {
        path: canonical_path.to_string_lossy().to_string(),
        content: final_content,
        size,
        truncated,
        line_count,
    })
    .into_response()
}
