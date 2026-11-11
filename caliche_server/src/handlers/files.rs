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
use std::io::Read;
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
#[cfg(feature = "document-loaders")]
pub(crate) fn validate_workspace_path(path: &Path) -> Result<PathBuf> {
    resolve_in_workspace(&get_workspace_root(), path).map(|(_, target)| target)
}

/// The canonical workspace and the canonical `path` inside it.
fn resolve_in_workspace(workspace: &Path, path: &Path) -> Result<(PathBuf, PathBuf)> {
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

    Ok((canonical_workspace, canonical_target))
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
    let requested = query.path;
    tokio::task::spawn_blocking(move || list_workspace_dir(workspace, requested))
        .await
        .unwrap_or_else(|join| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Directory listing task failed: {join}") })),
            )
                .into_response()
        })
}

/// List a directory inside `workspace` for the file browser (blocking).
///
/// The path is validated, then listed through a walk below the canonical
/// workspace that follows no link, so a directory swapped for a symlink
/// after validation is refused with 403 instead of listed.
fn list_workspace_dir(workspace: PathBuf, requested: Option<String>) -> axum::response::Response {
    let target_path = requested
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.clone());

    let (root, canonical_path) = match resolve_in_workspace(&workspace, &target_path) {
        Ok(resolved) => resolved,
        Err(e) => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };
    after_validate(&canonical_path);

    let listed = match caliche_sandbox::list_dir_beneath(&root, &canonical_path) {
        Ok(listed) => listed,
        Err(e) => {
            let status = match e.kind() {
                std::io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
                std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
                std::io::ErrorKind::NotADirectory => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return (
                status,
                Json(serde_json::json!({ "error": format!("Failed to read directory: {e}") })),
            )
                .into_response();
        }
    };
    let mut entries: Vec<FileEntry> = listed
        .into_iter()
        .filter_map(|entry| {
            let name = entry.name.to_string_lossy().to_string();
            // Skip hidden files (starting with .)
            if name.starts_with('.') {
                return None;
            }
            let path = canonical_path.join(&entry.name);
            let extension = path.extension().map(|e| e.to_string_lossy().to_string());
            let is_file = entry.kind == caliche_sandbox::EntryKind::File;
            Some(FileEntry {
                name,
                path: path.to_string_lossy().to_string(),
                is_dir: entry.kind == caliche_sandbox::EntryKind::Dir,
                is_file,
                size: is_file.then_some(entry.len),
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
    let max_lines = query.max_lines.unwrap_or(DEFAULT_MAX_LINES);
    let workspace = get_workspace_root();
    let read =
        tokio::task::spawn_blocking(move || read_workspace_file(&workspace, &path, max_lines))
            .await;
    match read {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err((status, error))) => {
            (status, Json(serde_json::json!({ "error": error }))).into_response()
        }
        Err(join) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("File read task failed: {join}") })),
        )
            .into_response(),
    }
}

/// Read a file inside `workspace` for the file browser.
///
/// The path is validated, then opened below the canonical workspace without
/// following any link, so a directory swapped for a symlink after validation
/// is refused with 403. At most `MAX_FILE_SIZE + 1` bytes are read, so a file
/// over the limit (or one that grew after a size check) is refused with 400.
fn read_workspace_file(
    workspace: &Path,
    path: &Path,
    max_lines: usize,
) -> std::result::Result<ReadFileResponse, (StatusCode, String)> {
    let (root, target) = resolve_in_workspace(workspace, path)
        .map_err(|e| (StatusCode::FORBIDDEN, e.to_string()))?;
    // A by-path check only picks the 400 message for a directory; the open
    // below is what refuses anything but a regular file.
    if !target.is_file() {
        return Err((StatusCode::BAD_REQUEST, "Path is not a file".to_string()));
    }
    after_validate(&target);

    let file = caliche_sandbox::open_file_beneath(&root, &target).map_err(|e| {
        let status = match e.kind() {
            std::io::ErrorKind::PermissionDenied => StatusCode::FORBIDDEN,
            std::io::ErrorKind::NotFound => StatusCode::NOT_FOUND,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, format!("Failed to open file: {e}"))
    })?;
    let metadata = file.metadata().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read file metadata: {e}"),
        )
    })?;
    if !metadata.is_file() {
        return Err((StatusCode::BAD_REQUEST, "Path is not a file".to_string()));
    }

    let mut bytes = Vec::new();
    file.take(MAX_FILE_SIZE + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read file: {e}"),
            )
        })?;
    let size = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if size > MAX_FILE_SIZE {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "File is too large (over {MAX_FILE_SIZE} bytes). Maximum is {MAX_FILE_SIZE} bytes."
            ),
        ));
    }
    let content = String::from_utf8(bytes).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read file (may be binary): {e}"),
        )
    })?;

    // Apply line limit
    let lines: Vec<&str> = content.lines().collect();
    let line_count = lines.len();
    let truncated = line_count > max_lines;

    let final_content = if truncated {
        lines[..max_lines].join("\n")
    } else {
        content
    };

    Ok(ReadFileResponse {
        path: target.to_string_lossy().to_string(),
        content: final_content,
        size,
        truncated,
        line_count,
    })
}

#[cfg(test)]
thread_local! {
    /// Test hook that runs between validation and open, to swap a component.
    static AFTER_VALIDATE: std::cell::RefCell<Option<Box<dyn FnOnce(&Path)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn after_validate(target: &Path) {
    if let Some(hook) = AFTER_VALIDATE.with(|cell| cell.borrow_mut().take()) {
        hook(target);
    }
}

#[cfg(not(test))]
const fn after_validate(_target: &Path) {}

#[cfg(test)]
mod tests {
    use super::{AFTER_VALIDATE, MAX_FILE_SIZE, list_workspace_dir, read_workspace_file};
    use axum::http::StatusCode;
    use std::path::{Path, PathBuf};

    fn workspace() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical workspace");
        (dir, root)
    }

    fn file_of_len(root: &Path, name: &str, len: u64) {
        let bytes = vec![b'a'; usize::try_from(len).expect("len fits usize")];
        std::fs::write(root.join(name), bytes).expect("write file");
    }

    #[test]
    fn read_at_the_limit_returns_the_file() {
        let (_dir, root) = workspace();
        file_of_len(&root, "max.txt", MAX_FILE_SIZE);
        let response = read_workspace_file(&root, Path::new("max.txt"), 10).expect("read");
        assert_eq!(response.size, 1_048_576);
        assert_eq!(response.line_count, 1);
    }

    #[test]
    fn read_of_a_directory_is_400_not_a_file() {
        let (_dir, root) = workspace();
        std::fs::create_dir(root.join("sub")).expect("sub");
        let Err((status, error)) = read_workspace_file(&root, Path::new("sub"), 10) else {
            panic!("a directory must be refused");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error, "Path is not a file");
    }

    #[test]
    fn read_over_the_limit_is_400_too_large() {
        let (_dir, root) = workspace();
        file_of_len(&root, "big.txt", MAX_FILE_SIZE + 1);
        let Err((status, error)) = read_workspace_file(&root, Path::new("big.txt"), 10) else {
            panic!("a file over the limit must be refused");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error.contains("too large"), "unexpected error: {error}");
    }

    async fn status_and_json(
        response: axum::response::Response,
    ) -> (StatusCode, serde_json::Value) {
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body");
        (status, serde_json::from_slice(&bytes).expect("json body"))
    }

    #[tokio::test]
    async fn list_returns_entries_with_directories_first() {
        let (_dir, root) = workspace();
        std::fs::create_dir(root.join("sub")).expect("sub");
        std::fs::write(root.join("sub/b.txt"), b"abc").expect("b.txt");
        std::fs::write(root.join("sub/.hidden"), b"h").expect(".hidden");
        std::fs::create_dir(root.join("sub/a")).expect("a");
        let sub = root.join("sub");

        let (status, body) = status_and_json(list_workspace_dir(
            root.clone(),
            Some(sub.to_string_lossy().to_string()),
        ))
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            serde_json::json!({
                "path": sub.to_string_lossy(),
                "parent": root.to_string_lossy(),
                "entries": [
                    {
                        "name": "a",
                        "path": sub.join("a").to_string_lossy(),
                        "is_dir": true,
                        "is_file": false,
                        "size": null,
                        "extension": null
                    },
                    {
                        "name": "b.txt",
                        "path": sub.join("b.txt").to_string_lossy(),
                        "is_dir": false,
                        "is_file": true,
                        "size": 3,
                        "extension": "txt"
                    }
                ]
            })
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn list_of_a_directory_swapped_for_a_symlink_is_403() {
        let (_dir, root) = workspace();
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("secret.txt"), b"secret").expect("outside file");
        std::fs::create_dir(root.join("sub")).expect("sub");
        let sub = root.join("sub");

        let ws = root.clone();
        let target = outside.path().to_path_buf();
        AFTER_VALIDATE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move |_validated: &Path| {
                std::fs::rename(ws.join("sub"), ws.join("sub.moved")).expect("move sub");
                std::os::unix::fs::symlink(&target, ws.join("sub")).expect("symlink");
            }));
        });

        let (status, body) = status_and_json(list_workspace_dir(
            root.clone(),
            Some(sub.to_string_lossy().to_string()),
        ))
        .await;

        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(
            body,
            serde_json::json!({
                "error": format!(
                    "Failed to read directory: refusing to follow symlink at 'sub' in '{}'",
                    sub.display()
                )
            })
        );
    }

    #[cfg(unix)]
    #[test]
    fn read_through_a_parent_swapped_for_a_symlink_is_403() {
        let (_dir, root) = workspace();
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("secret.txt"), b"secret").expect("outside file");
        std::fs::create_dir(root.join("sub")).expect("sub");
        std::fs::write(root.join("sub/secret.txt"), b"inside").expect("inside file");

        let ws = root.clone();
        let target = outside.path().to_path_buf();
        AFTER_VALIDATE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move |_validated: &Path| {
                std::fs::rename(ws.join("sub"), ws.join("sub.moved")).expect("move sub");
                std::os::unix::fs::symlink(&target, ws.join("sub")).expect("symlink");
            }));
        });

        let Err((status, error)) = read_workspace_file(&root, Path::new("sub/secret.txt"), 10)
        else {
            panic!("a swapped parent must be refused");
        };
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(error.contains("symlink"), "unexpected error: {error}");
    }
}
