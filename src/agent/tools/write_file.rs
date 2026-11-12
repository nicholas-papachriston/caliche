use crate::agent::tools::common::{errors, params, path};
use crate::agent::tools::file_creation_context;
use crate::agent::tools::validators;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::io::Write;
use std::path::Path;

const NAME: &str = "fs_write_internal";
const DESCRIPTION: &str =
    "Write or create a file with specified content. Creates parent directories as needed.";
const OVERWRITE_GUARD_MIN_EXISTING: u64 = 128;
const OVERWRITE_GUARD_MAX_NEW: u64 = 15;
const OVERWRITE_GUARD_RATIO: u64 = 20;

fn is_catastrophic_overwrite(existing_len: u64, new_len: usize, append: bool, force: bool) -> bool {
    if append || force {
        return false;
    }
    let new_len = u64::try_from(new_len).unwrap_or(u64::MAX);
    existing_len >= OVERWRITE_GUARD_MIN_EXISTING
        && new_len <= OVERWRITE_GUARD_MAX_NEW
        && new_len.saturating_mul(OVERWRITE_GUARD_RATIO) < existing_len
}

fn catastrophic_overwrite_error(path_str: &str, existing_len: u64, new_len: usize) -> ToolResult {
    ToolResult::error(format!(
        "Catastrophic overwrite prevented: existing file '{path_str}' is {existing_len} bytes but \
         new content is only {new_len} bytes. Use `fs edit` for targeted changes, \
         or pass force=true to overwrite unconditionally."
    ))
}

fn write_file_sync(
    file_path: &Path,
    path_str: &str,
    content: &[u8],
    create_dirs: bool,
    append: bool,
) -> Result<(), String> {
    // The open below creates missing directories, following no link.
    if !create_dirs
        && let Some(parent) = file_path.parent()
        && !parent.exists()
    {
        return Err(format!(
            "Parent directory does not exist: {}. Set create_dirs=true to create it automatically.",
            parent.display()
        ));
    }

    if append {
        let mut file = path::open_tool_file(file_path, path::ToolOpen::Append)
            .map_err(|e| errors::format_file_error("open", path_str, &e.to_string()))?;
        file.write_all(content)
            .map_err(|e| errors::format_file_error("write", path_str, &e.to_string()))?;
        file.flush()
            .map_err(|e| errors::format_file_error("write", path_str, &e.to_string()))?;
        return Ok(());
    }
    path::write_tool_file(file_path, content)
        .map_err(|e| errors::format_file_error("write", path_str, &e.to_string()))
}

#[cfg(feature = "sandbox")]
async fn existing_file_len(
    sandbox: &dyn caliche_sandbox::SandboxHandleOps,
    path_str: &str,
) -> Option<u64> {
    let host = std::path::Path::new(path_str);
    if host.is_absolute()
        && let Ok(metadata) = path::tool_metadata(host)
    {
        return Some(metadata.len());
    }
    sandbox
        .read_file(path_str)
        .await
        .ok()
        .map(|bytes| u64::try_from(bytes.len()).unwrap_or(u64::MAX))
}

#[derive(Debug)]
pub struct WriteFileTool;

impl ToolMeta for WriteFileTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write. Can be relative or absolute. Parent directories will be created if create_dirs is true."
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Whether to create parent directories if they don't exist (default: true)",
                    "default": true
                },
                "append": {
                    "type": "boolean",
                    "description": "Whether to append to the file instead of overwriting (default: false)",
                    "default": false
                },
                "force": {
                    "type": "boolean",
                    "description": "Skip the catastrophic-overwrite guard when overwriting an existing file (default: false)",
                    "default": false
                }
            },
            "required": ["path", "content"]
        })
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let content = match params::extract_string(&arguments, "content", None) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(e),
        };

        let create_dirs = params::extract_bool(&arguments, "create_dirs", true);
        let append = params::extract_bool(&arguments, "append", false);
        let force = params::extract_bool(&arguments, "force", false);

        let file_path = match path::resolve_write_path_for_tool(&path_str) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let validation_issues = {
            let is_lua = file_path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("lua"));
            if is_lua {
                validators::validate(&file_path, &content).err()
            } else {
                None
            }
        };

        // Create parent directories if needed / write on the blocking pool.
        let is_new = !file_path.exists();

        if !is_new
            && let Ok(metadata) = path::tool_metadata(&file_path)
            && is_catastrophic_overwrite(metadata.len(), content.len(), append, force)
        {
            return catastrophic_overwrite_error(&path_str, metadata.len(), content.len());
        }

        if let Some(msg) = file_creation_context::check_may_create(&file_path, is_new) {
            return ToolResult {
                success: false,
                content: msg.clone(),
                error: None,
                typed_payload: None,
                usage: None,
                warnings: Some(vec![msg]),
            };
        }

        let write_path = file_path.clone();
        let write_content = content;
        let path_for_io = path_str.clone();
        let io = crate::agent::tools::common::spawn_tool_blocking(move || {
            write_file_sync(
                &write_path,
                &path_for_io,
                write_content.as_bytes(),
                create_dirs,
                append,
            )
        })
        .await;
        let result = match io {
            Ok(inner) => inner,
            Err(e) => {
                return ToolResult::error(errors::format_file_error(
                    "write",
                    &path_str,
                    &e.to_string(),
                ));
            }
        };

        match result {
            Ok(()) => {
                if is_new {
                    file_creation_context::record_created(file_path.clone());
                }
                let action = if append { "appended to" } else { "written to" };
                let ok_msg = format!("Content successfully {action} file '{path_str}'");
                if let Some(validation_errors) = validation_issues {
                    return ToolResult {
                        success: false,
                        content: format!(
                            "{}\n\nFile written but has syntax or style issues:\n{}\n\nConsider reviewing and fixing these before continuing.",
                            ok_msg,
                            validation_errors.join("\n")
                        ),
                        error: None,
                        typed_payload: None,
                        usage: None,
                        warnings: Some(validation_errors),
                    };
                }
                ToolResult::success(ok_msg)
            }
            Err(e) => ToolResult::error(e),
        }
    }

    #[cfg(feature = "sandbox")]
    async fn execute_sandboxed(
        &self,
        arguments: serde_json::Value,
        sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> ToolResult {
        let Some(sandbox) = sandbox else {
            return self.execute(arguments).await;
        };
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };
        let content = match params::extract_string(&arguments, "content", None) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(e),
        };
        let create_dirs = params::extract_bool(&arguments, "create_dirs", true);
        let append = params::extract_bool(&arguments, "append", false);
        let force = params::extract_bool(&arguments, "force", false);
        if let Some(existing_len) = existing_file_len(sandbox, &path_str).await
            && is_catastrophic_overwrite(existing_len, content.len(), append, force)
        {
            return catastrophic_overwrite_error(&path_str, existing_len, content.len());
        }
        if create_dirs
            && let Some(parent) = std::path::Path::new(&path_str)
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
        {
            let dir = parent.to_string_lossy().replace('\'', "'\\''");
            let mkdir_script = format!("mkdir -p '{dir}'");
            let cmd: Vec<&str> = vec!["sh", "-c", &mkdir_script];
            match sandbox
                .exec(&cmd, None, None, std::time::Duration::from_secs(5))
                .await
            {
                Ok(r) if r.exit_code != 0 => {
                    return ToolResult::error(format!(
                        "Sandbox mkdir (create_dirs) failed: exit {} - {}",
                        r.exit_code, r.stderr
                    ));
                }
                Err(e) => {
                    return ToolResult::error(format!("Sandbox mkdir (create_dirs) failed: {e}"));
                }
                _ => {}
            }
        }
        if append {
            match sandbox.read_file(&path_str).await {
                Ok(existing) => {
                    let mut s = String::from_utf8_lossy(&existing).into_owned();
                    s.push_str(&content);
                    if let Err(e) = sandbox.write_file(&path_str, s.as_bytes()).await {
                        return ToolResult::error(format!(
                            "Sandbox write_file (append) failed: {e}"
                        ));
                    }
                }
                Err(e) => {
                    return ToolResult::error(format!("Sandbox read_file for append failed: {e}"));
                }
            }
        } else if let Err(e) = sandbox.write_file(&path_str, content.as_bytes()).await {
            return ToolResult::error(format!("Sandbox write_file failed: {e}"));
        }
        let action = if append { "appended to" } else { "written to" };
        ToolResult::success(format!("Content successfully {action} file '{path_str}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::common::set_tool_host_fs_unlocked;
    use tempfile::TempDir;

    /// Unlock host FS for a test and return a guard that re-locks on drop.
    fn unlock_host_fs() -> impl Drop {
        set_tool_host_fs_unlocked(true);
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                set_tool_host_fs_unlocked(false);
            }
        }
        Guard
    }

    #[tokio::test]
    async fn test_write_new_file() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "content": "Hello, World!"
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        // Verify file was created
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("Failed to read test file");
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_overwrite_file() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");

        // Create initial file
        tokio::fs::write(&file_path, "Old content")
            .await
            .expect("Failed to write initial file");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "content": "New content"
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        // Verify file was overwritten
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("Failed to read test file");
        assert_eq!(content, "New content");
    }

    #[tokio::test]
    async fn test_append_to_file() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");

        // Create initial file
        tokio::fs::write(&file_path, "Line 1\n")
            .await
            .expect("Failed to write initial file");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "content": "Line 2\n",
            "append": true
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        // Verify content was appended
        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("Failed to read test file");
        assert_eq!(content, "Line 1\nLine 2\n");
    }

    #[tokio::test]
    async fn test_create_directories() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("subdir").join("test.txt");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "content": "Content",
            "create_dirs": true
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        // Verify file was created
        assert!(file_path.exists());
    }

    // --- overwrite guard tests ---

    #[tokio::test]
    async fn test_overwrite_guard_rejects_tiny_overwrite() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("big.rs");
        // Write a file >= 128 bytes.
        let big_content = "x".repeat(200);
        tokio::fs::write(&file_path, &big_content)
            .await
            .expect("write initial");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("path"),
            "content": "A"
        });
        let result = tool.execute(args).await;
        assert!(!result.success, "should be rejected");
        let err = result.error.as_deref().unwrap_or("");
        assert!(
            err.contains("200") && err.contains("1"),
            "error should mention sizes: {err}"
        );
        assert!(
            err.contains("force=true") || err.contains("fs edit"),
            "error should mention remedy: {err}"
        );
        // File must not be modified.
        let on_disk = tokio::fs::read_to_string(&file_path).await.expect("read");
        assert_eq!(on_disk, big_content);
    }

    #[tokio::test]
    async fn test_overwrite_guard_force_bypasses_check() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("big.rs");
        tokio::fs::write(&file_path, "x".repeat(200))
            .await
            .expect("write initial");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("path"),
            "content": "A",
            "force": true
        });
        let result = tool.execute(args).await;
        assert!(result.success, "force=true should bypass guard");
        let on_disk = tokio::fs::read_to_string(&file_path).await.expect("read");
        assert_eq!(on_disk, "A");
    }

    #[tokio::test]
    async fn test_overwrite_guard_allows_small_existing_file() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("small.txt");
        // Existing file is only 10 bytes (< 128), so guard must not fire.
        tokio::fs::write(&file_path, "hello")
            .await
            .expect("write initial");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("path"),
            "content": "A"
        });
        let result = tool.execute(args).await;
        assert!(result.success, "small existing file should not be blocked");
    }

    #[tokio::test]
    async fn test_overwrite_guard_allows_large_to_large() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("large.rs");
        tokio::fs::write(&file_path, "x".repeat(300))
            .await
            .expect("write initial");

        let tool = WriteFileTool;
        // New content is 200 bytes: 200 * 20 = 4000 >= 300, so not rejected.
        let args = json!({
            "path": file_path.to_str().expect("path"),
            "content": "y".repeat(200)
        });
        let result = tool.execute(args).await;
        assert!(result.success, "large-to-large overwrite should be allowed");
    }

    #[tokio::test]
    async fn test_overwrite_guard_allows_new_file() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("new.rs");
        // File does not exist yet.
        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("path"),
            "content": "A"
        });
        let result = tool.execute(args).await;
        assert!(result.success, "new file should never be blocked");
    }

    #[tokio::test]
    async fn test_overwrite_guard_allows_append() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("append.rs");
        tokio::fs::write(&file_path, "x".repeat(200))
            .await
            .expect("write initial");

        let tool = WriteFileTool;
        let args = json!({
            "path": file_path.to_str().expect("path"),
            "content": "A",
            "append": true
        });
        let result = tool.execute(args).await;
        assert!(result.success, "append should never be blocked");
    }

    #[tokio::test]
    async fn test_path_traversal_blocked() {
        let tool = WriteFileTool;
        let args = json!({
            "path": "../../etc/passwd",
            "content": "malicious"
        });

        let result = tool.execute(args).await;
        assert!(!result.success, "Expected tool execution to fail");
        assert!(
            result.error.is_some(),
            "Expected error message, but error field was None. Result: {:?}",
            result
        );
        let error_msg = result.error.as_ref().expect("Error should be Some");
        assert!(
            error_msg.to_lowercase().contains("path traversal"),
            "Error message should contain 'path traversal', but got: {:?}",
            error_msg
        );
    }
}
