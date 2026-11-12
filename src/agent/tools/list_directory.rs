use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use caliche_sandbox::EntryKind;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Maximum number of entries to return in a directory listing.
/// Prevents context window overflow from huge directories.
const DEFAULT_MAX_ENTRIES: usize = 500;

/// Maximum output size in bytes (roughly 10K tokens at 4 bytes/token for ASCII text).
/// The listing is cut on a character boundary at or below this limit.
const MAX_OUTPUT_SIZE: usize = 40_000;

const NAME: &str = "list_directory";
const DESCRIPTION: &str = "Lists the contents of a directory, including files and subdirectories with their metadata (size, type). Output is limited to prevent context window overflow - use 'max_entries' to control the limit.";
#[derive(Debug)]
pub struct ListDirectoryTool;

impl ToolMeta for ListDirectoryTool {
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
                    "description": "The path to the directory to list (relative or absolute). Defaults to current directory '.'"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to recursively list subdirectories (default: false). Use with caution on large directories.",
                    "default": false
                },
                "max_depth": {
                    "type": "number",
                    "description": "Maximum depth for recursive listing (default: 2, only used if recursive is true)",
                    "default": 2
                },
                "max_entries": {
                    "type": "number",
                    "description": "Maximum number of entries to return (default: 500). Use smaller values for large directories.",
                    "default": 500
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files and directories.",
                    "default": false
                },
                "dirs_first": {
                    "type": "boolean",
                    "description": "List directories before files.",
                    "default": true
                },
                "extensions_summary": {
                    "type": "boolean",
                    "description": "Append file extension counts.",
                    "default": false
                }
            },
            "required": []
        })
    }
}

#[async_trait]
impl Tool for ListDirectoryTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let recursive = params::extract_bool(&arguments, "recursive", false);
        let max_depth = params::extract_number_or(&arguments, "max_depth", 2) as usize;
        let max_entries =
            params::extract_number_or(&arguments, "max_entries", DEFAULT_MAX_ENTRIES as u64)
                as usize;
        let include_hidden = params::extract_bool(&arguments, "include_hidden", false);
        let dirs_first = params::extract_bool(&arguments, "dirs_first", true);
        let extensions_summary = params::extract_bool(&arguments, "extensions_summary", false);

        // Cap max_entries to prevent abuse
        let max_entries = max_entries.min(2000);

        // Validate and resolve path (must be a directory)
        let normalized_path = match path::validate_and_resolve_path(&path_str, Some(false)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let worker_path = normalized_path;
        let listed = tokio::task::spawn_blocking(move || {
            let mut entry_count = 0usize;
            let output = if recursive {
                ListDirectoryTool::list_directory_recursive(
                    worker_path,
                    0,
                    max_depth,
                    max_entries,
                    &mut entry_count,
                    include_hidden,
                    dirs_first,
                )
            } else {
                ListDirectoryTool::list_directory_simple(
                    &worker_path,
                    max_entries,
                    &mut entry_count,
                    include_hidden,
                    dirs_first,
                    extensions_summary,
                )
            };
            output.map(|output| (output, entry_count))
        })
        .await;

        match listed {
            Ok(Ok((mut output, total_entries))) => {
                if output.len() > MAX_OUTPUT_SIZE {
                    let cut =
                        caliche_contracts::text::truncate_utf8(&output, MAX_OUTPUT_SIZE).len();
                    output.truncate(cut);
                    output.push_str("\n\n[Output truncated - too large for context window]");
                }

                if total_entries >= max_entries {
                    output.push_str(&format!(
                        "\n\n[Listing limited to {max_entries} entries. Use 'max_entries' parameter or filter by specific subdirectory for more results.]"
                    ));
                }

                ToolResult::success_with_payload(
                    output,
                    json!({
                        "op": "list",
                        "path": path_str,
                        "recursive": recursive,
                        "max_depth": max_depth,
                        "entry_count": total_entries.min(max_entries),
                        "max_entries": max_entries,
                        "truncated": total_entries >= max_entries,
                    }),
                )
            }
            Ok(Err(e)) => ToolResult::error(format!("Failed to list directory '{path_str}': {e}")),
            Err(e) => ToolResult::error(format!("list worker failed: {e}")),
        }
    }
}

impl ListDirectoryTool {
    fn list_directory_simple(
        path: &Path,
        max_entries: usize,
        entry_count: &mut usize,
        include_hidden: bool,
        dirs_first: bool,
        extensions_summary: bool,
    ) -> Result<String> {
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        let mut extension_counts = std::collections::BTreeMap::<String, usize>::new();

        for entry in path::list_resolved_dir(path)
            .with_context(|| format!("Failed to read directory '{}'", path.display()))?
        {
            if *entry_count >= max_entries {
                break;
            }
            let file_name = entry.name.to_str().unwrap_or("?");
            if !include_hidden && file_name.starts_with('.') {
                continue;
            }
            *entry_count = entry_count.saturating_add(1);

            let is_dir = entry.kind == EntryKind::Dir;
            let is_file = entry.kind == EntryKind::File;
            let entry_type = if is_dir {
                "dir "
            } else if is_file {
                "file"
            } else {
                "other"
            };
            let size = if is_file {
                let len = entry.len;
                if extensions_summary {
                    let ext = Path::new(&entry.name)
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map_or_else(|| "[no extension]".to_string(), |ext| format!(".{ext}"));
                    *extension_counts.entry(ext).or_insert(0) += 1;
                }
                Self::format_size(len)
            } else {
                "-".to_string()
            };
            let line = format!("  {entry_type} [{size}] {file_name}");
            if is_dir {
                dirs.push(line);
            } else {
                files.push(line);
            }
        }

        dirs.sort();
        files.sort();
        let mut results = if dirs_first {
            dirs.into_iter().chain(files).collect::<Vec<_>>()
        } else {
            files.into_iter().chain(dirs).collect::<Vec<_>>()
        };
        if extensions_summary && !extension_counts.is_empty() {
            results.push(String::new());
            results.push("Extension summary:".to_string());
            results.extend(
                extension_counts
                    .into_iter()
                    .map(|(ext, count)| format!("  {ext}: {count}")),
            );
        }
        Ok(format!(
            "Directory: {}\n{}",
            path.display(),
            results.join("\n")
        ))
    }

    fn format_size(bytes: u64) -> String {
        if bytes < 1024 {
            format!("{bytes} B")
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    fn list_directory_recursive(
        path: PathBuf,
        current_depth: usize,
        max_depth: usize,
        max_entries: usize,
        entry_count: &mut usize,
        include_hidden: bool,
        dirs_first: bool,
    ) -> Result<String> {
        if *entry_count >= max_entries {
            return Ok(String::new());
        }

        if current_depth > max_depth {
            return Ok("  [max depth reached]\n".to_string());
        }

        let mut output = Vec::new();
        let indent = "  ".repeat(current_depth);
        output.push(format!("{}Directory: {}", indent, path.display()));

        let mut files = Vec::new();
        let mut dirs = Vec::new();
        for entry in path::list_resolved_dir(&path)
            .with_context(|| format!("Failed to read directory '{}'", path.display()))?
        {
            if *entry_count >= max_entries {
                break;
            }
            let file_name = entry.name.to_string_lossy().into_owned();
            if !include_hidden && file_name.starts_with('.') {
                continue;
            }
            if entry.kind == EntryKind::Dir {
                dirs.push((path.join(&entry.name), file_name));
            } else {
                files.push((file_name, entry.len));
            }
            *entry_count = entry_count.saturating_add(1);
        }

        files.sort_by(|a, b| a.0.cmp(&b.0));
        dirs.sort_by(|a, b| a.1.cmp(&b.1));

        let render_files = |output: &mut Vec<String>, files: &[(String, u64)]| {
            for (file_name, size) in files {
                output.push(format!(
                    "{}  {} ({})",
                    indent,
                    file_name,
                    Self::format_size(*size)
                ));
            }
        };
        let mut render_dirs =
            |output: &mut Vec<String>, dirs: Vec<(PathBuf, String)>| -> Result<()> {
                for (dir_path, dir_name) in dirs {
                    if *entry_count >= max_entries {
                        output.push(format!("{indent}  [DIR] {dir_name} [...]"));
                        break;
                    }
                    output.push(format!("{indent}  [DIR] {dir_name}"));
                    let sub_output = Self::list_directory_recursive(
                        dir_path,
                        current_depth + 1,
                        max_depth,
                        max_entries,
                        entry_count,
                        include_hidden,
                        dirs_first,
                    )?;
                    if !sub_output.is_empty() {
                        output.push(sub_output.trim_end().to_string());
                    }
                }
                Ok(())
            };

        if dirs_first {
            render_dirs(&mut output, dirs)?;
            render_files(&mut output, &files);
        } else {
            render_files(&mut output, &files);
            render_dirs(&mut output, dirs)?;
        }

        Ok(output.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::{ListDirectoryTool, MAX_OUTPUT_SIZE};
    use crate::agent::tools::Tool;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("list-accuracy")
            .join(name);
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("fixture dir");
        root
    }

    #[test]
    fn hidden_entries_do_not_consume_max_entries() {
        let root = fixture("hidden-quota");
        for idx in 0..8 {
            fs::write(root.join(format!(".secret_{idx}")), "x\n").expect("write hidden");
        }
        let mut entry_count = 0usize;
        let output = ListDirectoryTool::list_directory_simple(
            &root,
            1,
            &mut entry_count,
            false,
            true,
            false,
        )
        .expect("list");
        assert_eq!(entry_count, 0);
        assert!(!output.contains(".secret_"));
    }

    #[test]
    fn visible_file_is_listed_when_hidden_files_are_present() {
        let root = fixture("hidden-with-visible");
        fs::write(root.join(".secret"), "x\n").expect("write hidden");
        fs::write(root.join("visible.rs"), "x\n").expect("write visible");
        let mut entry_count = 0usize;
        let output = ListDirectoryTool::list_directory_simple(
            &root,
            1,
            &mut entry_count,
            false,
            true,
            false,
        )
        .expect("list");
        assert_eq!(entry_count, 1);
        assert!(output.contains("visible.rs"));
        assert!(!output.contains(".secret"));
    }

    #[tokio::test]
    async fn long_listing_is_cut_before_a_multibyte_character() {
        const FILE_LINE: usize = 269; // "  file [0 B] " + 255 name bytes + "\n"
        let root = fixture("multibyte-cut")
            .canonicalize()
            .expect("canonical fixture");
        for idx in 0..160 {
            let last = char::from_u32(0x4E00 + idx).expect("CJK character");
            let name: String = std::iter::repeat_n('日', 84).chain([last]).collect();
            fs::write(root.join(name), "").expect("write file");
        }
        // Pick a directory name length that puts byte MAX_OUTPUT_SIZE one byte into a character.
        let header = format!("Directory: {}\n", root.display()).len();
        let pad = (1..=255)
            .find(|len| {
                let at = (MAX_OUTPUT_SIZE - header - "  dir  [-] ".len() - len - 1) % FILE_LINE;
                (13..268).contains(&at) && (at - 13) % 3 == 1
            })
            .expect("pad length");
        fs::create_dir(root.join("d".repeat(pad))).expect("pad dir");

        let result = ListDirectoryTool
            .execute(json!({ "path": root.display().to_string() }))
            .await;

        assert_eq!(result.error, None);
        let body = result
            .content
            .strip_suffix("\n\n[Output truncated - too large for context window]")
            .expect("truncation notice");
        assert_eq!(body.len(), 39_999);
    }
}

#[cfg(test)]
#[cfg(unix)]
mod link_tests {
    use super::ListDirectoryTool;

    /// A workspace whose `sub` is a link to a directory outside it, as when
    /// `sub` is swapped for a link after the parent listing saw a directory.
    fn swapped_sub() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.txt"), "x").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        let root = ws.path().canonicalize().expect("canonical workspace");
        std::os::unix::fs::symlink(outside.path(), root.join("sub")).expect("dir link");
        let sub = root.join("sub");
        (outside, ws, sub)
    }

    #[test]
    fn a_recursive_listing_refuses_a_directory_swapped_for_a_link() {
        let (_outside, _ws, sub) = swapped_sub();
        let mut entry_count = 0usize;
        let listed = ListDirectoryTool::list_directory_recursive(
            sub,
            1,
            3,
            100,
            &mut entry_count,
            false,
            true,
        );
        let err = listed.expect_err("a link must not be listed");
        assert!(!format!("{err:#}").contains("secret.txt"), "{err:#}");
        assert_eq!(entry_count, 0);
    }

    #[test]
    fn a_listing_refuses_a_directory_swapped_for_a_link() {
        let (_outside, _ws, sub) = swapped_sub();
        let mut entry_count = 0usize;
        let listed = ListDirectoryTool::list_directory_simple(
            &sub,
            100,
            &mut entry_count,
            false,
            true,
            false,
        );
        let err = listed.expect_err("a link must not be listed");
        assert!(!format!("{err:#}").contains("secret.txt"), "{err:#}");
        assert_eq!(entry_count, 0);
    }
}
