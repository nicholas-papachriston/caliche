use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use caliche_contracts::text::truncate_utf8;
use serde_json::json;
use std::path::{Path, PathBuf};

#[cfg(feature = "tools-grep")]
use grep_matcher::Matcher;
#[cfg(feature = "tools-grep")]
use grep_regex::RegexMatcher;
#[cfg(feature = "tools-grep")]
use ignore::WalkBuilder;

const NAME: &str = "workspace";
const DESCRIPTION: &str = "Detect and analyze workspace/project structure. Use this (not bash find/ls) to understand a codebase. \
        Actions: detect (project type), structure (top-level dirs/files), config (Cargo.toml, package.json, etc.), \
        entry_points (mains, binaries), members (authoritative list of Cargo workspace members + git submodules from Cargo.toml/.gitmodules), \
        codebase_context (one-shot: project type + AGENTS.md/README + structure + members + ranked file list). \
        Prefer workspace over fs op=list when you need project type or config; use codebase_context for quick orientation; use members when you need an exact crate count.";
/// The text of `name` in the canonical workspace `base`, read without
/// following any link; `None` when it is missing, a link, or unreadable.
fn read_text(base: &Path, name: &str) -> Option<String> {
    crate::discovery::auto::read_workspace_text(base, &base.join(name))
}

fn append_labeled_names(output: &mut String, title: &str, names: &[String], suffix: &str) {
    if names.is_empty() {
        return;
    }
    output.push_str(title);
    for name in names {
        output.push_str("  ");
        output.push_str(name);
        output.push_str(suffix);
        output.push('\n');
    }
    if suffix == "/" {
        output.push('\n');
    }
}

#[cfg(feature = "tools-grep")]
fn append_top_level_with_walker(output: &mut String, path: &Path) {
    let mut walker = WalkBuilder::new(path);
    walker.hidden(false);
    walker.git_ignore(true);
    walker.git_exclude(true);
    walker.ignore(true);
    walker.max_depth(Some(1));

    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for result in walker.build() {
        let entry = match result {
            Ok(entry) => entry,
            Err(err) => {
                tracing::debug!("Error walking directory: {err}");
                continue;
            }
        };
        let entry_path = entry.path();
        let name = entry_path
            .file_name()
            .and_then(|part| part.to_str())
            .unwrap_or("")
            .to_string();
        if name.starts_with('.') && name != ".git" {
            continue;
        }
        if entry_path.is_dir() {
            dirs.push(name);
        } else {
            files.push(name);
        }
    }
    dirs.sort();
    files.sort();
    append_labeled_names(output, "Directories:\n", &dirs, "/");
    append_labeled_names(output, "Files:\n", &files, "");
}

#[cfg(not(feature = "tools-grep"))]
fn append_top_level_with_read_dir(output: &mut String, path: &Path) -> std::io::Result<()> {
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(path)?.flatten() {
        let entry_path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if entry_path.is_dir() {
            dirs.push(name);
        } else {
            files.push(name);
        }
    }
    dirs.sort();
    files.sort();
    append_labeled_names(output, "Directories:\n", &dirs, "/");
    append_labeled_names(output, "Files:\n", &files, "");
    Ok(())
}

#[derive(Debug)]
pub struct WorkspaceTool;

impl ToolMeta for WorkspaceTool {
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
                    "description": "Path to the workspace/project root (default: current directory)",
                    "default": "."
                },
                "action": {
                    "type": "string",
                    "enum": ["detect", "structure", "config", "entry_points", "members", "codebase_context"],
                    "description": "Action: detect, structure, config, entry_points, members (authoritative Cargo workspace members + git submodules), or codebase_context (project type + context docs + structure + members + ranked file list)",
                    "default": "detect"
                }
            },
            "required": []
        })
    }
}

#[async_trait]
impl Tool for WorkspaceTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let action = params::extract_string_or(&arguments, "action", "detect");

        let workspace_path = if PathBuf::from(&path_str).is_absolute() {
            PathBuf::from(&path_str)
        } else {
            match std::env::current_dir() {
                Ok(dir) => dir.join(&path_str),
                Err(e) => {
                    return ToolResult::error(format!("Failed to get current directory: {e}"));
                }
            }
        };

        // Canonical, so the files below are read without following any link
        // under it: a cloned repository can plant `README.md` as a link to an
        // outside file, and its text would reach the model.
        let Ok(canonical) = workspace_path.canonicalize() else {
            return ToolResult::error(format!("Path does not exist: {path_str}"));
        };
        let workspace_path = caliche_sandbox::external_tool_path(&canonical);

        match action.as_str() {
            "detect" => Self::detect_project_type(&workspace_path),
            "structure" => Self::list_structure(&workspace_path).await,
            "config" => Self::read_configs(&workspace_path),
            "entry_points" => Self::find_entry_points(&workspace_path).await,
            "members" => Self::members(&workspace_path),
            "codebase_context" => Self::codebase_context(&workspace_path).await,
            _ => ToolResult::error(format!("Unknown action: {action}")),
        }
    }
}

impl WorkspaceTool {
    fn detect_project_type(workspace_path: &Path) -> ToolResult {
        let mut project_types = Vec::new();

        // Check for Rust project
        if workspace_path.join("Cargo.toml").exists() {
            project_types.push("Rust (Cargo)");
        }

        // Check for Node.js project
        if workspace_path.join("package.json").exists() {
            project_types.push("Node.js");
        }

        // Check for Python project
        if workspace_path.join("pyproject.toml").exists()
            || workspace_path.join("setup.py").exists()
            || workspace_path.join("requirements.txt").exists()
        {
            project_types.push("Python");
        }

        // Check for Go project
        if workspace_path.join("go.mod").exists() {
            project_types.push("Go");
        }

        // Check for Java project
        if workspace_path.join("pom.xml").exists() || workspace_path.join("build.gradle").exists() {
            project_types.push("Java");
        }

        if project_types.is_empty() {
            ToolResult::success("No recognized project type detected")
        } else {
            ToolResult::success(format!(
                "Detected project types: {}",
                project_types.join(", ")
            ))
        }
    }

    async fn list_structure(workspace_path: &Path) -> ToolResult {
        let path_buf = workspace_path.to_path_buf();
        let output = tokio::task::spawn_blocking(move || {
            let mut output = String::from("Project Structure:\n");
            output.push_str("=================\n\n");
            #[cfg(feature = "tools-grep")]
            append_top_level_with_walker(&mut output, &path_buf);
            #[cfg(not(feature = "tools-grep"))]
            append_top_level_with_read_dir(&mut output, &path_buf)?;
            Ok::<_, std::io::Error>(output)
        })
        .await;

        match output {
            Ok(Ok(s)) => ToolResult::success(s),
            Ok(Err(e)) => ToolResult::error(format!("Failed to read directory: {e}")),
            Err(e) => ToolResult::error(format!("Task join error: {e}")),
        }
    }

    fn read_configs(workspace_path: &Path) -> ToolResult {
        let mut output = String::from("Configuration Files:\n");
        output.push_str("===================\n\n");

        let config_files = vec![
            ("Cargo.toml", "Rust"),
            ("package.json", "Node.js"),
            ("pyproject.toml", "Python"),
            ("go.mod", "Go"),
            ("pom.xml", "Java"),
            ("build.gradle", "Java"),
            (".gitignore", "Git"),
            ("README.md", "Documentation"),
        ];

        for (filename, project_type) in config_files {
            let config_path = workspace_path.join(filename);
            if config_path.exists() {
                output.push_str(&format!("{filename} ({project_type})\n"));
                if let Some(content) = read_text(workspace_path, filename) {
                    // Limit content to first 500 chars
                    let preview = if content.len() > 500 {
                        format!("{}...", truncate_utf8(&content, 500))
                    } else {
                        content
                    };
                    output.push_str(&format!("  {}\n\n", preview.replace('\n', "\n  ")));
                }
            }
        }

        ToolResult::success(output)
    }

    /// One-shot orientation: project type + context docs (AGENTS.md/README) +
    /// top-level structure + Cargo workspace members + git submodules + `RepoMap`.
    async fn codebase_context(workspace_path: &Path) -> ToolResult {
        let mut out = String::new();
        let detect = Self::detect_project_type(workspace_path);
        if detect.success {
            out.push_str("## Project type\n");
            out.push_str(&detect.content);
            out.push_str("\n\n");
        }

        // Note: Project context docs (AGENTS.md, README.md) are typically
        // pre-injected into the system prompt by the CLI. Include a short note instead
        // of the full content to avoid duplication. Mark them so the agent treats their
        // contents as documentation rather than as something it discovered.
        for name in &["AGENTS.md", "README.md"] {
            let p = workspace_path.join(name);
            if p.exists() {
                out.push_str(&format!(
                    "## {name}\nProject context from {name} is available in the system prompt. Treat its contents as DOCUMENTATION (not as findings you discovered) and label them as such when restating them.\n\n",
                ));
                break;
            }
        }

        // Authoritative workspace member + submodule enumeration. This is the fact the
        // agent most commonly gets wrong (e.g., conflating git submodules with Cargo
        // workspace members), so surface it in every codebase_context call.
        let members_section = Self::collect_members(workspace_path);
        if !members_section.is_empty() {
            out.push_str("## Workspace members (authoritative)\n");
            out.push_str(&members_section);
            out.push('\n');
        }

        let structure = Self::list_structure(workspace_path).await;
        if structure.success {
            out.push_str("## Top-level structure\n");
            out.push_str(&structure.content);
        }

        #[cfg(feature = "tools-grep")]
        {
            let map_str = crate::repomap::build(workspace_path, 2000);
            if !map_str.is_empty() {
                out.push_str("\n\n");
                out.push_str(&map_str);
            }
        }

        ToolResult::success(out)
    }

    /// Standalone `members` action: returns the authoritative list of Cargo workspace
    /// members (from `Cargo.toml`'s `[workspace].members`) and git submodules
    /// (from `.gitmodules`), with explicit counts and a clear distinction between the
    /// two so the agent stops conflating them.
    fn members(workspace_path: &Path) -> ToolResult {
        let body = Self::collect_members(workspace_path);
        if body.is_empty() {
            ToolResult::success(
                "No Cargo workspace ([workspace].members) and no .gitmodules found at this path."
                    .to_string(),
            )
        } else {
            ToolResult::success(body)
        }
    }

    /// Build the workspace-members section. Best-effort: if no Cargo workspace
    /// and no .gitmodules exist, returns an empty string so callers can omit
    /// the section entirely.
    fn collect_members(workspace_path: &Path) -> String {
        let cargo_members = Self::read_cargo_workspace_members(workspace_path);
        let submodules = Self::read_git_submodules(workspace_path);

        if cargo_members.is_none() && submodules.is_empty() {
            return String::new();
        }

        let mut out = String::new();

        match cargo_members {
            Some(members) => {
                out.push_str(&format!(
                    "Cargo workspace members ({} entries, from `Cargo.toml` `[workspace].members`):\n",
                    members.len()
                ));
                if members.is_empty() {
                    out.push_str("  (none listed)\n");
                } else {
                    for m in &members {
                        out.push_str(&format!("  - {m}\n"));
                    }
                }
            }
            None => {
                out.push_str(
                    "Cargo workspace: no `[workspace]` table found in this directory's `Cargo.toml` (or no Cargo.toml).\n",
                );
            }
        }

        if !submodules.is_empty() {
            out.push_str(&format!(
                "\nGit submodules ({} entries, from `.gitmodules` -- NOT the same as Cargo workspace members):\n",
                submodules.len()
            ));
            for (name, path) in &submodules {
                out.push_str(&format!("  - {name} at {path}\n"));
            }
        }

        out.push_str("\nNOTE: Cargo workspace members and git submodules are independent sets. Do not add their counts together when answering 'how many crates are in this workspace?'.\n");
        out
    }

    /// Parse `[workspace].members` from `Cargo.toml`. Returns `None` if the file
    /// is missing or has no `[workspace]` table; returns an empty vec if the
    /// table exists but lists no members.
    fn read_cargo_workspace_members(workspace_path: &Path) -> Option<Vec<String>> {
        let raw = read_text(workspace_path, "Cargo.toml")?;
        let parsed: toml::Value = raw.parse().ok()?;
        let workspace = parsed.get("workspace")?;
        let members = workspace
            .get("members")
            .and_then(toml::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Some(members)
    }

    /// Parse `.gitmodules` for `[submodule "name"] path = "..."` entries. Returns
    /// `(name, path)` pairs in declaration order. Empty when the file is missing
    /// or has no safe submodule paths.
    fn read_git_submodules(workspace_path: &Path) -> Vec<(String, String)> {
        let Some(raw) = read_text(workspace_path, ".gitmodules") else {
            return Vec::new();
        };
        crate::gitmodules::parse_gitmodules(&raw)
            .into_iter()
            .map(|entry| (entry.name, entry.relative_path))
            .collect()
    }

    async fn find_entry_points(workspace_path: &Path) -> ToolResult {
        let mut entry_points: Vec<String> = Vec::new();

        // Check config files first (always done)
        // Rust entry points from Cargo.toml
        if let Some(cargo_content) = read_text(workspace_path, "Cargo.toml") {
            if cargo_content.contains("[[bin]]") {
                entry_points.push("Rust binary (see Cargo.toml [[bin]] section)".to_string());
            }
            if cargo_content.contains("[lib]") {
                entry_points.push("Rust library (see Cargo.toml [lib] section)".to_string());
            }
        }

        // Node.js entry points from package.json
        if let Some(package_content) = read_text(workspace_path, "package.json")
            && let Ok(package_json) = serde_json::from_str::<serde_json::Value>(&package_content)
        {
            if let Some(main) = package_json.get("main").and_then(|v| v.as_str()) {
                entry_points.push(format!("Node.js main: {main}"));
            }
            if let Some(bin) = package_json.get("bin") {
                entry_points.push(format!("Node.js bin: {bin:?}"));
            }
        }

        // Python entry points from common filenames
        if workspace_path.join("main.py").exists() {
            entry_points.push("Python: main.py".to_string());
        }
        if workspace_path.join("__main__.py").exists() {
            entry_points.push("Python: __main__.py".to_string());
        }

        // Use ripgrep to search for actual entry point patterns in code files
        #[cfg(feature = "tools-grep")]
        {
            let code_entry_points = Self::find_code_entry_points(workspace_path).await;
            entry_points.extend(code_entry_points);
        }

        if entry_points.is_empty() {
            ToolResult::success("No entry points detected")
        } else {
            let mut output = String::from("Entry Points:\n");
            output.push_str("=============\n\n");
            for ep in &entry_points {
                output.push_str(&format!("  - {ep}\n"));
            }
            ToolResult::success(output)
        }
    }

    #[cfg(feature = "tools-grep")]
    async fn find_code_entry_points(workspace_path: &Path) -> Vec<String> {
        let path_buf = workspace_path.to_path_buf();

        tokio::task::spawn_blocking(move || {
            let mut entry_points = Vec::new();

            let patterns = vec![
                ("Rust", r"fn\s+main\s*\(", ".rs"),
                ("Python", r#"if\s+__name__\s*==\s*['"]__main__['"]"#, ".py"),
                (
                    "JavaScript/TypeScript",
                    r"if\s+\(require\.main\s*===\s*module\)",
                    ".js",
                ),
                (
                    "TypeScript",
                    r"if\s+\(require\.main\s*===\s*module\)",
                    ".ts",
                ),
                ("Go", r"func\s+main\s*\(", ".go"),
                ("Java", r"public\s+static\s+void\s+main\s*\(", ".java"),
                ("C/C++", r"int\s+main\s*\(", ".c"),
                ("C/C++", r"int\s+main\s*\(", ".cpp"),
            ];

            for (lang, pattern, ext) in patterns {
                if let Ok(matcher) = RegexMatcher::new(pattern) {
                    let mut walker = WalkBuilder::new(&path_buf);
                    walker.hidden(false);
                    walker.git_ignore(true);
                    walker.git_exclude(true);
                    walker.ignore(true);
                    walker.max_depth(Some(10));

                    let ext_owned = ext.to_string();
                    walker.filter_entry(move |entry| {
                        entry
                            .path()
                            .to_str()
                            .is_some_and(|p| p.ends_with(&ext_owned))
                    });

                    let mut found_files = std::collections::HashSet::new();
                    let max_results = 10;

                    for result in walker.build() {
                        if found_files.len() >= max_results {
                            break;
                        }

                        if let Ok(entry) = result {
                            let path = entry.path();
                            if path.is_file() {
                                if let Ok(metadata) = path.metadata()
                                    && metadata.len() > 10_485_760
                                {
                                    continue;
                                }

                                if let Some(content) =
                                    crate::discovery::auto::read_workspace_text(&path_buf, path)
                                {
                                    let lines: Vec<&str> = content.lines().collect();
                                    for (idx, line) in lines.iter().enumerate() {
                                        if matcher.find(line.as_bytes()).ok().flatten().is_some() {
                                            let rel_path = path
                                                .strip_prefix(&path_buf)
                                                .unwrap_or(path)
                                                .display()
                                                .to_string();
                                            found_files.insert(format!(
                                                "{}: {} (line {})",
                                                lang,
                                                rel_path,
                                                idx + 1
                                            ));
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    entry_points.extend(found_files.into_iter());
                }
            }

            entry_points
        })
        .await
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// The canonical path of `dir`, as `execute` passes the workspace.
    fn canonical(dir: &tempfile::TempDir) -> PathBuf {
        dir.path()
            .canonicalize()
            .unwrap_or_else(|_| dir.path().to_path_buf())
    }

    #[test]
    fn read_cargo_workspace_members_parses_named_entries() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join("Cargo.toml"),
            r#"[workspace]
members = [
    "crate_a",
    "crate_b",
    "nested/crate_c",
    ".",
]
"#,
        )
        .expect("write Cargo.toml");

        let members = WorkspaceTool::read_cargo_workspace_members(&canonical(&dir))
            .expect("workspace table present");
        assert_eq!(
            members,
            vec![
                "crate_a".to_string(),
                "crate_b".to_string(),
                "nested/crate_c".to_string(),
                ".".to_string(),
            ]
        );
    }

    #[test]
    fn read_cargo_workspace_members_returns_none_without_workspace_table() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join("Cargo.toml"),
            r#"[package]
name = "lonely"
version = "0.1.0"
edition = "2021"
"#,
        )
        .expect("write Cargo.toml");

        assert!(WorkspaceTool::read_cargo_workspace_members(&canonical(&dir)).is_none());
    }

    #[test]
    fn read_cargo_workspace_members_returns_none_when_cargo_toml_missing() {
        let dir = tempdir().expect("tempdir");
        assert!(WorkspaceTool::read_cargo_workspace_members(&canonical(&dir)).is_none());
    }

    #[test]
    fn read_git_submodules_parses_multiple_entries() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join(".gitmodules"),
            r#"[submodule "vendor/foo"]
	path = vendor/foo
	url = git@example.com:org/foo.git
[submodule "vendor/bar"]
	path = vendor/bar
	url = git@example.com:org/bar.git
	branch = main
"#,
        )
        .expect("write .gitmodules");

        let subs = WorkspaceTool::read_git_submodules(&canonical(&dir));
        assert_eq!(
            subs,
            vec![
                ("vendor/foo".to_string(), "vendor/foo".to_string()),
                ("vendor/bar".to_string(), "vendor/bar".to_string()),
            ]
        );
    }

    #[test]
    fn read_git_submodules_keeps_quotes_and_drops_parent_paths() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"my child\"]\n\tpath = \"my child\"\n[submodule \"up\"]\n\tpath = ../outside\n# path = ignored\n",
        )
        .expect("write .gitmodules");

        let subs = WorkspaceTool::read_git_submodules(&canonical(&dir));
        assert_eq!(subs, vec![("my child".to_string(), "my child".to_string())]);
    }

    #[test]
    fn read_git_submodules_returns_empty_when_file_missing() {
        let dir = tempdir().expect("tempdir");
        assert!(WorkspaceTool::read_git_submodules(&canonical(&dir)).is_empty());
    }

    #[test]
    fn collect_members_distinguishes_workspace_from_submodules() {
        let dir = tempdir().expect("tempdir");
        fs::write(
            dir.path().join("Cargo.toml"),
            r#"[workspace]
members = ["crate_a", "crate_b"]
"#,
        )
        .expect("write Cargo.toml");
        fs::write(
            dir.path().join(".gitmodules"),
            r#"[submodule "external"]
	path = external
	url = git@example.com:org/external.git
"#,
        )
        .expect("write .gitmodules");

        let body = WorkspaceTool::collect_members(&canonical(&dir));
        assert!(body.contains("Cargo workspace members (2 entries"));
        assert!(body.contains("crate_a"));
        assert!(body.contains("Git submodules (1 entries"));
        assert!(body.contains("external at external"));
        assert!(body.contains("Do not add their counts together"));
    }

    #[test]
    fn collect_members_empty_when_neither_present() {
        let dir = tempdir().expect("tempdir");
        assert!(WorkspaceTool::collect_members(&canonical(&dir)).is_empty());
    }

    #[test]
    fn members_action_returns_friendly_message_when_neither_present() {
        let dir = tempdir().expect("tempdir");
        let result = WorkspaceTool::members(&canonical(&dir));
        assert!(result.success);
        assert!(result.content.contains("No Cargo workspace"));
    }
}

#[cfg(all(test, unix))]
mod link_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    const OUTSIDE: &str = "OUTSIDE-SECRET";

    /// A workspace and an outside folder holding `name` with outside text.
    fn workspace_and_outside(
        name: &str,
        text: &str,
    ) -> anyhow::Result<(tempfile::TempDir, PathBuf, tempfile::TempDir, PathBuf)> {
        let ws_dir = tempfile::tempdir()?;
        let ws = ws_dir.path().canonicalize()?;
        let out_dir = tempfile::tempdir()?;
        let outside = out_dir.path().canonicalize()?.join(name);
        std::fs::write(&outside, text)?;
        Ok((ws_dir, ws, out_dir, outside))
    }

    async fn run(ws: &Path, action: &str) -> ToolResult {
        WorkspaceTool
            .execute(json!({ "path": ws.to_string_lossy(), "action": action }))
            .await
    }

    #[tokio::test]
    async fn config_does_not_return_a_linked_file() -> anyhow::Result<()> {
        let (_ws_dir, ws, _out_dir, outside) = workspace_and_outside("secret.md", OUTSIDE)?;
        std::fs::write(ws.join("Cargo.toml"), "[package]\nname = \"demo\"\n")?;
        symlink(&outside, ws.join("README.md"))?;

        let result = run(&ws, "config").await;

        assert!(result.success);
        assert_eq!(
            result.content,
            "Configuration Files:\n===================\n\n\
             Cargo.toml (Rust)\n  [package]\n  name = \"demo\"\n  \n\n\
             README.md (Documentation)\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn members_does_not_read_linked_manifests() -> anyhow::Result<()> {
        let (_ws_dir, ws, _out_dir, outside) = workspace_and_outside(
            "Cargo.toml",
            &format!("[workspace]\nmembers = [\"{OUTSIDE}\"]\n"),
        )?;
        symlink(&outside, ws.join("Cargo.toml"))?;
        let gitmodules = outside.with_file_name("gitmodules");
        std::fs::write(
            &gitmodules,
            format!("[submodule \"{OUTSIDE}\"]\n\tpath = {OUTSIDE}\n"),
        )?;
        symlink(&gitmodules, ws.join(".gitmodules"))?;

        let result = run(&ws, "members").await;

        assert!(result.success);
        assert_eq!(
            result.content,
            "No Cargo workspace ([workspace].members) and no .gitmodules found at this path."
        );
        Ok(())
    }

    #[tokio::test]
    async fn entry_points_do_not_read_a_linked_manifest() -> anyhow::Result<()> {
        let (_ws_dir, ws, _out_dir, outside) =
            workspace_and_outside("package.json", r#"{"main": "OUTSIDE-SECRET.js"}"#)?;
        symlink(&outside, ws.join("package.json"))?;

        let result = run(&ws, "entry_points").await;

        assert!(result.success);
        assert_eq!(result.content, "No entry points detected");
        Ok(())
    }

    /// A source file that is a link to a file outside the workspace is not
    /// read for an entry point; a plain one is.
    #[cfg(feature = "tools-grep")]
    #[tokio::test]
    async fn entry_points_do_not_read_a_linked_source_file() -> anyhow::Result<()> {
        let (_ws_dir, ws, _out_dir, outside) = workspace_and_outside("main.rs", "fn main() {}\n")?;
        symlink(&outside, ws.join("leak.rs"))?;
        let plain_dir = tempfile::tempdir()?;
        let plain = plain_dir.path().canonicalize()?;
        std::fs::write(plain.join("main.rs"), "fn main() {}\n")?;

        let linked = run(&ws, "entry_points").await;
        let own = run(&plain, "entry_points").await;

        assert_eq!(linked.content, "No entry points detected");
        assert_eq!(
            own.content,
            "Entry Points:\n=============\n\n  - Rust: main.rs (line 1)\n"
        );
        Ok(())
    }
}
