//! Tool to check that recently created or modified files are wired into the project.
//! Surfaces likely integration gaps (exports not imported, components not registered).

use crate::agent::tools::common::{params, path as tool_path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;

const NAME: &str = "verify_integration";
const DESCRIPTION: &str = "Verify that recently created or modified files are integrated (imports, registrations, \
         event handlers). Checks for exports not imported elsewhere, modules not required, and \
         common integration gaps. Call after implementing features to ensure nothing is left unwired.";
#[derive(Debug)]
pub struct VerifyIntegrationTool;

impl ToolMeta for VerifyIntegrationTool {
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
                "paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional paths to verify (e.g. new screens, services). If omitted, scans recently modified files."
                },
                "project_root": {
                    "type": "string",
                    "description": "Project root to search (default: current directory)",
                    "default": "."
                }
            },
            "required": []
        })
    }
}

#[async_trait]
impl Tool for VerifyIntegrationTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let project_root = params::extract_string_or(&arguments, "project_root", ".");
        let root = match std::env::current_dir() {
            Ok(dir) => dir.join(&project_root),
            Err(e) => return ToolResult::error(format!("Failed to get current directory: {e}")),
        };
        // The walks below read files under this canonical root without
        // following a link.
        let root = match root.canonicalize() {
            Ok(root) if root.is_dir() => root,
            _ => {
                return ToolResult::error(format!(
                    "Project root does not exist or is not a directory: {project_root}"
                ));
            }
        };

        let paths: Vec<String> = arguments
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let report = run_checks(&root, &paths).await;
        ToolResult::success(report)
    }
}

async fn run_checks(root: &std::path::Path, paths: &[String]) -> String {
    let mut out = String::new();

    let to_check: Vec<PathBuf> = if paths.is_empty() {
        list_recent_files(root)
    } else {
        paths
            .iter()
            .map(|p| {
                if PathBuf::from(p).is_absolute() {
                    PathBuf::from(p)
                } else {
                    root.join(p)
                }
            })
            .filter_map(|p| p.canonicalize().ok())
            .collect()
    };

    if to_check.is_empty() {
        return "No paths to verify. Provide 'paths' or ensure project has recent modified files."
            .to_string();
    }

    out.push_str("Integration check:\n");
    for path in &to_check {
        let rel = path.strip_prefix(root).unwrap_or(path);
        let content = match read_checked_file(root, path).await {
            Ok(c) => c,
            Err(_) => continue,
        };
        let issues = check_file(root, path, &content);
        if issues.is_empty() {
            out.push_str(&format!("  [OK] {}\n", rel.display()));
        } else {
            out.push_str(&format!("  [CHECK] {}\n", rel.display()));
            for i in &issues {
                out.push_str(&format!("    - {i}\n"));
            }
        }
    }
    out.push_str(
        "\nResolve any [CHECK] items: add imports, registrations, or event handlers as needed.",
    );
    out
}

/// Read `path`, canonical, without following a link: below `root` when it
/// is there, and from the file system root for a path the gate let through
/// from outside it.
async fn read_checked_file(
    root: &std::path::Path,
    path: &std::path::Path,
) -> std::io::Result<String> {
    let (root, path) = (root.to_path_buf(), path.to_path_buf());
    tokio::task::spawn_blocking(move || {
        if path.starts_with(&root) {
            tool_path::read_walked_text(&root, &path)
        } else {
            tool_path::read_resolved_text(&path)
        }
    })
    .await
    .map_err(std::io::Error::other)?
}

fn list_recent_files(root: &std::path::Path) -> Vec<PathBuf> {
    const MAX_FILES: usize = 20;
    const MAX_CANDIDATES: usize = 10_000;
    const IGNORED_DIRS: &[&str] = &[
        "target",
        "node_modules",
        "dist",
        "build",
        "out",
        ".git",
        "__pycache__",
    ];
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    fn walk(
        dir: &std::path::Path,
        out: &mut Vec<(std::time::SystemTime, PathBuf)>,
        visited: &mut usize,
    ) {
        if *visited >= MAX_CANDIDATES {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            *visited += 1;
            if *visited >= MAX_CANDIDATES {
                return;
            }
            let p = e.path();
            let Some(file_name) = p.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if file_name.starts_with('.') || IGNORED_DIRS.contains(&file_name) {
                continue;
            }
            // The entry's own type: a link is neither, so it is skipped.
            let Ok(kind) = e.file_type() else {
                continue;
            };
            if kind.is_dir() {
                walk(&p, out, visited);
                continue;
            }
            if !kind.is_file() {
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "lua" | "ts" | "tsx" | "js" | "jsx" | "rs" | "py")
                && let Ok(m) = e.metadata().and_then(|m| m.modified())
            {
                out.push((m, p));
            }
        }
    }
    let mut visited = 0usize;
    walk(root, &mut candidates, &mut visited);
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    candidates.truncate(MAX_FILES);
    candidates.into_iter().map(|(_, p)| p).collect()
}

fn check_file(root: &std::path::Path, path: &std::path::Path, content: &str) -> Vec<String> {
    let mut issues = Vec::new();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "lua" => {
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if (content.contains("return {") || content.contains("return ")) && !name.is_empty() {
                let needle = name.replace('_', ".");
                let count = search_count_in_project(root, &needle, Some(path));
                if count <= 1 {
                    issues.push(format!(
                        "Module may not be required: '{name}' not found in other files"
                    ));
                }
            }
        }
        "ts" | "tsx" | "js" | "jsx" => {
            if content.contains("export ") {
                let snippet = content
                    .lines()
                    .find(|l| l.contains("export "))
                    .unwrap_or_default();
                let sym = extract_export_symbol(snippet);
                if let Some(s) = sym {
                    let count = search_count_in_project(root, &s, Some(path));
                    if count <= 1 {
                        issues.push(format!("Export '{s}' may not be imported elsewhere"));
                    }
                }
            }
        }
        "rs" => {
            for symbol in extract_rust_public_items(content) {
                let count = search_count_in_project(root, &symbol, Some(path));
                if count <= 1 {
                    issues.push(format!(
                        "pub item '{symbol}' may not be referenced elsewhere (missing `pub use` or `mod` declaration?)"
                    ));
                }
            }

            if let Some(module) = path.file_stem().and_then(|s| s.to_str())
                && module != "mod"
                && module != "lib"
                && module != "main"
            {
                let declaration_count =
                    search_count_in_project(root, &format!("mod {module}"), Some(path));
                if declaration_count == 0 {
                    issues.push(format!(
                        "module file '{module}.rs' is not declared with `mod {module};` anywhere"
                    ));
                }
            }
        }
        _ => {}
    }
    issues
}

/// Extract a small set of `pub` item names (functions, structs, enums, traits,
/// and type aliases) from a Rust source file for integration cross-referencing.
fn extract_rust_public_items(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("pub ") else {
            continue;
        };
        let rest = rest
            .strip_prefix("(crate) ")
            .or_else(|| rest.strip_prefix("(super) "))
            .unwrap_or(rest);
        let keyword_name = [
            ("fn ", [' ', '(', '<'].as_slice()),
            ("struct ", &[' ', '(', '<', '{', ';']),
            ("enum ", &[' ', '<', '{']),
            ("trait ", &[' ', '<', ':', '{']),
            ("type ", &[' ', '<', '=']),
            ("const ", &[' ', ':', '=']),
            ("static ", &[' ', ':', '=']),
        ];
        for (kw, delims) in keyword_name {
            if let Some(after) = rest.strip_prefix(kw) {
                let name: String = after
                    .trim_start()
                    .chars()
                    .take_while(|c| !delims.contains(c))
                    .collect();
                if !name.is_empty() {
                    out.push(name);
                }
                break;
            }
        }
        if out.len() >= 3 {
            break;
        }
    }
    out
}

fn extract_export_symbol(line: &str) -> Option<String> {
    let line = line.trim();
    if let Some(r) = line.strip_prefix("export ") {
        let r = r.trim();
        if let Some(name) = r
            .strip_prefix("const ")
            .or_else(|| r.strip_prefix("function "))
        {
            return name.split([' ', '=', '(']).next().map(String::from);
        }
        if let Some(name) = r.strip_prefix("default ") {
            return name.split([' ', ';']).next().map(String::from);
        }
        return r.split([' ', ';', '{']).next().map(String::from);
    }
    None
}

/// Count the files under the canonical `root` that contain `needle`. The walk
/// follows no link, and each file is opened without following one.
fn search_count_in_project(
    root: &std::path::Path,
    needle: &str,
    exclude: Option<&std::path::Path>,
) -> usize {
    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        needle: &str,
        exclude: Option<&std::path::Path>,
        count: &mut usize,
    ) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.file_name()
                .is_none_or(|n| n.to_str().is_none_or(|s| s.starts_with('.')))
            {
                continue;
            }
            let Ok(kind) = e.file_type() else {
                continue;
            };
            if kind.is_dir() {
                walk(root, &p, needle, exclude, count);
            } else if kind.is_file()
                && p.extension().is_some_and(|e| {
                    matches!(e.to_str(), Some("lua" | "ts" | "tsx" | "js" | "jsx" | "rs"))
                })
            {
                if exclude.is_some_and(|ex| p == ex) {
                    continue;
                }
                if let Ok(c) = tool_path::read_walked_text(root, &p)
                    && c.contains(needle)
                {
                    *count += 1;
                }
            }
        }
    }
    let mut n = 0;
    walk(root, root, needle, exclude, &mut n);
    n
}

#[cfg(test)]
#[cfg(unix)]
mod link_tests {
    use super::{run_checks, search_count_in_project};

    /// A canonical workspace with `ok.rs`, a file link `leak.rs`, and a
    /// directory link `linked`, both to files outside that hold `needle`.
    fn workspace() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.rs"), "fn needle() {}\n").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        let root = ws.path().canonicalize().expect("canonical workspace");
        std::fs::write(root.join("ok.rs"), "fn needle() {}\n").expect("ok");
        std::os::unix::fs::symlink(outside.path().join("secret.rs"), root.join("leak.rs"))
            .expect("file link");
        std::os::unix::fs::symlink(outside.path(), root.join("linked")).expect("dir link");
        (outside, ws, root)
    }

    #[test]
    fn a_project_search_counts_no_file_behind_a_link() {
        let (_outside, _ws, root) = workspace();

        assert_eq!(search_count_in_project(&root, "needle", None), 1);
    }

    #[tokio::test]
    async fn recent_files_checked_are_not_behind_a_link() {
        let (_outside, _ws, root) = workspace();

        let report = run_checks(&root, &[]).await;

        let checked: Vec<&str> = report
            .lines()
            .filter(|line| line.starts_with("  [OK] ") || line.starts_with("  [CHECK] "))
            .collect();
        assert_eq!(checked.len(), 1, "{report}");
        assert!(checked[0].ends_with(" ok.rs"), "{report}");
    }
}
