//! Shared helpers used across command handlers.

use std::path::Path;

/// Format tool arguments for display in the REPL.
/// Shows meaningful information based on the tool type.
pub fn format_tool_args(tool_name: &str, arguments: &serde_json::Value) -> String {
    let obj = match arguments.as_object() {
        Some(o) => o,
        None => return String::new(),
    };

    match tool_name {
        "bash" | "shell" | "execute_command" => {
            if let Some(cmd) = obj.get("command").and_then(|v| v.as_str()) {
                let preview: String = cmd.chars().take(100).collect();
                let suffix = if cmd.len() > 100 { "..." } else { "" };
                return format!("\x1b[2m$ {}{}\x1b[0m", preview, suffix);
            }
        }
        "fs" => {
            if let Some(path) = obj.get("path").and_then(|v| v.as_str()) {
                let op = obj.get("op").and_then(|v| v.as_str()).unwrap_or("?");
                return format!("\x1b[2m{} {}\x1b[0m", op, path);
            }
        }
        "glob" | "find_files" => {
            if let Some(pattern) = obj
                .get("pattern")
                .or_else(|| obj.get("glob"))
                .and_then(|v| v.as_str())
            {
                return format!("\x1b[2m{}\x1b[0m", pattern);
            }
        }
        "http" | "fetch" | "http_request" => {
            let method = obj.get("method").and_then(|v| v.as_str()).unwrap_or("GET");
            if let Some(url) = obj.get("url").and_then(|v| v.as_str()) {
                let preview: String = url.chars().take(80).collect();
                let suffix = if url.len() > 80 { "..." } else { "" };
                return format!("\x1b[2m{} {}{}\x1b[0m", method, preview, suffix);
            }
        }
        "git" => {
            if let Some(subcmd) = obj
                .get("subcommand")
                .or_else(|| obj.get("command"))
                .and_then(|v| v.as_str())
            {
                return format!("\x1b[2mgit {}\x1b[0m", subcmd);
            }
        }
        _ => {
            if let Some(text) = format_generic_args(obj) {
                return text;
            }
        }
    }

    String::new()
}

fn format_generic_args(obj: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    let previews: Vec<String> = obj
        .iter()
        .take(2)
        .filter_map(|(key, value)| {
            let text = match value {
                serde_json::Value::String(text) => {
                    let preview: String = text.chars().take(30).collect();
                    if text.len() > 30 {
                        format!("{preview}...")
                    } else {
                        preview
                    }
                }
                serde_json::Value::Number(number) => number.to_string(),
                serde_json::Value::Bool(flag) => flag.to_string(),
                _ => return None,
            };
            Some(format!("{key}={text}"))
        })
        .collect();
    if previews.is_empty() {
        None
    } else {
        Some(format!("\x1b[2m{}\x1b[0m", previews.join(", ")))
    }
}

/// Detect project type from directory (lowercase, for config generation).
pub fn detect_project_type(path: &Path) -> &'static str {
    if path.join("Cargo.toml").exists() {
        "rust"
    } else if path.join("package.json").exists() {
        "node"
    } else if path.join("pyproject.toml").exists() || path.join("setup.py").exists() {
        "python"
    } else if path.join("go.mod").exists() {
        "go"
    } else if path.join("Gemfile").exists() {
        "ruby"
    } else if path.join("pom.xml").exists() || path.join("build.gradle").exists() {
        "java"
    } else {
        "generic"
    }
}

/// Detect project type for display (title case).
pub fn detect_project_type_for_display() -> &'static str {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return "Unknown",
    };

    if cwd.join("Cargo.toml").exists() {
        "Rust"
    } else if cwd.join("package.json").exists() {
        if cwd.join("tsconfig.json").exists() {
            "TypeScript"
        } else {
            "JavaScript"
        }
    } else if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
        "Python"
    } else if cwd.join("go.mod").exists() {
        "Go"
    } else if cwd.join("Gemfile").exists() {
        "Ruby"
    } else if cwd.join("pom.xml").exists() {
        "Java (Maven)"
    } else if cwd.join("build.gradle").exists() || cwd.join("build.gradle.kts").exists() {
        "Java (Gradle)"
    } else if cwd.join("Makefile").exists() {
        "Make"
    } else if cwd.join(".git").exists() {
        "Git Repository"
    } else {
        "Generic"
    }
}
