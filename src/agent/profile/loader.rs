//! Load agent profiles from Markdown files with YAML frontmatter.

use super::AgentProfile;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Parses a Markdown file with optional YAML frontmatter.
/// Format: optional first line "---", YAML block, "---", then body.
fn parse_frontmatter(content: &str) -> Result<(Option<serde_norway::Value>, String)> {
    let content = content.trim();
    if !content.starts_with("---") {
        return Ok((None, content.to_string()));
    }
    let after_first = content.strip_prefix("---").unwrap_or(content).trim_start();
    let end = after_first
        .find("\n---")
        .or_else(|| after_first.find("\r\n---"));
    let (yaml_block, body) = match end {
        Some(i) => {
            let yaml_str = after_first[..i].trim();
            let body_start = i + 1;
            let rest = after_first[body_start..].trim_start();
            let body_rest = rest.strip_prefix("---").unwrap_or(rest).trim_start();
            (yaml_str, body_rest.to_string())
        }
        None => (after_first, String::new()),
    };
    if yaml_block.is_empty() {
        return Ok((None, body));
    }
    let value: serde_norway::Value =
        serde_norway::from_str(yaml_block).context("Parse YAML frontmatter")?;
    Ok((Some(value), body))
}

/// Load a single profile from a Markdown file path.
pub fn load_profile_file(path: &Path) -> Result<(String, AgentProfile)> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Read profile file: {}", path.display()))?;
    load_profile_from_str(&content, path)
}

/// Load a profile from string content (used for tests and from file).
pub fn load_profile_from_str(content: &str, path: &Path) -> Result<(String, AgentProfile)> {
    let (yaml_opt, body) = parse_frontmatter(content)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let mut profile: AgentProfile = if let Some(v) = yaml_opt {
        serde_norway::from_value(v).context("Deserialize profile frontmatter")?
    } else {
        AgentProfile::default()
    };
    if !body.trim().is_empty() {
        profile.system_prompt_body = Some(body.trim().to_string());
    }
    Ok((name, profile))
}

/// Default agents directory name under config/project.
pub const AGENTS_DIR_NAME: &str = "agents";

/// Build the default search path for agent profiles: project `.caliche/agents`, then `~/.config/caliche/agents`.
pub fn default_search_path(git_root: Option<&PathBuf>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(root) = git_root {
        paths.push(root.join(".caliche").join(AGENTS_DIR_NAME));
    }
    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join("caliche").join(AGENTS_DIR_NAME));
    }
    paths
}

/// Load all profiles from a list of directories. Later directories override earlier for same name.
pub fn load_profiles_from_search_path(
    search_path: &[PathBuf],
) -> Result<HashMap<String, AgentProfile>> {
    let mut registry = HashMap::new();
    for dir in search_path {
        if !dir.exists() {
            continue;
        }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            match load_profile_file(&path) {
                Ok((name, profile)) => {
                    registry.insert(name, profile);
                }
                Err(e) => {
                    tracing::warn!("Skip profile {}: {}", path.display(), e);
                }
            }
        }
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter_empty() {
        let (yaml, body) = parse_frontmatter("hello world").unwrap();
        assert!(yaml.is_none());
        assert_eq!(body, "hello world");
    }

    #[test]
    fn test_parse_frontmatter_with_yaml() {
        let content = r#"---
description: A reviewer
mode: subagent
temperature: 0.1
---
You are a reviewer."#;
        let (yaml, body) = parse_frontmatter(content).unwrap();
        let v = yaml.unwrap();
        assert_eq!(
            v.get("description").and_then(|x| x.as_str()),
            Some("A reviewer")
        );
        assert_eq!(body.trim(), "You are a reviewer.");
    }
}
