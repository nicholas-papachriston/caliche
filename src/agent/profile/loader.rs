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

/// A directory of agent profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileDir {
    /// The directory.
    pub path: PathBuf,
    /// The canonical workspace base the directory and its files must stay
    /// below, read without following any link, or `None` for a directory the
    /// user owns, whose links are followed.
    pub confine: Option<PathBuf>,
}

impl ProfileDir {
    /// A directory the user owns, such as one named in their config.
    #[must_use]
    pub fn user(path: PathBuf) -> Self {
        Self {
            path,
            confine: None,
        }
    }
}

/// Build the default search path for agent profiles: project `.caliche/agents`
/// (read without following links, since a cloned repository can plant one
/// to an outside file), then `~/.config/caliche/agents`.
pub fn default_search_path(git_root: Option<&PathBuf>) -> Vec<ProfileDir> {
    let mut paths = Vec::new();
    if let Some(root) = git_root.and_then(|root| root.canonicalize().ok()) {
        let root = caliche_sandbox::external_tool_path(&root);
        paths.push(ProfileDir {
            path: root.join(".caliche").join(AGENTS_DIR_NAME),
            confine: Some(root),
        });
    }
    if let Some(config_dir) = dirs::config_dir() {
        paths.push(ProfileDir::user(
            config_dir.join("caliche").join(AGENTS_DIR_NAME),
        ));
    }
    paths
}

/// The `.md` files in `dir` and their text, skipping any that cannot be read.
fn profile_files(dir: &ProfileDir) -> Vec<(PathBuf, String)> {
    let is_md = |path: &Path| path.extension().and_then(|e| e.to_str()) == Some("md");
    let Some(base) = &dir.confine else {
        let Ok(entries) = std::fs::read_dir(&dir.path) else {
            return Vec::new();
        };
        return entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| is_md(path))
            .filter_map(|path| match std::fs::read_to_string(&path) {
                Ok(text) => Some((path, text)),
                Err(e) => {
                    tracing::warn!("Skip profile {}: {}", path.display(), e);
                    None
                }
            })
            .collect();
    };
    #[cfg(any(unix, windows))]
    {
        let Ok(entries) = caliche_sandbox::list_dir_beneath(base, &dir.path) else {
            return Vec::new();
        };
        entries
            .into_iter()
            .filter(|entry| entry.kind == caliche_sandbox::EntryKind::File)
            .map(|entry| dir.path.join(entry.name))
            .filter(|path| is_md(path))
            .filter_map(|path| {
                crate::discovery::auto::read_workspace_text(base, &path).map(|text| (path, text))
            })
            .collect()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = base;
        Vec::new()
    }
}

/// Load all profiles from a list of directories. Later directories override earlier for same name.
pub fn load_profiles_from_search_path(
    search_path: &[ProfileDir],
) -> Result<HashMap<String, AgentProfile>> {
    let mut registry = HashMap::new();
    for dir in search_path {
        for (path, content) in profile_files(dir) {
            match load_profile_from_str(&content, &path) {
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

#[cfg(all(test, unix))]
mod project_dir_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    /// A repository with `.caliche/agents/review.md`, and an outside file.
    fn repo_and_outside() -> Result<(tempfile::TempDir, PathBuf, tempfile::TempDir, PathBuf)> {
        let repo_dir = tempfile::tempdir()?;
        let repo = repo_dir.path().canonicalize()?;
        std::fs::create_dir(repo.join(".git"))?;
        let agents = repo.join(".caliche").join(AGENTS_DIR_NAME);
        std::fs::create_dir_all(&agents)?;
        std::fs::write(agents.join("review.md"), "You review.")?;
        let outside_dir = tempfile::tempdir()?;
        let outside = outside_dir.path().canonicalize()?.join("secret.md");
        std::fs::write(&outside, "OUTSIDE-SECRET")?;
        Ok((repo_dir, repo, outside_dir, outside))
    }

    fn project_prompts(repo: &Path) -> Result<Vec<(String, Option<String>)>> {
        let project = default_search_path(Some(&repo.to_path_buf()))
            .into_iter()
            .filter(|dir| dir.confine.is_some())
            .collect::<Vec<_>>();
        let mut prompts: Vec<(String, Option<String>)> = load_profiles_from_search_path(&project)?
            .into_iter()
            .map(|(name, profile)| (name, profile.system_prompt().map(str::to_string)))
            .collect();
        prompts.sort();
        Ok(prompts)
    }

    #[test]
    fn a_symlinked_project_profile_is_skipped() -> Result<()> {
        let (_repo_dir, repo, _outside_dir, outside) = repo_and_outside()?;
        symlink(&outside, repo.join(".caliche/agents/build.md"))?;

        assert_eq!(
            project_prompts(&repo)?,
            vec![("review".to_string(), Some("You review.".to_string()))]
        );
        Ok(())
    }

    #[test]
    fn a_symlinked_project_agents_dir_is_skipped() -> Result<()> {
        let (_repo_dir, repo, _outside_dir, outside) = repo_and_outside()?;
        let outside_agents = outside.parent().map(Path::to_path_buf).unwrap_or_default();
        std::fs::remove_dir_all(repo.join(".caliche").join(AGENTS_DIR_NAME))?;
        symlink(&outside_agents, repo.join(".caliche").join(AGENTS_DIR_NAME))?;

        assert_eq!(project_prompts(&repo)?, Vec::new());
        Ok(())
    }

    #[test]
    fn a_fifo_project_profile_is_skipped_without_blocking() -> Result<()> {
        let (_repo_dir, repo, _outside_dir, _outside) = repo_and_outside()?;
        let fifo = repo.join(".caliche/agents/hang.md");
        let made = std::process::Command::new("mkfifo").arg(&fifo).status()?;
        assert!(made.success());

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(project_prompts(&repo).map_err(|e| e.to_string()));
        });
        let prompts = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|_| anyhow::anyhow!("loading the profiles blocked on the FIFO"))?
            .map_err(anyhow::Error::msg)?;

        assert_eq!(
            prompts,
            vec![("review".to_string(), Some("You review.".to_string()))]
        );
        Ok(())
    }
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
