//! Declarative workspace package for session warm-start.
//!
//! A TOML document lists the workspace root, git repos to clone or attach,
//! MCP servers, and optional skills so a new session starts with those
//! bindings instead of rediscovering them. The indexer is not run here.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default relative path for a workspace declaration.
pub const DEFAULT_WORKSPACE_TOML: &str = ".caliche/workspace.toml";

/// Git repo to clone or attach under the workspace root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRepoSpec {
    /// Remote URL used when `attach` is false and `path` is missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Relative or absolute path of the worktree.
    pub path: PathBuf,
    /// Attach an existing checkout instead of cloning.
    #[serde(default)]
    pub attach: bool,
}

/// MCP server declared for a warm session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceMcpSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Skill file declared for a warm session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceSkillSpec {
    pub name: String,
    pub path: PathBuf,
}

/// On-disk workspace package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceDeclaration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Workspace root; relative paths resolve against the declaration file.
    #[serde(default = "default_root")]
    pub root: PathBuf,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repos: Vec<WorkspaceRepoSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<WorkspaceMcpSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<WorkspaceSkillSpec>,
    /// Directory that contained the TOML file (not serialized).
    #[serde(skip)]
    pub source_dir: PathBuf,
}

fn default_root() -> PathBuf {
    PathBuf::from(".")
}

/// MCP transport resolved from a declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarmStartMcpTransport {
    Stdio { command: String, args: Vec<String> },
    Http { url: String },
}

/// MCP server available to a new session after apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmStartMcpServer {
    pub name: String,
    pub transport: WarmStartMcpTransport,
}

/// Skill available to a new session after apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmStartSkill {
    pub name: String,
    pub path: PathBuf,
}

/// Result of applying a workspace declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmStartPlan {
    pub name: Option<String>,
    pub root: PathBuf,
    pub repos: Vec<PathBuf>,
    pub mcp_servers: Vec<WarmStartMcpServer>,
    pub skills: Vec<WarmStartSkill>,
}

/// Options for [`WorkspaceDeclaration::apply`].
#[derive(Debug, Clone, Copy)]
pub struct ApplyOptions {
    /// Run `git clone` for missing remote repos. Tests should leave this false
    /// and use `attach = true`.
    pub clone_missing: bool,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            clone_missing: true,
        }
    }
}

impl WorkspaceDeclaration {
    /// Load a declaration, resolving relative paths against `path`'s parent.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed.
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read workspace declaration {}", path.display()))?;
        let source_dir = path.parent().unwrap_or(path).to_path_buf();
        Self::load_from_str(&raw, &source_dir)
    }

    /// Parse TOML with `source_dir` as the path base.
    ///
    /// # Errors
    ///
    /// Returns an error when the document is invalid.
    pub fn load_from_str(raw: &str, source_dir: &Path) -> Result<Self> {
        let mut declaration: Self =
            toml::from_str(raw).context("parse workspace declaration TOML")?;
        declaration.source_dir = source_dir.to_path_buf();
        declaration.validate()?;
        Ok(declaration)
    }

    /// Load `.caliche/workspace.toml` under `root` if present.
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but is invalid.
    pub fn discover(root: &Path) -> Result<Option<Self>> {
        let path = root.join(DEFAULT_WORKSPACE_TOML);
        if path.exists() {
            return Self::load_from_path(&path).map(Some);
        }
        let fallback = root.join("workspace.toml");
        if fallback.exists() {
            return Self::load_from_path(&fallback).map(Some);
        }
        Ok(None)
    }

    fn validate(&self) -> Result<()> {
        for repo in &self.repos {
            if repo.path.as_os_str().is_empty() {
                bail!("workspace repo path is empty");
            }
            if !repo.attach && repo.url.as_ref().is_none_or(|url| url.trim().is_empty()) {
                bail!(
                    "workspace repo '{}' must set url unless attach = true",
                    repo.path.display()
                );
            }
        }
        for server in &self.mcp {
            if server.name.trim().is_empty() {
                bail!("workspace MCP server name is empty");
            }
            let has_cmd = server
                .command
                .as_ref()
                .is_some_and(|cmd| !cmd.trim().is_empty());
            let has_url = server
                .url
                .as_ref()
                .is_some_and(|url| !url.trim().is_empty());
            if has_cmd == has_url {
                bail!(
                    "workspace MCP server '{}' must set exactly one of command or url",
                    server.name
                );
            }
        }
        for skill in &self.skills {
            if skill.name.trim().is_empty() {
                bail!("workspace skill name is empty");
            }
        }
        Ok(())
    }

    /// Resolved workspace root (declaration `root` against the source dir).
    #[must_use]
    pub fn resolved_root(&self) -> PathBuf {
        resolve_path(&self.source_dir, &self.root)
    }

    /// Clone or attach repos and resolve MCP/skills for a new session.
    ///
    /// Does not start the workspace indexer.
    ///
    /// # Errors
    ///
    /// Returns an error when a repo cannot be attached/cloned or a skill is missing.
    pub fn apply(&self, options: ApplyOptions) -> Result<WarmStartPlan> {
        let root = self.resolved_root();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create workspace root {}", root.display()))?;

        let mut repos = Vec::new();
        for spec in &self.repos {
            let path = resolve_path(&root, &spec.path);
            if spec.attach || path.exists() {
                if !path.exists() {
                    bail!("workspace repo attach path missing: {}", path.display());
                }
                repos.push(canonical_or_owned(&path));
                continue;
            }
            if !options.clone_missing {
                bail!(
                    "workspace repo '{}' is missing and clone_missing is false",
                    path.display()
                );
            }
            let url = spec
                .url
                .as_deref()
                .filter(|url| !url.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!("workspace repo '{}' missing url", spec.path.display())
                })?;
            clone_repo(url, &path)?;
            repos.push(canonical_or_owned(&path));
        }

        let mut mcp_servers = Vec::new();
        for spec in &self.mcp {
            let transport = if let Some(command) = spec.command.as_ref() {
                WarmStartMcpTransport::Stdio {
                    command: command.clone(),
                    args: spec.args.clone(),
                }
            } else {
                let url = spec.url.clone().unwrap_or_default();
                crate::gateway::egress::current_policy()?
                    .validate_static(crate::gateway::egress::EgressClass::Mcp, &url)
                    .with_context(|| format!("workspace MCP '{}' url rejected", spec.name))?;
                WarmStartMcpTransport::Http { url }
            };
            mcp_servers.push(WarmStartMcpServer {
                name: spec.name.clone(),
                transport,
            });
        }

        let mut skills = Vec::new();
        for spec in &self.skills {
            let path = resolve_path(&root, &spec.path);
            if !path.exists() {
                bail!(
                    "workspace skill '{}' path missing: {}",
                    spec.name,
                    path.display()
                );
            }
            skills.push(WarmStartSkill {
                name: spec.name.clone(),
                path: canonical_or_owned(&path),
            });
        }

        Ok(WarmStartPlan {
            name: self.name.clone(),
            root: canonical_or_owned(&root),
            repos,
            mcp_servers,
            skills,
        })
    }
}

impl WarmStartPlan {
    /// MCP server names available to a new session.
    #[must_use]
    pub fn mcp_names(&self) -> Vec<&str> {
        self.mcp_servers
            .iter()
            .map(|server| server.name.as_str())
            .collect()
    }

    /// Convert declared MCP servers to discovery records.
    #[must_use]
    pub fn to_discovered_mcp(&self) -> Vec<crate::discovery::auto::DiscoveredMcpServer> {
        self.mcp_servers
            .iter()
            .map(|server| crate::discovery::auto::DiscoveredMcpServer {
                name: server.name.clone(),
                transport: match &server.transport {
                    WarmStartMcpTransport::Stdio { command, args } => {
                        crate::discovery::auto::DiscoveredMcpTransport::Stdio {
                            command: command.clone(),
                            args: args.clone(),
                            env: Vec::new(),
                        }
                    }
                    WarmStartMcpTransport::Http { url } => {
                        crate::discovery::auto::DiscoveredMcpTransport::Http { url: url.clone() }
                    }
                },
                source_path: self.root.join(DEFAULT_WORKSPACE_TOML),
                scope: crate::discovery::auto::SourceScope::WorkspaceLocal,
            })
            .collect()
    }

    /// Convert declared MCP servers to client configs.
    ///
    /// # Errors
    ///
    /// Returns an error when a discovered transport cannot be converted.
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn to_mcp_client_configs(&self) -> Result<Vec<crate::mcp::McpClientConfig>> {
        self.to_discovered_mcp()
            .into_iter()
            .map(|server| server.to_mcp_client_config())
            .collect()
    }
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn canonical_or_owned(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn clone_repo(url: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create repo parent {}", parent.display()))?;
    }
    let status = Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(url)
        .arg(dest)
        .status()
        .with_context(|| format!("spawn git clone for {url}"))?;
    if !status.success() {
        bail!("git clone {url} failed with {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_git_repo(path: &Path) {
        std::fs::create_dir_all(path.join(".git")).expect("git dir");
        std::fs::write(path.join("README.md"), "repo").expect("readme");
    }

    #[test]
    fn apply_attaches_repo_and_exposes_mcp_and_skills() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let root = dir.path();
        let vendor = root.join("vendor").join("lib");
        write_git_repo(&vendor);
        let skill_dir = root.join("skills").join("review");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(skill_dir.join("SKILL.md"), "# review\n").expect("skill");

        let toml = r#"
name = "warm"
root = "."

[[repos]]
path = "vendor/lib"
attach = true

[[mcp]]
name = "docs"
url = "https://mcp.example.com/mcp"

[[mcp]]
name = "fs"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]

[[skills]]
name = "review"
path = "skills/review/SKILL.md"
"#;
        let declaration = WorkspaceDeclaration::load_from_str(toml, root).expect("parse");
        let plan = declaration
            .apply(ApplyOptions {
                clone_missing: false,
            })
            .expect("apply");

        assert_eq!(plan.name.as_deref(), Some("warm"));
        assert_eq!(plan.repos.len(), 1);
        assert!(plan.repos[0].ends_with("lib") || plan.repos[0].ends_with("vendor/lib"));
        assert_eq!(plan.mcp_names(), vec!["docs", "fs"]);
        assert_eq!(plan.skills.len(), 1);
        assert_eq!(plan.skills[0].name, "review");

        let discovered = plan.to_discovered_mcp();
        assert_eq!(discovered.len(), 2);
        assert!(discovered.iter().any(|server| server.name == "docs"
            && matches!(
                server.transport,
                crate::discovery::auto::DiscoveredMcpTransport::Http { .. }
            )));
        assert!(discovered.iter().any(|server| server.name == "fs"
            && matches!(
                server.transport,
                crate::discovery::auto::DiscoveredMcpTransport::Stdio { .. }
            )));
    }

    #[test]
    fn discover_loads_default_path() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let pkg = dir.path().join(".caliche");
        std::fs::create_dir_all(&pkg).expect("pkg dir");
        std::fs::write(
            pkg.join("workspace.toml"),
            "name = \"found\"\nroot = \".\"\n",
        )
        .expect("write");
        let loaded = WorkspaceDeclaration::discover(dir.path())
            .expect("discover")
            .expect("present");
        assert_eq!(loaded.name.as_deref(), Some("found"));
    }

    #[test]
    fn missing_attach_repo_fails_without_clone() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let toml = r#"
[[repos]]
path = "missing"
attach = true
"#;
        let declaration = WorkspaceDeclaration::load_from_str(toml, dir.path()).expect("parse");
        let err = declaration
            .apply(ApplyOptions {
                clone_missing: false,
            })
            .expect_err("missing attach");
        assert!(err.to_string().contains("missing"));
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    #[test]
    fn apply_plan_is_available_to_new_builder_session() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let toml = r#"
name = "session"
root = "."

[[mcp]]
name = "docs"
url = "https://mcp.example.com/mcp"
"#;
        let declaration = WorkspaceDeclaration::load_from_str(toml, dir.path()).expect("parse");
        let plan = declaration
            .apply(ApplyOptions {
                clone_missing: false,
            })
            .expect("apply");
        let builder = crate::agent::AgentBuilder::new()
            .apply_workspace_warm_start(&plan)
            .expect("warm start");
        let configs = plan.to_mcp_client_configs().expect("configs");
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "docs");
        drop(builder);
    }
}
