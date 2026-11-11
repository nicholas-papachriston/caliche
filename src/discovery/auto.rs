//! Unified auto-discovery for rules, skills, commands, subagents, and MCP.

use anyhow::Context;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourceScope {
    WorkspaceLocal,
    RepoRoot,
    UserHome,
    SystemDefault,
}

impl SourceScope {
    pub fn rank(self) -> u8 {
        match self {
            Self::WorkspaceLocal => 4,
            Self::RepoRoot => 3,
            Self::UserHome => 2,
            Self::SystemDefault => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveredRule {
    pub id: String,
    pub content: String,
    pub source_path: PathBuf,
    pub scope: SourceScope,
}

#[derive(Debug, Clone)]
pub struct DiscoveredSkill {
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub required_tools: Vec<String>,
    pub triggers: Vec<String>,
    pub source_path: PathBuf,
    pub scope: SourceScope,
}

#[derive(Debug, Clone)]
pub struct DiscoveredCommand {
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub source_path: PathBuf,
    pub scope: SourceScope,
}

#[derive(Debug, Clone)]
pub struct DiscoveredSubagent {
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub tools: Vec<String>,
    /// Task invocation ACL from the `task_permissions` frontmatter key.
    ///
    /// Written as an inline mapping, for example
    /// `task_permissions: { "*": deny, reviewer: allow }`. Keys are glob patterns over
    /// target agent names; values are `allow` or `deny`.
    pub task_permissions: BTreeMap<String, String>,
    pub source_path: PathBuf,
    pub scope: SourceScope,
}

impl DiscoveredSubagent {
    /// Build coordinator metadata for this descriptor.
    ///
    /// The description becomes a `Custom` capability so capability routing can still find a
    /// discovered agent without inventing a fixed taxonomy for it.
    pub fn to_agent_metadata(&self) -> crate::orchestration::AgentMetadata {
        let mut metadata = crate::orchestration::AgentMetadata::new(
            self.name.clone(),
            self.description.clone(),
            vec![crate::orchestration::AgentCapability::Custom(
                self.name.clone(),
            )],
        );
        metadata.metadata = serde_json::json!({
            "source": "discovery",
            "source_path": self.source_path.display().to_string(),
            "scope": format!("{:?}", self.scope),
            "declared_tools": self.tools,
        });
        metadata
    }

    /// Build the privilege ceiling and task ACL declared by this descriptor.
    ///
    /// Declared tools become a hard ceiling, so a discovered file cannot grant its agent more
    /// than it lists. A descriptor without a `tools` list keeps whatever allowlist the agent
    /// instance was built with.
    pub fn to_privileges(&self) -> anyhow::Result<crate::orchestration::AgentPrivileges> {
        let mut privileges = crate::orchestration::AgentPrivileges::unrestricted();
        if !self.tools.is_empty() {
            privileges = privileges.with_max_tools(self.tools.clone());
        }
        if !self.task_permissions.is_empty() {
            let policy =
                crate::orchestration::TaskInvocationPolicy::from_config_map(&self.task_permissions)
                    .with_context(|| {
                        format!(
                            "Parse task_permissions for subagent '{}' ({})",
                            self.name,
                            self.source_path.display()
                        )
                    })?;
            privileges = privileges.with_task_policy(policy);
        }
        if !self.prompt.trim().is_empty() {
            privileges = privileges.with_persona(self.prompt.trim());
        }
        Ok(privileges)
    }
}

#[derive(Debug, Clone)]
pub enum DiscoveredMcpTransport {
    /// `env` is the server's configured environment, read from its `env`
    /// table. A value that is exactly `${NAME}` takes `NAME` from this
    /// process when the client config is built.
    Stdio {
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
    },
    /// HTTP endpoint (connected via MCP streamable HTTP). Discovered from
    /// `url` / `sse_url` keys in host config files.
    Http { url: String },
}

#[derive(Debug, Clone)]
pub struct DiscoveredMcpServer {
    pub name: String,
    pub transport: DiscoveredMcpTransport,
    pub source_path: PathBuf,
    pub scope: SourceScope,
}

#[derive(Debug, Clone, Default)]
pub struct DiscoveryDiagnostics {
    pub scanned_paths: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DiscoverySnapshot {
    pub rules: Vec<DiscoveredRule>,
    pub skills: Vec<DiscoveredSkill>,
    pub commands: Vec<DiscoveredCommand>,
    pub subagents: Vec<DiscoveredSubagent>,
    pub mcp_servers: Vec<DiscoveredMcpServer>,
    pub diagnostics: DiscoveryDiagnostics,
}

#[derive(Debug, Clone)]
pub struct DiscoveryOptions {
    pub rules: bool,
    pub skills: bool,
    pub commands: bool,
    pub subagents: bool,
    pub mcp: bool,
}

impl Default for DiscoveryOptions {
    fn default() -> Self {
        Self {
            rules: discovery_enabled("rules"),
            skills: discovery_enabled("skills"),
            commands: discovery_enabled("commands"),
            subagents: discovery_enabled("subagents"),
            mcp: discovery_enabled("mcp"),
        }
    }
}

fn discovery_enabled(category: &str) -> bool {
    if env_is_true("CALICHE_DISCOVERY_DISABLE_ALL") {
        return false;
    }
    let key = format!(
        "CALICHE_DISCOVERY_DISABLE_{}",
        category.to_ascii_uppercase()
    );
    !env_is_true(&key)
}

fn env_is_true(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            s == "1" || s == "true" || s == "yes" || s == "on"
        })
        .unwrap_or(false)
}

pub fn discover_all(options: &DiscoveryOptions) -> DiscoverySnapshot {
    // Canonical, so the no-link walk below the workspace starts from a path
    // with no link of its own (macOS `/tmp` is one).
    let cwd = match std::env::current_dir().and_then(|cwd| cwd.canonicalize()) {
        Ok(v) => v,
        Err(_) => return DiscoverySnapshot::default(),
    };
    let home_dir = std::env::var("HOME").ok().map(PathBuf::from);
    discover_in(&cwd, home_dir.as_deref(), options)
}

/// Discover from the canonical working directory `cwd`, its repository root,
/// and `home_dir`.
fn discover_in(
    cwd: &Path,
    home_dir: Option<&Path>,
    options: &DiscoveryOptions,
) -> DiscoverySnapshot {
    let cwd = cwd.to_path_buf();
    let repo_root = find_git_root(&cwd);

    let mut out = DiscoverySnapshot::default();
    if options.rules {
        discover_rules(&cwd, repo_root.as_deref(), home_dir.as_deref(), &mut out);
    }
    if options.skills {
        discover_skills(&cwd, repo_root.as_deref(), home_dir.as_deref(), &mut out);
    }
    if options.commands {
        discover_commands(&cwd, repo_root.as_deref(), home_dir.as_deref(), &mut out);
    }
    if options.subagents {
        discover_subagents(&cwd, repo_root.as_deref(), home_dir.as_deref(), &mut out);
    }
    if options.mcp {
        discover_mcp_servers(&cwd, repo_root.as_deref(), home_dir.as_deref(), &mut out);
    }

    dedupe_all(&mut out);
    out
}

fn find_git_root(cwd: &Path) -> Option<PathBuf> {
    let mut cur = Some(cwd);
    while let Some(dir) = cur {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        cur = dir.parent();
    }
    None
}

fn classify_scope(
    path: &Path,
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
) -> SourceScope {
    if path.starts_with(cwd) {
        return SourceScope::WorkspaceLocal;
    }
    if let Some(root) = repo_root
        && path.starts_with(root)
    {
        return SourceScope::RepoRoot;
    }
    if let Some(home) = home_dir
        && path.starts_with(home)
    {
        return SourceScope::UserHome;
    }
    SourceScope::SystemDefault
}

/// A discovered file and the workspace base it must stay below, if any.
///
/// Files under the working directory and the repository root are read
/// without following any link below that base: a cloned repository or a
/// sandboxed command can plant a link there that would pull an outside file
/// into the prompt or an MCP config. Files under the home directory keep
/// following links (`confine` is `None`): users symlink their own dotfiles
/// and skills there, and the sandbox cannot write to it.
struct Candidate {
    path: PathBuf,
    confine: Option<PathBuf>,
}

fn collect_candidates<F>(
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    candidates_fn: F,
) -> Vec<Candidate>
where
    F: Fn(&Path) -> Vec<PathBuf>,
{
    let mut candidates = Vec::new();
    let mut add = |base: &Path, confine: bool| {
        for path in candidates_fn(base) {
            candidates.push(Candidate {
                path,
                confine: confine.then(|| base.to_path_buf()),
            });
        }
    };
    add(cwd, true);
    if let Some(root) = repo_root
        && root != cwd
    {
        add(root, true);
    }
    if let Some(home) = home_dir {
        add(home, false);
    }
    candidates
}

/// Largest discovered file that is read; a bigger one is skipped.
const MAX_DISCOVERED_FILE_BYTES: u64 = 1024 * 1024;

fn read_file_or_skip(candidate: &Candidate, scanned_paths: &mut Vec<PathBuf>) -> Option<String> {
    let path = &candidate.path;
    scanned_paths.push(path.clone());
    let content = match &candidate.confine {
        Some(base) => read_workspace_text(base, path)?,
        None => std::fs::read_to_string(path).ok()?,
    };
    if content.trim().is_empty() {
        return None;
    }
    Some(content)
}

/// Read the text file `path` below the canonical workspace `base`, following
/// no link.
///
/// A missing file is skipped quietly. A file that is a link, sits below a
/// link, is not a regular file, is over 1 MiB, or is not UTF-8 is skipped
/// with one warning naming the path and the reason.
#[must_use]
pub fn read_workspace_text(base: &Path, path: &Path) -> Option<String> {
    use std::io::Read;
    let skip = |reason: String| {
        tracing::warn!(path = %path.display(), "skipping workspace file: {reason}");
        None
    };
    #[cfg(not(any(unix, windows)))]
    {
        let _ = base;
        return skip("no-link reads are not supported on this platform".to_string());
    }
    #[cfg(any(unix, windows))]
    let file = match caliche_sandbox::open_file_beneath(base, path) {
        Ok(file) => file,
        Err(err)
            if matches!(
                err.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return None;
        }
        Err(err) => return skip(err.to_string()),
    };
    let mut bytes = Vec::new();
    if let Err(err) = file
        .take(MAX_DISCOVERED_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
    {
        return skip(err.to_string());
    }
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_DISCOVERED_FILE_BYTES {
        return skip(format!("it is over {MAX_DISCOVERED_FILE_BYTES} bytes"));
    }
    String::from_utf8(bytes).map_or_else(|_| skip("it is not UTF-8".to_string()), Some)
}

fn discover_rules(
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let candidates = collect_candidates(cwd, repo_root, home_dir, markdown_rule_candidates);

    for candidate in candidates {
        let Some(content) = read_file_or_skip(&candidate, &mut out.diagnostics.scanned_paths)
        else {
            continue;
        };
        let path = candidate.path;
        let id = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("rule")
            .to_ascii_lowercase();
        let scope = classify_scope(&path, cwd, repo_root, home_dir);
        out.rules.push(DiscoveredRule {
            id,
            content,
            source_path: path,
            scope,
        });
    }
}

fn markdown_rule_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let names = [
        "AGENTS.md",
        "AGENT.md",
        "agent.md",
        ".caliche.md",
        ".caliche/config.md",
    ];
    for name in names {
        out.push(base.join(name));
    }
    let cursor_rules = base.join(".cursor").join("rules");
    collect_markdown_files_recursive(&cursor_rules, &mut out, 3);
    out
}

fn discover_skills(
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let candidates = collect_candidates(cwd, repo_root, home_dir, skill_candidates);

    for candidate in candidates {
        let Some(content) = read_file_or_skip(&candidate, &mut out.diagnostics.scanned_paths)
        else {
            continue;
        };
        let path = candidate.path;

        let (meta, prompt) = split_front_matter(&content);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("skill")
            .to_string();

        let name = meta
            .get("name")
            .cloned()
            .unwrap_or_else(|| normalize_name(&stem));
        let description = meta
            .get("description")
            .cloned()
            .unwrap_or_else(|| format!("Auto-discovered skill from {}", path.display()));
        let required_tools = parse_csv_list(meta.get("required_tools"));
        let triggers = parse_csv_list(meta.get("triggers"));

        let scope = classify_scope(&path, cwd, repo_root, home_dir);
        out.skills.push(DiscoveredSkill {
            name,
            description,
            prompt,
            required_tools,
            triggers,
            source_path: path,
            scope,
        });
    }
}

fn skill_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_markdown_files_recursive(&base.join(".cursor").join("skills"), &mut out, 4);
    collect_named_files_recursive(&base.join(".cursor").join("rules"), "SKILL.md", &mut out, 4);
    collect_named_files_recursive(
        &base.join(".caliche").join("skills"),
        "SKILL.md",
        &mut out,
        3,
    );
    collect_markdown_files_recursive(&base.join(".caliche").join("skills"), &mut out, 3);
    for relative in [".agents/skills", ".claude/skills", ".codex/skills"] {
        collect_named_files_recursive(&base.join(relative), "SKILL.md", &mut out, 4);
    }
    // AGENTS.md and .cursor/rules live on the rules path (`discover_rules`).
    // Including them here inflated the progressive skill index and system prompt.
    out
}

fn discover_commands(
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let candidates = collect_candidates(cwd, repo_root, home_dir, command_candidates);

    for candidate in candidates {
        let Some(content) = read_file_or_skip(&candidate, &mut out.diagnostics.scanned_paths)
        else {
            continue;
        };
        let path = candidate.path;
        let (meta, prompt) = split_front_matter(&content);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(normalize_name)
            .unwrap_or_else(|| "command".to_string());
        let description = meta
            .get("description")
            .cloned()
            .unwrap_or_else(|| "Auto-discovered command".to_string());
        let scope = classify_scope(&path, cwd, repo_root, home_dir);
        out.commands.push(DiscoveredCommand {
            name,
            description,
            prompt,
            source_path: path,
            scope,
        });
    }
}

fn command_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_markdown_files_recursive(&base.join(".caliche").join("commands"), &mut out, 3);
    collect_markdown_files_recursive(&base.join(".cursor").join("commands"), &mut out, 3);
    collect_markdown_files_recursive(&base.join(".claude").join("commands"), &mut out, 3);
    out
}

fn discover_subagents(
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let candidates = collect_candidates(cwd, repo_root, home_dir, subagent_candidates);

    for candidate in candidates {
        let Some(content) = read_file_or_skip(&candidate, &mut out.diagnostics.scanned_paths)
        else {
            continue;
        };
        let path = candidate.path;
        let (meta, prompt) = split_front_matter(&content);
        let name = meta
            .get("name")
            .cloned()
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(normalize_name)
            })
            .unwrap_or_else(|| "subagent".to_string());
        let description = meta
            .get("description")
            .cloned()
            .unwrap_or_else(|| "Auto-discovered subagent".to_string());
        let tools = parse_csv_list(meta.get("tools"));
        let task_permissions = parse_inline_map(meta.get("task_permissions"));
        let scope = classify_scope(&path, cwd, repo_root, home_dir);
        out.subagents.push(DiscoveredSubagent {
            name,
            description,
            prompt,
            tools,
            task_permissions,
            source_path: path,
            scope,
        });
    }
}

/// Directories scanned for subagent descriptor files.
///
/// The list covers Caliche's own layouts (`.caliche/agents`, `.caliche/subagents`),
/// the Shotput default (`.agents`), and Cursor (`.cursor/subagents`), so one descriptor file
/// works for the runtime, the template engine, and the editor without being copied.
fn subagent_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_markdown_files_recursive(&base.join(".cursor").join("subagents"), &mut out, 4);
    collect_markdown_files_recursive(&base.join(".caliche").join("subagents"), &mut out, 3);
    collect_markdown_files_recursive(
        &base.join(".caliche").join(SUBAGENT_PROFILE_DIR),
        &mut out,
        3,
    );
    collect_markdown_files_recursive(&base.join(SHOTPUT_SUBAGENT_DIR), &mut out, 3);
    out
}

/// Agent profile directory name shared with `agent::profile::AGENTS_DIR_NAME`.
const SUBAGENT_PROFILE_DIR: &str = "agents";

/// Shotput's default `subagentsDir`.
const SHOTPUT_SUBAGENT_DIR: &str = ".agents";

fn discover_mcp_servers(
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let candidates = collect_candidates(cwd, repo_root, home_dir, mcp_candidates);

    for candidate in candidates {
        let Some(content) = read_file_or_skip(&candidate, &mut out.diagnostics.scanned_paths)
        else {
            continue;
        };
        let path = candidate.path;
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            parse_mcp_toml(&content, &path, cwd, repo_root, home_dir, out);
        } else {
            parse_mcp_json(&content, &path, cwd, repo_root, home_dir, out);
        }
    }
}

fn mcp_candidates(base: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let names = [
        "mcp.json",
        ".mcp.json",
        ".cursor/mcp.json",
        ".cursor/mcp_config.json",
        ".claude/mcp.json",
        "caliche.toml",
        ".caliche/config.toml",
    ];
    for name in names {
        out.push(base.join(name));
    }
    out.push(
        base.join("Library")
            .join("Application Support")
            .join("Claude")
            .join("claude_desktop_config.json"),
    );
    out.push(base.join(".config").join("claude").join("mcp.json"));
    out.push(base.join(".config").join("caliche").join("mcp.json"));
    out
}

fn parse_mcp_json(
    content: &str,
    path: &Path,
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let v: serde_json::Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(_) => return,
    };
    let servers_obj = v
        .get("mcpServers")
        .or_else(|| v.get("mcp_servers"))
        .or_else(|| v.get("servers"))
        .and_then(|x| x.as_object());

    let Some(servers) = servers_obj else {
        return;
    };
    for (name, entry) in servers {
        let Some(server_obj) = entry.as_object() else {
            continue;
        };
        if let Some(cmd) = server_obj.get("command").and_then(|v| v.as_str()) {
            let args = server_obj
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let env = server_obj
                .get("env")
                .and_then(|v| v.as_object())
                .map(|table| {
                    table
                        .iter()
                        .filter_map(|(key, v)| Some((key.clone(), v.as_str()?.to_string())))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            out.mcp_servers.push(DiscoveredMcpServer {
                name: name.to_string(),
                transport: DiscoveredMcpTransport::Stdio {
                    command: cmd.to_string(),
                    args,
                    env,
                },
                source_path: path.to_path_buf(),
                scope: classify_scope(path, cwd, repo_root, home_dir),
            });
            continue;
        }
        let url = server_obj
            .get("url")
            .or_else(|| server_obj.get("sse_url"))
            .and_then(|v| v.as_str());
        if let Some(url) = url {
            out.mcp_servers.push(DiscoveredMcpServer {
                name: name.to_string(),
                transport: DiscoveredMcpTransport::Http {
                    url: url.to_string(),
                },
                source_path: path.to_path_buf(),
                scope: classify_scope(path, cwd, repo_root, home_dir),
            });
        }
    }
}

fn parse_mcp_toml(
    content: &str,
    path: &Path,
    cwd: &Path,
    repo_root: Option<&Path>,
    home_dir: Option<&Path>,
    out: &mut DiscoverySnapshot,
) {
    let value: toml::Value = match toml::from_str(content) {
        Ok(v) => v,
        Err(_) => return,
    };

    let mcp_table = match value.get("mcp").and_then(|v| v.as_table()) {
        Some(v) => v,
        None => return,
    };

    let servers_table = mcp_table
        .get("servers")
        .and_then(|v| v.as_table())
        .or_else(|| mcp_table.get("mcpServers").and_then(|v| v.as_table()));
    let Some(servers) = servers_table else {
        return;
    };

    for (name, entry) in servers {
        let Some(tbl) = entry.as_table() else {
            continue;
        };
        if let Some(command) = tbl.get("command").and_then(|v| v.as_str()) {
            let args = tbl
                .get("args")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(ToString::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let env = tbl
                .get("env")
                .and_then(|v| v.as_table())
                .map(|table| {
                    table
                        .iter()
                        .filter_map(|(key, v)| Some((key.clone(), v.as_str()?.to_string())))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            out.mcp_servers.push(DiscoveredMcpServer {
                name: name.to_string(),
                transport: DiscoveredMcpTransport::Stdio {
                    command: command.to_string(),
                    args,
                    env,
                },
                source_path: path.to_path_buf(),
                scope: classify_scope(path, cwd, repo_root, home_dir),
            });
            continue;
        }
        if let Some(url) = tbl
            .get("url")
            .or_else(|| tbl.get("sse_url"))
            .and_then(|v| v.as_str())
        {
            out.mcp_servers.push(DiscoveredMcpServer {
                name: name.to_string(),
                transport: DiscoveredMcpTransport::Http {
                    url: url.to_string(),
                },
                source_path: path.to_path_buf(),
                scope: classify_scope(path, cwd, repo_root, home_dir),
            });
        }
    }
}

fn split_front_matter(content: &str) -> (HashMap<String, String>, String) {
    let trimmed = content.trim();
    if !trimmed.starts_with("---\n") {
        return (HashMap::new(), trimmed.to_string());
    }
    let rest = &trimmed[4..];
    let Some(pos) = rest.find("\n---") else {
        return (HashMap::new(), trimmed.to_string());
    };
    let front = &rest[..pos];
    let body = rest[pos + 4..].trim().to_string();
    let mut map = HashMap::new();
    for line in front.lines() {
        if let Some((k, v)) = line.split_once(':') {
            map.insert(
                k.trim().to_ascii_lowercase(),
                v.trim().trim_matches('"').to_string(),
            );
        }
    }
    (map, body)
}

fn parse_csv_list(value: Option<&String>) -> Vec<String> {
    value
        .map(|v| {
            let raw = v.trim().trim_matches('[').trim_matches(']');
            raw.split(',')
                .map(|part| part.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// Parse an inline YAML flow mapping such as `{ "*": deny, reviewer: allow }`.
///
/// The frontmatter reader is line oriented, so only the inline form is supported here. Keys
/// keep their case because they are glob patterns over agent names; values are lowercased.
fn parse_inline_map(value: Option<&String>) -> BTreeMap<String, String> {
    let Some(raw) = value else {
        return BTreeMap::new();
    };
    let body = raw
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();
    if body.is_empty() {
        return BTreeMap::new();
    }
    let mut out = BTreeMap::new();
    for entry in body.split(',') {
        let Some((key, effect)) = entry.split_once(':').or_else(|| entry.split_once('=')) else {
            continue;
        };
        let key = key.trim().trim_matches('"').trim_matches('\'').trim();
        let effect = effect
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .to_ascii_lowercase();
        if key.is_empty() || effect.is_empty() {
            continue;
        }
        out.insert(key.to_string(), effect);
    }
    out
}

fn normalize_name(s: &str) -> String {
    s.to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn collect_markdown_files_recursive(base: &Path, out: &mut Vec<PathBuf>, max_depth: usize) {
    collect_files_recursive(base, out, max_depth, |p| {
        p.extension().and_then(|e| e.to_str()) == Some("md")
    });
}

fn collect_named_files_recursive(
    base: &Path,
    file_name: &str,
    out: &mut Vec<PathBuf>,
    max_depth: usize,
) {
    collect_files_recursive(base, out, max_depth, |p| {
        p.file_name().and_then(|n| n.to_str()) == Some(file_name)
    });
}

fn collect_files_recursive<F>(base: &Path, out: &mut Vec<PathBuf>, max_depth: usize, matcher: F)
where
    F: Fn(&Path) -> bool + Copy,
{
    if max_depth == 0 || !base.exists() || !base.is_dir() {
        return;
    }
    let rd = match std::fs::read_dir(base) {
        Ok(rd) => rd,
        Err(_) => return,
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, out, max_depth - 1, matcher);
            continue;
        }
        if matcher(&path) {
            out.push(path);
        }
    }
}

fn dedupe_all(snapshot: &mut DiscoverySnapshot) {
    snapshot.rules = dedupe_by_key(
        std::mem::take(&mut snapshot.rules),
        |r| r.id.clone(),
        |r| r.scope.rank(),
    );
    snapshot.skills = dedupe_by_key(
        std::mem::take(&mut snapshot.skills),
        |s| s.name.clone(),
        |s| s.scope.rank(),
    );
    snapshot.commands = dedupe_by_key(
        std::mem::take(&mut snapshot.commands),
        |c| c.name.clone(),
        |c| c.scope.rank(),
    );
    snapshot.subagents = dedupe_by_key(
        std::mem::take(&mut snapshot.subagents),
        |s| s.name.clone(),
        |s| s.scope.rank(),
    );
    snapshot.mcp_servers = dedupe_by_key(
        std::mem::take(&mut snapshot.mcp_servers),
        |m| m.name.clone(),
        |m| m.scope.rank(),
    );
}

fn dedupe_by_key<T, FKey, FRank>(items: Vec<T>, mut key_fn: FKey, rank_fn: FRank) -> Vec<T>
where
    FKey: FnMut(&T) -> String,
    FRank: Fn(&T) -> u8,
{
    let mut index: HashMap<String, (u8, T)> = HashMap::new();
    for item in items {
        let key = key_fn(&item);
        let rank = rank_fn(&item);
        let keep_existing = match index.get(&key) {
            Some((existing_rank, _)) => *existing_rank > rank,
            None => false,
        };
        if !keep_existing {
            index.insert(key, (rank, item));
        }
    }
    index.into_values().map(|(_, item)| item).collect()
}

impl DiscoverySnapshot {
    pub fn log_summary(&self, context: &str) {
        tracing::info!(
            target: "caliche::discovery",
            context = context,
            rules = self.rules.len(),
            skills = self.skills.len(),
            commands = self.commands.len(),
            subagents = self.subagents.len(),
            mcp_servers = self.mcp_servers.len(),
            "discovery snapshot loaded"
        );
        for warning in &self.diagnostics.warnings {
            tracing::warn!(target: "caliche::discovery", context = context, "{warning}");
        }
    }
}

/// `${NAME}` (the whole value) reads `NAME` from this process; any other
/// value is literal. An unset `NAME` gives an empty value.
#[cfg(all(feature = "mcp", feature = "mcp-client"))]
fn resolve_env_reference(value: &str) -> String {
    value
        .strip_prefix("${")
        .and_then(|rest| rest.strip_suffix('}'))
        .map_or_else(
            || value.to_string(),
            |name| caliche_contracts::credential_env::var(name).unwrap_or_default(),
        )
}

#[cfg(all(feature = "mcp", feature = "mcp-client"))]
impl DiscoveredMcpServer {
    pub fn to_mcp_client_config(&self) -> anyhow::Result<crate::mcp::McpClientConfig> {
        self.to_mcp_client_config_with(crate::mcp::stdio_command_allowlist_from_env)
    }

    /// `stdio_allowlist` is read only for a workspace or repo-local stdio server.
    fn to_mcp_client_config_with(
        &self,
        stdio_allowlist: impl FnOnce() -> anyhow::Result<Option<std::collections::HashSet<String>>>,
    ) -> anyhow::Result<crate::mcp::McpClientConfig> {
        let transport = match &self.transport {
            DiscoveredMcpTransport::Stdio { command, args, env } => {
                if command.trim().is_empty() {
                    anyhow::bail!("discovered MCP server '{}' has empty command", self.name);
                }
                if matches!(
                    self.scope,
                    SourceScope::WorkspaceLocal | SourceScope::RepoRoot
                ) {
                    // A cloned repo must not choose a server's environment:
                    // `NODE_OPTIONS` or `LD_PRELOAD` would run its code
                    // inside an allowlisted command.
                    if !env.is_empty() {
                        anyhow::bail!(
                            "discovered MCP stdio server '{}' from '{}' is workspace/repo-local and sets env; move it to a user config",
                            self.name,
                            self.source_path.display()
                        );
                    }
                    let checked = stdio_allowlist().and_then(|allowlist| {
                        crate::mcp::validate_stdio_command_against(command, allowlist.as_ref())
                    });
                    checked.with_context(|| {
                        format!(
                            "discovered MCP stdio server '{}' from '{}' is workspace/repo-local; set CALICHE_MCP_COMMAND_ALLOWLIST",
                            self.name,
                            self.source_path.display()
                        )
                    })?;
                }
                crate::mcp::McpClientTransportConfig::Stdio {
                    command: command.clone(),
                    args: args.clone(),
                    env: env
                        .iter()
                        .map(|(key, value)| (key.clone(), resolve_env_reference(value)))
                        .collect(),
                }
            }
            DiscoveredMcpTransport::Http { url } => {
                if url.trim().is_empty() {
                    anyhow::bail!("discovered MCP server '{}' has empty url", self.name);
                }
                crate::gateway::egress::current_policy()?
                    .validate_static(crate::gateway::egress::EgressClass::Mcp, url)
                    .with_context(|| {
                        format!("discovered MCP server '{}' url rejected: {url}", self.name)
                    })?;
                crate::mcp::McpClientTransportConfig::StreamableHttp { url: url.clone() }
            }
        };
        Ok(crate::mcp::McpClientConfig {
            name: self.name.clone(),
            transport,
            skip_stdio_allowlist: false,
        })
    }
}

pub fn discover_all_with_context(options: &DiscoveryOptions) -> anyhow::Result<DiscoverySnapshot> {
    let snapshot = discover_all(options);
    validate_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn validate_snapshot(snapshot: &DiscoverySnapshot) -> anyhow::Result<()> {
    for skill in &snapshot.skills {
        if skill.name.trim().is_empty() {
            anyhow::bail!(
                "discovered skill from '{}' has empty name",
                skill.source_path.display()
            );
        }
    }
    for command in &snapshot.commands {
        if command.name.trim().is_empty() {
            anyhow::bail!(
                "discovered command from '{}' has empty name",
                command.source_path.display()
            );
        }
    }
    for server in &snapshot.mcp_servers {
        if server.name.trim().is_empty() {
            anyhow::bail!(
                "discovered mcp server from '{}' has empty name",
                server.source_path.display()
            );
        }
    }
    Ok(())
}

pub fn read_file(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn skill_candidates_exclude_markdown_rules() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let base = dir.path();
        fs::write(base.join("AGENTS.md"), "# agents").expect("write AGENTS.md");
        fs::write(base.join("CLAUDE.md"), "# claude").expect("write CLAUDE.md");
        let skills_dir = base.join(".caliche").join("skills").join("demo");
        fs::create_dir_all(&skills_dir).expect("skills dir");
        fs::write(
            skills_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: demo skill\n---\nbody\n",
        )
        .expect("write SKILL.md");

        let candidates = skill_candidates(base);
        assert!(
            candidates
                .iter()
                .any(|p| p.file_name().and_then(|n| n.to_str()) == Some("SKILL.md")),
            "expected real SKILL.md candidate, got {candidates:?}"
        );
        assert!(
            candidates.iter().all(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name != "AGENTS.md" && name != "CLAUDE.md"
            }),
            "markdown rules must not be skill candidates: {candidates:?}"
        );
    }

    #[test]
    fn markdown_rules_read_agents_md_and_skip_claude_md() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let base = dir.path();
        fs::write(base.join("AGENTS.md"), "# agents").expect("write AGENTS.md");
        fs::write(base.join("CLAUDE.md"), "# other").expect("write CLAUDE.md");
        let names: Vec<String> = markdown_rule_candidates(base)
            .into_iter()
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
            })
            .collect();
        assert!(names.iter().any(|name| name == "AGENTS.md"), "{names:?}");
        assert!(names.iter().all(|name| name != "CLAUDE.md"), "{names:?}");
    }

    #[test]
    fn skill_candidates_include_harness_skill_dirs() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let base = dir.path();
        for relative in [
            ".agents/skills/demo",
            ".claude/skills/demo",
            ".codex/skills/demo",
        ] {
            let skill = base.join(relative);
            fs::create_dir_all(&skill).expect("skill dir");
            fs::write(skill.join("SKILL.md"), "---\nname: demo\n---\nbody\n").expect("skill");
        }
        let candidates = skill_candidates(base);
        assert_eq!(candidates.len(), 3, "{candidates:?}");
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    #[test]
    fn discovered_mcp_http_allows_loopback_rejects_metadata() {
        let loopback = DiscoveredMcpServer {
            name: "local".to_string(),
            transport: DiscoveredMcpTransport::Http {
                url: "http://127.0.0.1:3000/mcp".to_string(),
            },
            source_path: PathBuf::from("mcp.json"),
            scope: SourceScope::WorkspaceLocal,
        };
        loopback
            .to_mcp_client_config()
            .expect("loopback MCP is user customization");

        let metadata = DiscoveredMcpServer {
            name: "evil".to_string(),
            transport: DiscoveredMcpTransport::Http {
                url: "http://169.254.169.254/latest/meta-data/".to_string(),
            },
            source_path: PathBuf::from("mcp.json"),
            scope: SourceScope::WorkspaceLocal,
        };
        metadata
            .to_mcp_client_config()
            .expect_err("metadata MCP must fail closed");
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    #[test]
    fn discovered_mcp_stdio_from_workspace_requires_allowlist() {
        let workspace = DiscoveredMcpServer {
            name: "evil".to_string(),
            transport: DiscoveredMcpTransport::Stdio {
                command: "caliche-unallowlisted-stdio-bin".to_string(),
                args: vec!["https://evil.test".to_string()],
                env: Vec::new(),
            },
            source_path: PathBuf::from("mcp.json"),
            scope: SourceScope::WorkspaceLocal,
        };
        let err = workspace
            .to_mcp_client_config_with(|| Ok(None))
            .expect_err("cloned-repo mcp.json stdio is untrusted");
        assert_eq!(
            format!("{:#}", err),
            "discovered MCP stdio server 'evil' from 'mcp.json' is workspace/repo-local; \
             set CALICHE_MCP_COMMAND_ALLOWLIST: stdio MCP commands require \
             CALICHE_MCP_COMMAND_ALLOWLIST to be set"
        );
        let listed =
            std::collections::HashSet::from(["caliche-unallowlisted-stdio-bin".to_string()]);
        let config = workspace
            .to_mcp_client_config_with(|| Ok(Some(listed)))
            .expect("an allowlisted command is accepted");
        assert_eq!(config.name, "evil");

        let user_home = DiscoveredMcpServer {
            name: "local".to_string(),
            transport: DiscoveredMcpTransport::Stdio {
                command: "npx".to_string(),
                args: vec![
                    "-y".to_string(),
                    "@modelcontextprotocol/server-memory".to_string(),
                ],
                env: Vec::new(),
            },
            source_path: PathBuf::from("/home/user/.cursor/mcp.json"),
            scope: SourceScope::UserHome,
        };
        user_home
            .to_mcp_client_config()
            .expect("user-home MCP stdio is local customization");
    }

    #[test]
    fn mcp_stdio_env_is_read_from_json_and_toml() {
        let home = Path::new("/home/user");
        let cwd = Path::new("/work/proj");
        let mut out = DiscoverySnapshot::default();
        parse_mcp_json(
            r#"{"mcpServers":{"gh":{"command":"npx","args":["gh-mcp"],"env":{"GITHUB_TOKEN":"${GITHUB_TOKEN}","MODE":"ro","N":1}}}}"#,
            &home.join(".cursor/mcp.json"),
            cwd,
            None,
            Some(home),
            &mut out,
        );
        parse_mcp_toml(
            "[mcp.servers.db]\ncommand = \"db-mcp\"\nenv = { DB_URL = \"postgres://x\" }\n",
            &home.join(".caliche/config.toml"),
            cwd,
            None,
            Some(home),
            &mut out,
        );
        let envs: Vec<_> = out
            .mcp_servers
            .iter()
            .map(|server| match &server.transport {
                DiscoveredMcpTransport::Stdio { env, .. } => env.clone(),
                DiscoveredMcpTransport::Http { .. } => Vec::new(),
            })
            .collect();
        assert_eq!(
            envs,
            vec![
                vec![
                    ("GITHUB_TOKEN".to_string(), "${GITHUB_TOKEN}".to_string()),
                    ("MODE".to_string(), "ro".to_string()),
                ],
                vec![("DB_URL".to_string(), "postgres://x".to_string())],
            ]
        );
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    #[test]
    fn workspace_mcp_stdio_may_not_set_env() {
        for (scope, path) in [
            (SourceScope::WorkspaceLocal, "/work/proj/.mcp.json"),
            (SourceScope::RepoRoot, "/work/caliche.toml"),
        ] {
            let server = DiscoveredMcpServer {
                name: "repo".to_string(),
                transport: DiscoveredMcpTransport::Stdio {
                    command: "npx".to_string(),
                    args: Vec::new(),
                    env: vec![("NODE_OPTIONS".to_string(), "--require ./x.js".to_string())],
                },
                source_path: PathBuf::from(path),
                scope,
            };
            let err = server
                .to_mcp_client_config()
                .expect_err("repo-local env is refused");
            assert_eq!(
                err.to_string(),
                format!(
                    "discovered MCP stdio server 'repo' from '{path}' is workspace/repo-local and sets env; move it to a user config"
                )
            );
        }
    }
}

/// Workspace bases are read without following links; the home directory is not.
#[cfg(all(test, unix))]
mod link_tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;

    const OUTSIDE: &str = "OUTSIDE-SECRET-TEXT";

    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn canonical_dir() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().canonicalize().expect("canonical temp dir");
        (dir, path)
    }

    fn only(rules: bool, commands: bool, mcp: bool) -> DiscoveryOptions {
        DiscoveryOptions {
            rules,
            skills: false,
            commands,
            subagents: false,
            mcp,
        }
    }

    fn command_names(snapshot: &DiscoverySnapshot) -> Vec<&str> {
        let mut names: Vec<&str> = snapshot.commands.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        names
    }

    fn mentions_outside(snapshot: &DiscoverySnapshot) -> bool {
        snapshot.commands.iter().any(|c| c.prompt.contains(OUTSIDE))
            || snapshot
                .mcp_servers
                .iter()
                .any(|s| s.name.contains(OUTSIDE))
    }

    #[test]
    fn a_symlinked_command_file_under_the_workspace_is_skipped() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        fs::write(outside.join("secret.md"), OUTSIDE).expect("outside file");
        let commands = ws.join(".caliche/commands");
        fs::create_dir_all(&commands).expect("commands dir");
        fs::write(commands.join("ok.md"), "run the tests").expect("ok.md");
        symlink(outside.join("secret.md"), commands.join("evil.md")).expect("symlink");

        let snapshot = discover_in(&ws, None, &only(false, true, false));

        assert_eq!(command_names(&snapshot), vec!["ok"]);
        assert_eq!(snapshot.commands[0].prompt, "run the tests");
        assert!(!mentions_outside(&snapshot));
    }

    #[test]
    fn a_symlinked_commands_dir_under_the_workspace_is_skipped() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        fs::write(outside.join("secret.md"), OUTSIDE).expect("outside file");
        fs::create_dir_all(ws.join(".caliche")).expect(".caliche");
        symlink(&outside, ws.join(".caliche/commands")).expect("symlink");

        let snapshot = discover_in(&ws, None, &only(false, true, false));

        assert_eq!(command_names(&snapshot), Vec::<&str>::new());
        assert!(!mentions_outside(&snapshot));
    }

    #[test]
    fn a_symlinked_mcp_config_under_the_workspace_is_skipped() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        fs::write(
            outside.join("mcp.json"),
            format!(r#"{{"mcpServers":{{"{OUTSIDE}":{{"command":"outside-server"}}}}}}"#),
        )
        .expect("outside config");
        symlink(outside.join("mcp.json"), ws.join(".mcp.json")).expect("symlink");

        let snapshot = discover_in(&ws, None, &only(false, false, true));

        assert_eq!(snapshot.mcp_servers.len(), 0);
        assert!(!mentions_outside(&snapshot));
    }

    #[test]
    fn a_symlinked_command_under_home_is_still_read() {
        let (_ws_dir, ws) = canonical_dir();
        let (_home_dir, home) = canonical_dir();
        let (_dot_dir, dotfiles) = canonical_dir();
        fs::write(dotfiles.join("mine.md"), "my own command").expect("dotfile");
        let commands = home.join(".claude/commands");
        fs::create_dir_all(&commands).expect("home commands dir");
        symlink(dotfiles.join("mine.md"), commands.join("mine.md")).expect("symlink");

        let snapshot = discover_in(&ws, Some(&home), &only(false, true, false));

        assert_eq!(command_names(&snapshot), vec!["mine"]);
        assert_eq!(snapshot.commands[0].prompt, "my own command");
        assert_eq!(snapshot.commands[0].scope, SourceScope::UserHome);
    }
}
