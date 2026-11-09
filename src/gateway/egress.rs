//! Declarative gateway egress policy for outbound HTTP.
//!
//! Isolation stays in the sandbox; this policy is the product allowlist for
//! requests that actually leave the process (HTTP tools, web fetch/search,
//! MCP HTTP, and optional model-provider URLs). Plugin/webhook callers keep
//! [`crate::security::http_policy::HttpPolicy::from_env`] and inherit shared
//! hosts from this policy when it is configured.
//!
//! # Safe defaults
//!
//! - Unconfigured local/dev: SSRF-only (`allow_public`) for tools — private,
//!   loopback, metadata, and CGNAT stay blocked.
//! - MCP and provider URLs in `allow_public` may use loopback and private
//!   hosts, so a local model server or a local MCP server can run. Metadata
//!   and link-local stay blocked. `default_deny` allows those hosts only
//!   when the class lists them.
//! - Production, or an explicit `default_deny` mode: untrusted/tool HTTP is
//!   default-deny until hosts are listed.
//! - A lone `*` in a host list allows any public host (SSRF checks remain).

use crate::security::http_policy::{HttpPolicy, host_matches_allowlist};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// Outbound traffic class. Tool HTTP is untrusted; providers and MCP are
/// configured endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EgressClass {
    /// `http` tool, `web_fetch` / `web_search`, llms.txt probes.
    Tool,
    /// Streamable MCP HTTP endpoints.
    Mcp,
    /// Model-provider base URLs.
    Provider,
    /// Plugin steps, gateway webhooks, scheduled delivery.
    Plugin,
}

impl fmt::Display for EgressClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tool => write!(f, "tool"),
            Self::Mcp => write!(f, "mcp"),
            Self::Provider => write!(f, "provider"),
            Self::Plugin => write!(f, "plugin"),
        }
    }
}

/// How unlisted public hosts are treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EgressMode {
    /// SSRF-only: public hosts allowed unless a class allowlist is set.
    #[default]
    AllowPublic,
    /// Unlisted hosts are denied. Empty allowlist means deny-all for that class.
    DefaultDeny,
}

impl EgressMode {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "allow_public" | "allow-public" | "public" => Ok(Self::AllowPublic),
            "default_deny" | "default-deny" | "deny" => Ok(Self::DefaultDeny),
            other => bail!("unknown egress mode '{other}' (use allow_public or default_deny)"),
        }
    }
}

/// Well-known hosted model APIs. Opt-in via `include_default_providers`.
pub const DEFAULT_PROVIDER_HOSTS: &[&str] = &[
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "openrouter.ai",
    "api.perplexity.ai",
    "api.groq.com",
];

/// Declarative egress policy loaded from TOML and/or environment variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgressPolicy {
    mode: EgressMode,
    shared: Vec<String>,
    tools: Vec<String>,
    mcp: Vec<String>,
    providers: Vec<String>,
    plugin: Vec<String>,
    include_default_providers: bool,
}

impl Default for EgressPolicy {
    fn default() -> Self {
        Self {
            mode: EgressMode::AllowPublic,
            shared: Vec::new(),
            tools: Vec::new(),
            mcp: Vec::new(),
            providers: Vec::new(),
            plugin: Vec::new(),
            include_default_providers: false,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct EgressFile {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    include_default_providers: Option<bool>,
    #[serde(default)]
    allow: EgressAllowFile,
}

#[derive(Debug, Default, Deserialize)]
struct EgressAllowFile {
    #[serde(default)]
    hosts: Vec<String>,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    mcp: Vec<String>,
    #[serde(default)]
    providers: Vec<String>,
    #[serde(default)]
    plugin: Vec<String>,
}

impl EgressPolicy {
    /// Empty SSRF-only policy (local/dev default).
    #[must_use]
    pub fn allow_public() -> Self {
        Self::default()
    }

    /// Default-deny policy with no hosts (blocks all classed HTTP).
    #[must_use]
    pub fn default_deny() -> Self {
        Self {
            mode: EgressMode::DefaultDeny,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_mode(mut self, mode: EgressMode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub fn with_shared_hosts(mut self, hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.shared = normalize_hosts(hosts);
        self
    }

    #[must_use]
    pub fn with_tool_hosts(mut self, hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = normalize_hosts(hosts);
        self
    }

    #[must_use]
    pub fn with_mcp_hosts(mut self, hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.mcp = normalize_hosts(hosts);
        self
    }

    #[must_use]
    pub fn with_provider_hosts(
        mut self,
        hosts: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.providers = normalize_hosts(hosts);
        self
    }

    #[must_use]
    pub fn with_plugin_hosts(mut self, hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.plugin = normalize_hosts(hosts);
        self
    }

    #[must_use]
    pub fn with_default_providers(mut self, enabled: bool) -> Self {
        self.include_default_providers = enabled;
        self
    }

    #[must_use]
    pub fn mode(&self) -> EgressMode {
        self.mode
    }

    /// Load TOML from `path` (see `docs/gateway.md`).
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed.
    pub fn from_toml_path(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read egress policy {}", path.display()))?;
        Self::from_toml_str(&raw).with_context(|| format!("parse egress policy {}", path.display()))
    }

    /// Parse a TOML policy document.
    ///
    /// # Errors
    ///
    /// Returns an error when the document is invalid.
    pub fn from_toml_str(raw: &str) -> Result<Self> {
        let file: EgressFile = toml::from_str(raw).context("parse egress policy TOML")?;
        let mut policy = Self::default();
        if let Some(mode) = file.mode.as_deref() {
            policy.mode = EgressMode::parse(mode)?;
        }
        policy.include_default_providers = file.include_default_providers.unwrap_or(false);
        policy.shared = normalize_hosts(file.allow.hosts);
        policy.tools = normalize_hosts(file.allow.tools);
        policy.mcp = normalize_hosts(file.allow.mcp);
        policy.providers = normalize_hosts(file.allow.providers);
        policy.plugin = normalize_hosts(file.allow.plugin);
        Ok(policy)
    }

    /// File (if `CALICHE_EGRESS_POLICY_PATH` is set) overlaid with env vars.
    ///
    /// Production without an allowlist or `*` fails closed unless
    /// `CALICHE_ALLOW_INSECURE_PRODUCTION` is set.
    ///
    /// # Errors
    ///
    /// Returns an error when the policy file is invalid or production has no
    /// allowlist.
    pub fn from_env() -> Result<Self> {
        let mut policy = if let Some(path) = policy_path_from_env() {
            Self::from_toml_path(&path)?
        } else {
            Self::default()
        };

        if let Ok(mode) = std::env::var("CALICHE_EGRESS_MODE")
            && !mode.trim().is_empty()
        {
            policy.mode = EgressMode::parse(&mode)?;
        }
        policy.shared = merge_unique(policy.shared, hosts_from_env("CALICHE_EGRESS_ALLOW"));
        policy.tools = merge_unique(policy.tools, hosts_from_env("CALICHE_EGRESS_TOOLS"));
        policy.mcp = merge_unique(policy.mcp, hosts_from_env("CALICHE_EGRESS_MCP"));
        policy.providers =
            merge_unique(policy.providers, hosts_from_env("CALICHE_EGRESS_PROVIDERS"));
        if let Some(plugin) = HttpPolicy::allowlist_from_env() {
            policy.plugin = merge_unique(policy.plugin, plugin);
            if policy.mode == EgressMode::DefaultDeny || crate::security::is_production_profile() {
                policy.shared = merge_unique(policy.shared.clone(), policy.plugin.clone());
            }
        }
        if crate::security::env_truthy("CALICHE_EGRESS_INCLUDE_DEFAULT_PROVIDERS") {
            policy.include_default_providers = true;
        }

        if crate::security::is_production_profile() && !crate::security::allow_insecure_production()
        {
            if policy.mode == EgressMode::AllowPublic && !policy.has_any_allowlist() {
                policy.mode = EgressMode::DefaultDeny;
            }
            if !policy.has_any_allowlist() {
                bail!(
                    "egress allowlist required when CALICHE_ENV=production; \
                     set CALICHE_EGRESS_ALLOW, CALICHE_EGRESS_POLICY_PATH, \
                     or PLUGIN_HTTP_ALLOWLIST (use '*' to allow any public host)"
                );
            }
        }
        Ok(policy)
    }

    fn has_any_allowlist(&self) -> bool {
        !self.shared.is_empty()
            || !self.tools.is_empty()
            || !self.mcp.is_empty()
            || !self.providers.is_empty()
            || !self.plugin.is_empty()
            || self.include_default_providers
    }

    /// Hosts that apply to `class`, including shared hosts and default providers.
    #[must_use]
    pub fn hosts_for(&self, class: EgressClass) -> Vec<String> {
        let mut hosts = self.shared.clone();
        match class {
            EgressClass::Tool => hosts.extend(self.tools.iter().cloned()),
            EgressClass::Mcp => hosts.extend(self.mcp.iter().cloned()),
            EgressClass::Provider => {
                hosts.extend(self.providers.iter().cloned());
                if self.include_default_providers {
                    hosts.extend(
                        DEFAULT_PROVIDER_HOSTS
                            .iter()
                            .map(|host| (*host).to_string()),
                    );
                }
            }
            EgressClass::Plugin => {
                hosts.extend(self.plugin.iter().cloned());
            }
        }
        unique_hosts(hosts)
    }

    /// Whether this class is default-deny (unlisted hosts blocked).
    #[must_use]
    pub fn is_default_deny(&self, class: EgressClass) -> bool {
        match class {
            EgressClass::Tool | EgressClass::Plugin => {
                self.mode == EgressMode::DefaultDeny || !self.hosts_for(class).is_empty()
            }
            EgressClass::Mcp | EgressClass::Provider => self.mode == EgressMode::DefaultDeny,
        }
    }

    /// Build the SSRF [`HttpPolicy`] used for `class`.
    #[must_use]
    pub fn http_policy(&self, class: EgressClass) -> HttpPolicy {
        let hosts = self.hosts_for(class);
        let allowlist = if self.is_default_deny(class) {
            Some(hosts)
        } else if hosts.is_empty() {
            None
        } else {
            Some(hosts)
        };
        let mut policy = HttpPolicy::new(allowlist);
        if self.configured_endpoint_allows_loopback(class) {
            policy = policy.allowing_loopback_and_private();
        }
        policy
    }

    /// Local MCP servers and local model servers are configured endpoints.
    /// Tool HTTP stays fail-closed for loopback.
    fn configured_endpoint_allows_loopback(&self, class: EgressClass) -> bool {
        if !matches!(class, EgressClass::Mcp | EgressClass::Provider) {
            return false;
        }
        self.mode == EgressMode::AllowPublic
            || self
                .hosts_for(class)
                .iter()
                .any(|host| is_loopback_allow_entry(host))
    }

    /// True when `host` is permitted for `class`.
    #[must_use]
    pub fn host_allowed(&self, class: EgressClass, host: &str) -> bool {
        let hosts = self.hosts_for(class);
        if !self.is_default_deny(class) && hosts.is_empty() {
            return true;
        }
        hosts
            .iter()
            .any(|allowed| host_matches_allowlist(host, allowed))
    }

    /// Allowlist + SSRF checks without DNS (plugin/MCP load, unit tests).
    ///
    /// # Errors
    ///
    /// Returns an error when the host is denied or the URL is blocked.
    pub fn validate_static(&self, class: EgressClass, url: &str) -> Result<()> {
        self.deny_unlisted(class, url)?;
        self.http_policy(class)
            .validate_static_url(url)
            .with_context(|| format!("egress {class} url rejected: {url}"))
    }

    /// Allowlist + DNS-time SSRF checks for a live request.
    ///
    /// # Errors
    ///
    /// Returns an error when the host is denied or the URL is blocked.
    pub fn validate(
        &self,
        class: EgressClass,
        url: &str,
    ) -> Result<crate::security::ValidatedHttpUrl> {
        self.deny_unlisted(class, url)?;
        self.http_policy(class)
            .validate_resolved_url(url)
            .with_context(|| format!("egress {class} url rejected: {url}"))
    }

    fn deny_unlisted(&self, class: EgressClass, url: &str) -> Result<()> {
        let parsed = url::Url::parse(url).with_context(|| format!("invalid egress url: {url}"))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("egress url missing host: {url}"))?;
        if self.host_allowed(class, host) {
            return Ok(());
        }
        bail!(
            "egress denied for {class} HTTP: host '{host}' is not on the allowlist \
             (set CALICHE_EGRESS_ALLOW / CALICHE_EGRESS_{} or a policy file)",
            env_suffix(class)
        );
    }
}

fn env_suffix(class: EgressClass) -> &'static str {
    match class {
        EgressClass::Tool => "TOOLS",
        EgressClass::Mcp => "MCP",
        EgressClass::Provider => "PROVIDERS",
        EgressClass::Plugin => "ALLOW (or PLUGIN_HTTP_ALLOWLIST)",
    }
}

fn policy_path_from_env() -> Option<PathBuf> {
    std::env::var("CALICHE_EGRESS_POLICY_PATH")
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
}

fn hosts_from_env(name: &str) -> Vec<String> {
    std::env::var(name)
        .ok()
        .map(parse_host_list)
        .unwrap_or_default()
}

fn parse_host_list(raw: String) -> Vec<String> {
    normalize_hosts(raw.split(','))
}

fn normalize_hosts(hosts: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
    unique_hosts(
        hosts
            .into_iter()
            .map(Into::into)
            .map(|host| host.trim().to_string())
            .filter(|host| !host.is_empty()),
    )
}

fn unique_hosts(hosts: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out = Vec::new();
    for host in hosts {
        if !out
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&host))
        {
            out.push(host);
        }
    }
    out
}

fn merge_unique(base: Vec<String>, extra: Vec<String>) -> Vec<String> {
    unique_hosts(base.into_iter().chain(extra))
}

fn is_loopback_allow_entry(host: &str) -> bool {
    matches!(
        host.to_ascii_lowercase().as_str(),
        "localhost" | "localhost.localdomain" | "127.0.0.1" | "::1"
    )
}

/// Process-wide policy used by outbound HTTP call sites.
///
/// # Errors
///
/// Returns an error when the configured policy cannot be loaded.
pub fn current_policy() -> Result<EgressPolicy> {
    EgressPolicy::from_env()
}

/// HTTP policy for untrusted tool / web-fetch traffic.
///
/// # Errors
///
/// Returns an error when the configured policy cannot be loaded.
pub fn tool_http_policy() -> Result<HttpPolicy> {
    Ok(current_policy()?.http_policy(EgressClass::Tool))
}

/// HTTP policy for MCP streamable HTTP.
///
/// # Errors
///
/// Returns an error when the configured policy cannot be loaded.
pub fn mcp_http_policy() -> Result<HttpPolicy> {
    Ok(current_policy()?.http_policy(EgressClass::Mcp))
}

/// HTTP policy for model-provider URLs.
///
/// # Errors
///
/// Returns an error when the configured policy cannot be loaded.
pub fn provider_http_policy() -> Result<HttpPolicy> {
    Ok(current_policy()?.http_policy(EgressClass::Provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_deny_blocks_unlisted_tool_host() {
        let policy = EgressPolicy::default_deny().with_tool_hosts(["api.github.com"]);
        policy
            .validate_static(EgressClass::Tool, "https://api.github.com/repos")
            .expect("allowlisted tool host");
        let err = policy
            .validate_static(EgressClass::Tool, "https://evil.example/ssrf")
            .expect_err("unlisted tool host");
        let message = err.to_string();
        assert!(
            message.contains("not on the allowlist"),
            "expected clear allowlist error, got: {message}"
        );
        assert!(message.contains("evil.example"));
    }

    #[test]
    fn allow_public_permits_unlisted_public_host() {
        let policy = EgressPolicy::allow_public();
        policy
            .validate_static(EgressClass::Tool, "https://example.invalid/v1")
            .expect("ssrf-only default");
        assert!(policy.host_allowed(EgressClass::Tool, "example.invalid"));
    }

    #[test]
    fn mcp_and_provider_allow_paths() {
        let policy = EgressPolicy::default_deny()
            .with_mcp_hosts(["localhost", "mcp.example.com"])
            .with_provider_hosts(["api.openai.com"])
            .with_default_providers(true);
        policy
            .validate_static(EgressClass::Mcp, "http://localhost:3000/mcp")
            .expect("listed local MCP");
        policy
            .validate_static(EgressClass::Mcp, "https://mcp.example.com/v1")
            .expect("listed remote MCP");
        let denied = policy
            .validate_static(EgressClass::Mcp, "https://evil.example/mcp")
            .expect_err("unlisted MCP");
        assert!(denied.to_string().contains("not on the allowlist"));
        policy
            .validate_static(EgressClass::Provider, "https://api.openai.com/v1")
            .expect("listed provider");
        policy
            .validate_static(
                EgressClass::Provider,
                "https://api.anthropic.com/v1/messages",
            )
            .expect("default provider host");
        let provider_denied = policy
            .validate_static(EgressClass::Provider, "https://not-a-provider.example/v1")
            .expect_err("unlisted provider");
        assert!(provider_denied.to_string().contains("not on the allowlist"));
    }

    #[test]
    fn toml_round_trip_sets_default_deny() {
        let raw = r#"
mode = "default_deny"
include_default_providers = true

[allow]
hosts = ["api.openai.com"]
tools = ["api.github.com"]
mcp = ["localhost"]
"#;
        let policy = EgressPolicy::from_toml_str(raw).expect("toml");
        assert_eq!(policy.mode(), EgressMode::DefaultDeny);
        assert!(policy.host_allowed(EgressClass::Tool, "api.github.com"));
        assert!(policy.host_allowed(EgressClass::Tool, "api.openai.com"));
        assert!(!policy.host_allowed(EgressClass::Tool, "evil.example"));
        assert!(policy.host_allowed(EgressClass::Provider, "api.anthropic.com"));
    }

    #[test]
    fn allow_public_permits_local_provider_and_blocks_tool_loopback() {
        let policy = EgressPolicy::allow_public();
        policy
            .validate_static(EgressClass::Provider, "http://localhost:11434/v1")
            .expect("local ollama");
        policy
            .validate_static(EgressClass::Provider, "http://127.0.0.1:1234/v1")
            .expect("loopback provider address");
        let denied = policy
            .validate_static(EgressClass::Tool, "http://localhost:11434/")
            .expect_err("tool loopback stays blocked");
        let chain = format!("{denied:#}");
        assert!(
            chain.contains("blocked host"),
            "expected an SSRF denial, got: {chain}"
        );
    }

    #[test]
    fn default_deny_local_provider_requires_an_allowlist_entry() {
        let blocked = EgressPolicy::default_deny()
            .validate_static(EgressClass::Provider, "http://localhost:11434/v1")
            .expect_err("unlisted local provider");
        assert!(blocked.to_string().contains("not on the allowlist"));
        EgressPolicy::default_deny()
            .with_provider_hosts(["localhost"])
            .validate_static(EgressClass::Provider, "http://localhost:11434/v1")
            .expect("listed local provider");
    }

    #[test]
    fn empty_default_deny_blocks_all_tool_hosts() {
        let policy = EgressPolicy::default_deny();
        let err = policy
            .validate_static(EgressClass::Tool, "https://example.com/")
            .expect_err("deny-all");
        assert!(err.to_string().contains("not on the allowlist"));
    }
}
