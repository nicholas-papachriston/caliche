//! Core MCP server implementation.

use crate::agent::hooks::HookRegistry;
#[cfg(feature = "tools-rag")]
use crate::agent::tools::RagTool;
use crate::agent::tools::ToolRegistry;
use crate::mcp::protocol::*;
use crate::templates::TemplateRegistry;
use anyhow::Result;
use beef::lean::Cow;
use std::sync::Arc;
use std::time::Duration;

/// MCP server configuration
#[derive(Clone)]
pub struct McpServerConfig {
    /// Server name
    pub name: Cow<'static, str>,
    /// Server version
    pub version: Option<Cow<'static, str>>,
    /// Enable tools capability
    pub enable_tools: bool,
    /// Enable prompts capability
    pub enable_prompts: bool,
    /// Enable resources capability
    pub enable_resources: bool,
    /// Optional timeout applied to every `handle_request` call.
    ///
    /// When set, handlers that exceed the budget are aborted and replied to
    /// with `-32001 Request timeout`.
    pub request_timeout: Option<Duration>,
}

impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServerConfig")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("enable_tools", &self.enable_tools)
            .field("enable_prompts", &self.enable_prompts)
            .field("enable_resources", &self.enable_resources)
            .field("request_timeout", &self.request_timeout)
            .finish()
    }
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            name: Cow::borrowed("caliche"),
            version: Some(Cow::borrowed(env!("CARGO_PKG_VERSION"))),
            enable_tools: true,
            enable_prompts: true,
            enable_resources: true,
            request_timeout: None,
        }
    }
}

impl McpServerConfig {
    /// Create a builder for MCP server configuration
    pub fn builder() -> McpServerConfigBuilder {
        McpServerConfigBuilder::default()
    }

    /// Create configuration from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `MCP_ENABLED`: Enable MCP server functionality (default: `true`).
    /// - `MCP_SERVER_NAME`: Server name (default: `"caliche"`).
    /// - `MCP_SERVER_VERSION`: Server version (default: package version).
    /// - `MCP_CAPABILITIES`: Comma-separated capabilities to enable. Valid
    ///   tokens are `tools`, `prompts`, `resources`, or `all`. An empty or
    ///   unset value means "use defaults" (all enabled); previously an
    ///   empty string silently disabled every capability.
    /// - `MCP_REQUEST_TIMEOUT_MS`: Optional per-request timeout in
    ///   milliseconds (default: unset).
    ///
    /// Returns `None` if `MCP_ENABLED` is explicitly `false`.
    pub fn from_env() -> Option<Self> {
        let enabled = std::env::var("MCP_ENABLED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(true);
        if !enabled {
            return None;
        }

        let name = std::env::var("MCP_SERVER_NAME")
            .map(Cow::owned)
            .unwrap_or_else(|_| Cow::borrowed("caliche"));

        let version = std::env::var("MCP_SERVER_VERSION")
            .ok()
            .map(Cow::owned)
            .or(Some(Cow::borrowed(env!("CARGO_PKG_VERSION"))));

        let defaults = Self::default();
        let (enable_tools, enable_prompts, enable_resources) =
            parse_capability_env(std::env::var("MCP_CAPABILITIES").ok().as_deref()).unwrap_or((
                defaults.enable_tools,
                defaults.enable_prompts,
                defaults.enable_resources,
            ));

        let request_timeout = std::env::var("MCP_REQUEST_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|ms| *ms > 0)
            .map(Duration::from_millis);

        Some(Self {
            name,
            version,
            enable_tools,
            enable_prompts,
            enable_resources,
            request_timeout,
        })
    }
}

/// Parse the `MCP_CAPABILITIES` env var into `(tools, prompts, resources)`.
///
/// Returns `None` when the input is empty or contains nothing actionable so
/// callers can fall back to defaults.
fn parse_capability_env(value: Option<&str>) -> Option<(bool, bool, bool)> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }

    let tokens: std::collections::HashSet<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect();

    if tokens.is_empty() {
        return None;
    }

    let all = tokens.contains("all");
    let tools = all || tokens.contains("tools");
    let prompts = all || tokens.contains("prompts");
    let resources = all || tokens.contains("resources");
    let none = tokens.contains("none");

    // Explicit "none" disables every capability. Unrecognized tokens alone fall
    // back to unset so callers keep their defaults instead of zeroing caps.
    if none {
        return Some((false, false, false));
    }
    if !tools && !prompts && !resources {
        return None;
    }
    Some((tools, prompts, resources))
}

/// Builder for MCP server configuration
#[derive(Debug, Default)]
pub struct McpServerConfigBuilder {
    name: Option<Cow<'static, str>>,
    version: Option<Cow<'static, str>>,
    enable_tools: Option<bool>,
    enable_prompts: Option<bool>,
    enable_resources: Option<bool>,
    request_timeout: Option<Duration>,
}

impl McpServerConfigBuilder {
    /// Set the server name
    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the server version
    pub fn version(mut self, version: impl Into<Cow<'static, str>>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Enable or disable tools capability
    pub fn enable_tools(mut self, enabled: bool) -> Self {
        self.enable_tools = Some(enabled);
        self
    }

    /// Enable or disable prompts capability
    pub fn enable_prompts(mut self, enabled: bool) -> Self {
        self.enable_prompts = Some(enabled);
        self
    }

    /// Enable or disable resources capability
    pub fn enable_resources(mut self, enabled: bool) -> Self {
        self.enable_resources = Some(enabled);
        self
    }

    /// Apply a timeout to every request handled by the server.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Build the configuration
    pub fn build(self) -> McpServerConfig {
        let defaults = McpServerConfig::default();
        McpServerConfig {
            name: self.name.unwrap_or(defaults.name),
            version: self.version.or(defaults.version),
            enable_tools: self.enable_tools.unwrap_or(defaults.enable_tools),
            enable_prompts: self.enable_prompts.unwrap_or(defaults.enable_prompts),
            enable_resources: self.enable_resources.unwrap_or(defaults.enable_resources),
            request_timeout: self.request_timeout.or(defaults.request_timeout),
        }
    }
}

/// Builder for creating MCP servers with custom tools, prompts, and resources
#[derive(Default)]
pub struct McpServerBuilder {
    config: McpServerConfig,
    tool_registry: Option<Arc<ToolRegistry>>,
    template_registry: Option<TemplateRegistry>,
    #[cfg(feature = "tools-rag")]
    rag_tool: Option<Arc<RagTool>>,
    document_collections: Option<Arc<dashmap::DashMap<String, crate::mcp::CollectionMetadata>>>,
}

impl McpServerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set server name
    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set server version
    pub fn version(mut self, version: impl Into<Cow<'static, str>>) -> Self {
        self.config.version = Some(version.into());
        self
    }

    /// Enable or disable tools capability
    pub fn enable_tools(mut self, enabled: bool) -> Self {
        self.config.enable_tools = enabled;
        self
    }

    /// Enable or disable prompts capability
    pub fn enable_prompts(mut self, enabled: bool) -> Self {
        self.config.enable_prompts = enabled;
        self
    }

    /// Enable or disable resources capability
    pub fn enable_resources(mut self, enabled: bool) -> Self {
        self.config.enable_resources = enabled;
        self
    }

    /// Apply a per-request timeout to the resulting server.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = Some(timeout);
        self
    }

    /// Set custom tool registry
    pub fn tool_registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Set custom template registry
    pub fn template_registry(mut self, registry: TemplateRegistry) -> Self {
        self.template_registry = Some(registry);
        self
    }

    /// Add a tool to the registry.
    ///
    /// [`ToolRegistry::register`] now takes `&self`, so we can add tools without
    /// cloning the entire registry — even when multiple `Arc` references exist.
    pub fn add_tool(mut self, tool: Arc<dyn crate::agent::tools::Tool>) -> Self {
        let registry = self
            .tool_registry
            .get_or_insert_with(|| Arc::new(ToolRegistry::new()));
        registry.register(tool);
        self
    }

    /// Set RAG tool (requires tools-rag feature)
    #[cfg(feature = "tools-rag")]
    pub fn rag_tool(mut self, rag_tool: Option<Arc<RagTool>>) -> Self {
        self.rag_tool = rag_tool;
        self
    }

    /// Set document collections
    pub fn document_collections(
        mut self,
        collections: Option<Arc<dashmap::DashMap<String, crate::mcp::CollectionMetadata>>>,
    ) -> Self {
        self.document_collections = collections;
        self
    }

    /// Borrow the configured server name.
    ///
    /// Useful for language bindings (JS/Python) that need to introspect
    /// builder state before constructing a fresh builder across an `await`.
    pub fn get_name(&self) -> &str {
        &self.config.name
    }

    /// Borrow the configured server version, if any.
    pub fn get_version(&self) -> Option<&str> {
        self.config.version.as_deref()
    }

    /// Whether the tools capability will be exposed.
    pub fn tools_enabled(&self) -> bool {
        self.config.enable_tools
    }

    /// Whether the prompts capability will be exposed.
    pub fn prompts_enabled(&self) -> bool {
        self.config.enable_prompts
    }

    /// Whether the resources capability will be exposed.
    pub fn resources_enabled(&self) -> bool {
        self.config.enable_resources
    }

    /// Borrow the configured tool registry, if one was set.
    pub fn get_tool_registry(&self) -> Option<&Arc<ToolRegistry>> {
        self.tool_registry.as_ref()
    }

    /// Borrow the configured template registry, if one was set.
    pub fn get_template_registry(&self) -> Option<&TemplateRegistry> {
        self.template_registry.as_ref()
    }

    /// Borrow the configured RAG tool, if one was set.
    #[cfg(feature = "tools-rag")]
    pub fn get_rag_tool(&self) -> Option<&Arc<RagTool>> {
        self.rag_tool.as_ref()
    }

    /// Borrow the configured document-collections handle, if any.
    pub fn get_document_collections(
        &self,
    ) -> Option<&Arc<dashmap::DashMap<String, crate::mcp::CollectionMetadata>>> {
        self.document_collections.as_ref()
    }

    /// Build a Code Mode MCP server that exposes only `search` and `execute`.
    #[cfg(feature = "code-mode")]
    pub fn code_mode(self, options: crate::mcp::code_mode::CodeModeOptions) -> Result<McpServer> {
        let mut builder = crate::mcp::code_mode::CodeModeMcpServerBuilder::new()
            .name(self.config.name)
            .schema(options.schema)
            .api_base_url(options.api_base_url)
            .api_auth(options.api_auth)
            .sandbox_config(options.sandbox_config)
            .enable_prompts(self.config.enable_prompts)
            .enable_resources(self.config.enable_resources);
        if let Some(version) = self.config.version {
            builder = builder.version(version);
        }
        if let Some(template_registry) = self.template_registry {
            builder = builder.template_registry(template_registry);
        }
        builder.build()
    }

    /// Build the MCP server.
    pub fn build(self) -> Result<McpServer> {
        let tool_registry = self
            .tool_registry
            .unwrap_or_else(|| Arc::new(ToolRegistry::new()));
        let template_registry = self.template_registry.unwrap_or_default();
        let server = McpServer::with_config(tool_registry, template_registry, self.config)
            .with_document_collections(self.document_collections);
        #[cfg(feature = "tools-rag")]
        let server = server.with_rag_tool(self.rag_tool);
        Ok(server)
    }
}

/// Core MCP server that manages capabilities and handles requests
pub struct McpServer {
    /// Tool registry for tools capability
    tool_registry: Arc<ToolRegistry>,
    /// Template registry for prompts capability
    template_registry: Arc<TemplateRegistry>,
    /// Optional RAG tool for resources capability
    #[cfg(feature = "tools-rag")]
    rag_tool: Option<Arc<RagTool>>,
    /// Document collections (from API state)
    document_collections: Option<Arc<dashmap::DashMap<String, crate::mcp::CollectionMetadata>>>,
    /// Server configuration
    config: McpServerConfig,
    /// Pre-tool hooks. Empty means only the compiled policy floor applies.
    hooks: HookRegistry,
}

impl McpServer {
    /// Create a new MCP server with default configuration.
    pub fn new(tool_registry: Arc<ToolRegistry>, template_registry: TemplateRegistry) -> Self {
        Self::with_config(tool_registry, template_registry, McpServerConfig::default())
    }

    /// Create a new MCP server with custom configuration.
    pub fn with_config(
        tool_registry: Arc<ToolRegistry>,
        template_registry: TemplateRegistry,
        config: McpServerConfig,
    ) -> Self {
        Self {
            tool_registry,
            template_registry: Arc::new(template_registry),
            #[cfg(feature = "tools-rag")]
            rag_tool: None,
            document_collections: None,
            config,
            hooks: HookRegistry::new(),
        }
    }

    /// Attach pre-tool hooks. A deny or an approval request blocks the tool.
    pub fn with_hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = hooks;
        self
    }

    /// Attach an optional RAG tool for the resources capability.
    #[cfg(feature = "tools-rag")]
    pub fn with_rag_tool(mut self, rag_tool: Option<Arc<RagTool>>) -> Self {
        self.rag_tool = rag_tool;
        self
    }

    /// Attach an optional document-collection map for the resources capability.
    pub fn with_document_collections(
        mut self,
        collections: Option<Arc<dashmap::DashMap<String, crate::mcp::CollectionMetadata>>>,
    ) -> Self {
        self.document_collections = collections;
        self
    }

    /// Get server capabilities based on configuration
    pub fn get_capabilities(&self) -> ServerCapabilities {
        ServerCapabilities {
            tools: if self.config.enable_tools {
                Some(ToolsCapability {
                    list_changed: Some(false),
                })
            } else {
                None
            },
            prompts: if self.config.enable_prompts {
                Some(PromptsCapability {
                    list_changed: Some(false),
                })
            } else {
                None
            },
            resources: if self.config.enable_resources {
                Some(ResourcesCapability {
                    subscribe: Some(false),
                    list_changed: Some(false),
                })
            } else {
                None
            },
            logging: None,
            completions: None,
            tasks: None,
            experimental: None,
        }
    }

    /// Get server info
    pub fn get_server_info(&self) -> ServerInfo {
        ServerInfo {
            name: self.config.name.clone(),
            version: self.config.version.clone(),
            title: None,
            description: None,
            icons: None,
        }
    }

    /// Get tool registry (for capabilities)
    pub fn tool_registry(&self) -> &Arc<ToolRegistry> {
        &self.tool_registry
    }

    /// Get template registry (for capabilities)
    pub fn template_registry(&self) -> &Arc<TemplateRegistry> {
        &self.template_registry
    }

    /// Get RAG tool (for resources capability)
    #[cfg(feature = "tools-rag")]
    pub fn rag_tool(&self) -> Option<&Arc<RagTool>> {
        self.rag_tool.as_ref()
    }

    /// Get document collections (for resources capability)
    pub fn document_collections(
        &self,
    ) -> Option<&Arc<dashmap::DashMap<String, crate::mcp::CollectionMetadata>>> {
        self.document_collections.as_ref()
    }

    /// Per-request timeout derived from [`McpServerConfig::request_timeout`].
    pub fn request_timeout(&self) -> Option<Duration> {
        self.config.request_timeout
    }

    pub(crate) fn hooks(&self) -> &HookRegistry {
        &self.hooks
    }

    pub(crate) fn tools_enabled(&self) -> bool {
        self.config.enable_tools
    }

    pub(crate) fn prompts_enabled(&self) -> bool {
        self.config.enable_prompts
    }

    pub(crate) fn resources_enabled(&self) -> bool {
        self.config.enable_resources
    }

    /// Create an MCP handler from this server
    pub fn create_handler(self) -> crate::mcp::handler::McpHandler {
        crate::mcp::handler::McpHandler::new(self)
    }
}

impl Clone for McpServer {
    fn clone(&self) -> Self {
        Self {
            tool_registry: Arc::clone(&self.tool_registry),
            template_registry: Arc::clone(&self.template_registry),
            #[cfg(feature = "tools-rag")]
            rag_tool: self.rag_tool.clone(),
            document_collections: self.document_collections.clone(),
            config: self.config.clone(),
            hooks: self.hooks.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_capabilities_all_token() {
        let (t, p, r) = parse_capability_env(Some("all")).expect("all returns Some");
        assert!(t && p && r);
    }

    #[test]
    fn parse_capabilities_subset() {
        let (t, p, r) = parse_capability_env(Some("tools, prompts")).expect("subset returns Some");
        assert!(t && p && !r);
    }

    #[test]
    fn parse_capabilities_empty_falls_back() {
        assert!(parse_capability_env(Some("")).is_none());
        assert!(parse_capability_env(Some("   ")).is_none());
        assert!(parse_capability_env(None).is_none());
    }

    #[test]
    fn parse_capabilities_garbage_tokens_fall_back() {
        assert!(parse_capability_env(Some(",,,")).is_none());
        assert!(parse_capability_env(Some("foo,bar")).is_none());
    }

    #[test]
    fn parse_capabilities_none_disables_all() {
        let (t, p, r) = parse_capability_env(Some("none")).expect("none returns Some");
        assert!(!t && !p && !r);
    }
}
