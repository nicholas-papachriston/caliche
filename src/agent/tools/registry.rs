use crate::agent::memory::TokenCounter;
use crate::agent::tools::tool::{Tool, ToolResult};
use crate::observability::tool_metrics::{ToolMetrics, categorize_error};
use crate::utils::user_context::UserContext;
use anyhow::{Result, bail};
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const MAX_SCHEMA_REF_INLINE_DEPTH: usize = 16;

/// Registry that holds [`Tool`] implementations available to an agent.
///
/// The tool map is stored behind an `RwLock` so tools can be registered
/// concurrently without cloning the entire registry. Schema caches are
/// invalidated automatically via an atomic version counter whenever a tool
/// is added.
pub struct ToolRegistry {
    /// Lock-protected tool map — `register` takes `&self` so callers holding
    /// an `Arc<ToolRegistry>` don't need to clone the whole registry first.
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
    /// When `Some`, only these registered tools are exposed in schemas / executable
    /// as "active". `None` means every registered tool is active (default).
    ///
    /// This enables plugin-style deferred catalogs: register many tools, activate
    /// a subset for the current turn.
    active: RwLock<Option<HashSet<String>>>,
    /// Cached tool schemas, keyed by version number.
    cached_schemas: RwLock<Option<(Arc<Vec<crate::messages::ToolSchema>>, usize)>>,
    /// Cached schema token estimate, keyed by version number.
    cached_schema_tokens: RwLock<Option<(usize, usize)>>,
    /// Monotonically increasing counter; bumped on every `register` call.
    version: AtomicUsize,
    /// Optional metrics tracker for observability.
    metrics: Option<Arc<ToolMetrics>>,
    /// Current user context for tool execution.
    user_context: Option<UserContext>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.tools.read();
        let mut names: Vec<&str> = guard.keys().map(std::string::String::as_str).collect();
        names.sort_unstable();

        f.debug_struct("ToolRegistry")
            .field("tool_count", &names.len())
            .field("tools", &names)
            .finish()
    }
}

impl ToolRegistry {
    /// Resolve canonical ergonomic aliases to their registered tool names.
    ///
    /// This is the name a call runs as, so the policy gate and hooks decide
    /// on this name too: surrounding space is trimmed, and `memory`, `web`,
    /// `shell` and `session` become the tools they run.
    ///
    /// Note: bare `search` is intentionally not aliased to `semantic_search` — it
    /// collided with MCP code-mode `search` and pulled RAG tools into coding sessions.
    /// Use `fs` op=grep for codebase search, or request `semantic_search` explicitly.
    ///
    /// `session` resolves to `sessions_send` for get/execute lookup. Active-set
    /// filters expand `session` to the full sessions_* family via
    /// [`Self::expand_tool_alias_names`].
    #[must_use]
    pub fn canonical_tool_name(name: &str) -> &str {
        caliche_policy::tool_name_as_run(name)
    }

    /// Expand ergonomic aliases used when building active-tool filters.
    ///
    /// Matches `ToolRegistryBuilder::expand_canonical_aliases`: `session` becomes
    /// the full sessions_* family so activation stays consistent with registration.
    fn expand_tool_alias_names(name: &str) -> Vec<String> {
        let trimmed = name.trim();
        if trimmed.eq_ignore_ascii_case("session") {
            return vec![
                "sessions_list".to_string(),
                "sessions_history".to_string(),
                "sessions_send".to_string(),
                "sessions_spawn".to_string(),
            ];
        }
        vec![Self::canonical_tool_name(trimmed).to_string()]
    }

    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            active: RwLock::new(None),
            cached_schemas: RwLock::new(None),
            cached_schema_tokens: RwLock::new(None),
            version: AtomicUsize::new(0),
            metrics: None,
            user_context: None,
        }
    }

    /// Create a new registry with metrics tracking.
    #[must_use]
    pub fn with_metrics(metrics: Arc<ToolMetrics>) -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            active: RwLock::new(None),
            cached_schemas: RwLock::new(None),
            cached_schema_tokens: RwLock::new(None),
            version: AtomicUsize::new(0),
            metrics: Some(metrics),
            user_context: None,
        }
    }

    fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::Release);
        *self.cached_schemas.write() = None;
        *self.cached_schema_tokens.write() = None;
    }

    fn is_name_active_locked(active: &Option<HashSet<String>>, name: &str) -> bool {
        match active {
            None => true,
            Some(set) => set.contains(name),
        }
    }

    /// Returns true when the tool is registered and currently active.
    pub fn is_active(&self, name: &str) -> bool {
        let canonical = Self::canonical_tool_name(name);
        if !self.tools.read().contains_key(canonical) {
            return false;
        }
        Self::is_name_active_locked(&self.active.read(), canonical)
    }

    /// Replace the active set. `None` activates all registered tools.
    pub fn set_active_tools(&self, names: Option<impl IntoIterator<Item = impl Into<String>>>) {
        let next = names.map(|iter| {
            let mut set = HashSet::new();
            for name in iter {
                let owned = name.into();
                for expanded in Self::expand_tool_alias_names(&owned) {
                    set.insert(expanded);
                }
            }
            set
        });
        *self.active.write() = next;
        self.bump_version();
    }

    /// Mark tools as active (creating an active set if currently unrestricted).
    pub fn activate(&self, names: impl IntoIterator<Item = impl AsRef<str>>) {
        let mut guard = self.active.write();
        let set = guard.get_or_insert_with(HashSet::new);
        for name in names {
            for expanded in Self::expand_tool_alias_names(name.as_ref()) {
                set.insert(expanded);
            }
        }
        drop(guard);
        self.bump_version();
    }

    /// Remove tools from the active set. No-op when all tools are active (`None`).
    pub fn deactivate(&self, names: impl IntoIterator<Item = impl AsRef<str>>) {
        let mut guard = self.active.write();
        let Some(set) = guard.as_mut() else {
            return;
        };
        for name in names {
            for expanded in Self::expand_tool_alias_names(name.as_ref()) {
                set.remove(expanded.as_str());
            }
        }
        drop(guard);
        self.bump_version();
    }

    /// Clear the active filter so every registered tool is active again.
    pub fn activate_all(&self) {
        *self.active.write() = None;
        self.bump_version();
    }

    /// Names of currently active tools (sorted).
    pub fn active_tool_names(&self) -> Vec<String> {
        let tools = self.tools.read();
        let active = self.active.read();
        let mut names: Vec<String> = tools
            .keys()
            .filter(|name| Self::is_name_active_locked(&active, name))
            .cloned()
            .collect();
        names.sort_unstable();
        names
    }

    /// Set metrics tracker (for existing registry).
    pub fn set_metrics(&mut self, metrics: Arc<ToolMetrics>) {
        self.metrics = Some(metrics);
    }

    /// Set user context for tool execution.
    pub fn set_user_context(&mut self, context: Option<UserContext>) {
        self.user_context = context;
    }

    /// Register a tool.
    ///
    /// Takes `&self` so callers holding an `Arc<ToolRegistry>` can add tools
    /// without cloning the registry — even when multiple references exist.
    pub fn register(&self, tool: Arc<dyn Tool>) {
        self.tools.write().insert(tool.name().to_string(), tool);
        self.bump_version();
    }

    /// Register a tool without exposing it in schemas until [`Self::activate`].
    ///
    /// If no active set exists yet, one is created from currently registered tools
    /// (excluding the deferred tool) so existing tools remain visible.
    pub fn register_deferred(&self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        {
            let mut tools = self.tools.write();
            let mut active = self.active.write();
            if active.is_none() {
                let current: HashSet<String> = tools.keys().cloned().collect();
                *active = Some(current);
            }
            tools.insert(name, tool);
        }
        self.bump_version();
    }

    /// Remove a tool by name (e.g. for plugin delete/update).
    pub fn unregister(&self, name: &str) {
        self.tools.write().remove(name);
        if let Some(set) = self.active.write().as_mut() {
            set.remove(name);
        }
        self.bump_version();
    }

    /// Look up a tool by name.
    ///
    /// Returns a clone of the `Arc` to avoid holding the read lock across
    /// await points.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let canonical = Self::canonical_tool_name(name);
        self.tools.read().get(canonical).cloned()
    }

    /// Returns `true` if the named tool is read-only and safe in Plan/Ask modes.
    pub fn is_readonly(&self, name: &str) -> bool {
        let canonical = Self::canonical_tool_name(name);
        self.tools.read().get(canonical).map_or_else(
            || !crate::agent::modes::is_write_tool(name),
            |tool| tool.is_readonly(),
        )
    }

    /// Number of registered tools.
    ///
    /// Prefer this over `list().len()`: [`Self::list`] clones every name and description
    /// and rebuilds every parameter schema.
    pub fn len(&self) -> usize {
        self.tools.read().len()
    }

    /// Returns `true` when no tool is registered.
    ///
    /// Prefer this over `list().is_empty()`, which materializes the whole tool listing.
    pub fn is_empty(&self) -> bool {
        self.tools.read().is_empty()
    }

    pub fn list(&self) -> Vec<(String, String, serde_json::Value)> {
        self.tools
            .read()
            .values()
            .map(|tool| {
                (
                    tool.name().to_string(),
                    tool.description().to_string(),
                    tool.parameters(),
                )
            })
            .collect()
    }

    /// Execute a tool with optional user context and metrics.
    ///
    /// Context and metrics are passed as parameters rather than stored on the
    /// registry so that a shared `Arc<ToolRegistry>` can be used across requests
    /// without cloning.
    pub async fn execute_with_context(
        &self,
        name: &str,
        arguments: serde_json::Value,
        user_context: Option<&UserContext>,
        metrics: Option<&Arc<ToolMetrics>>,
    ) -> ToolResult {
        let start = metrics.is_some().then(Instant::now);
        let canonical_name = Self::canonical_tool_name(name);

        // Clone the Arc (O(1)) before any await to avoid holding the read lock.
        let Some(tool) = self.get(canonical_name) else {
            return ToolResult::error(format!("Tool '{name}' not found"));
        };
        if !Self::is_name_active_locked(&self.active.read(), canonical_name) {
            return ToolResult::error(format!(
                "Tool '{canonical_name}' is registered but not active"
            ));
        }
        let result = tool.execute_with_context(arguments, user_context).await;

        // Record metrics if available
        if let Some(metrics) = metrics {
            let duration_ms = start.map_or(0, |started| started.elapsed().as_millis() as u64);
            let success = result.success;
            // Store error type in owned String to avoid temporary reference
            let error_type_owned = result.error.as_ref().map(|e| categorize_error(e));
            let error_type = error_type_owned.as_deref();

            metrics
                .record_execution(canonical_name, duration_ms, success, error_type)
                .await;
        }

        result
    }

    /// Execute a tool using stored user context and metrics.
    /// Prefer using `execute_with_context` when `ToolRegistry` is shared to avoid cloning.
    pub async fn execute(&self, name: &str, arguments: serde_json::Value) -> ToolResult {
        self.execute_with_context(
            name,
            arguments,
            self.user_context.as_ref(),
            self.metrics.as_ref(),
        )
        .await
    }

    /// Execute a tool inside a sandbox when provided; otherwise behaves like `execute_with_context`.
    /// Use when a sandbox handle is available (e.g. workspace isolation). The agent chat loop
    /// currently calls [`execute_with_context`](Self::execute_with_context) only; call this
    /// when you have a [`SandboxHandle`](caliche_sandbox::SandboxHandle) and want tools
    /// that support it (bash, fs) to run inside the sandbox.
    #[cfg(feature = "sandbox")]
    pub async fn execute_sandboxed(
        &self,
        name: &str,
        arguments: serde_json::Value,
        sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
        user_context: Option<&UserContext>,
        metrics: Option<&Arc<ToolMetrics>>,
    ) -> ToolResult {
        let start = metrics.is_some().then(Instant::now);
        let canonical_name = Self::canonical_tool_name(name);

        // Clone the Arc (O(1)) before any await to avoid holding the read lock.
        let Some(tool) = self.get(canonical_name) else {
            return ToolResult::error(format!("Tool '{name}' not found"));
        };
        if !Self::is_name_active_locked(&self.active.read(), canonical_name) {
            return ToolResult::error(format!(
                "Tool '{canonical_name}' is registered but not active"
            ));
        }
        let result = {
            if let Some(sandbox_handle) = sandbox {
                if !sandbox_handle.sandbox_active() {
                    tool.execute_with_context(arguments, user_context).await
                } else if !sandbox_handle.is_tool_allowed(canonical_name) {
                    ToolResult::error(format!(
                        "Tool '{canonical_name}' is denied by sandbox tool policy"
                    ))
                } else {
                    tool.execute_sandboxed(arguments, Some(sandbox_handle))
                        .await
                }
            } else {
                tool.execute_with_context(arguments, user_context).await
            }
        };

        if let Some(metrics) = metrics {
            let duration_ms = start.map_or(0, |started| started.elapsed().as_millis() as u64);
            let success = result.success;
            let error_type_owned = result.error.as_ref().map(|e| categorize_error(e));
            let error_type = error_type_owned.as_deref();
            metrics
                .record_execution(canonical_name, duration_ms, success, error_type)
                .await;
        }

        result
    }

    fn parse_relaxed_argument_string(raw: &str) -> Option<serde_json::Value> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return Some(v);
        }

        // Best-effort fallback for weak model outputs such as:
        // "path: /tmp/foo, op: read" or multi-line key:value pairs.
        let mut map = serde_json::Map::new();
        for segment in trimmed.split([',', '\n']) {
            let seg = segment.trim();
            if seg.is_empty() {
                continue;
            }
            let Some((k, v)) = seg.split_once(':') else {
                continue;
            };
            let key = k.trim().trim_matches('"').trim_matches('\'');
            if key.is_empty() {
                continue;
            }
            let value_raw = v.trim();
            let value = if value_raw.eq_ignore_ascii_case("true") {
                serde_json::Value::Bool(true)
            } else if value_raw.eq_ignore_ascii_case("false") {
                serde_json::Value::Bool(false)
            } else if value_raw.eq_ignore_ascii_case("null") {
                serde_json::Value::Null
            } else if let Ok(n) = value_raw.parse::<i64>() {
                serde_json::Value::Number(n.into())
            } else {
                serde_json::Value::String(
                    value_raw.trim_matches('"').trim_matches('\'').to_string(),
                )
            };
            map.insert(key.to_string(), value);
        }

        if map.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(map))
        }
    }

    fn infer_fs_op(args: &mut serde_json::Map<String, serde_json::Value>) {
        if args.contains_key("op") {
            return;
        }
        let op = if args.contains_key("operations") {
            "edit"
        } else if args.contains_key("content") {
            "write"
        } else if args
            .get("pattern")
            .and_then(|v| v.as_str())
            .is_some_and(|pattern| pattern.contains('*') || pattern.contains('?'))
        {
            "glob"
        } else if args.contains_key("pattern") {
            "grep"
        } else if args
            .get("path")
            .and_then(|v| v.as_str())
            .is_some_and(|p| p.ends_with('/') || p.ends_with('\\'))
        {
            "list"
        } else {
            // Default to read for path-only calls; execution can still return
            // a concrete error if the path is a directory.
            "read"
        };
        args.insert("op".to_string(), serde_json::Value::String(op.to_string()));
    }

    /// Normalize weak / malformed tool argument payloads into a usable object.
    /// Handles:
    /// - {"raw":"..."} wrappers (and mixed wrappers with extra metadata)
    /// - string payloads containing JSON or key:value pairs
    /// - fs calls missing `op` by inferring from provided fields
    pub fn normalize_tool_arguments(
        name: &str,
        arguments: &serde_json::Value,
    ) -> serde_json::Value {
        let canonical_name = Self::canonical_tool_name(name);
        let mut normalized = match arguments {
            serde_json::Value::String(s) => {
                Self::parse_relaxed_argument_string(s).unwrap_or_else(|| arguments.clone())
            }
            _ => arguments.clone(),
        };

        let parsed_from_raw = normalized
            .as_object()
            .and_then(|obj| obj.get("raw"))
            .and_then(|v| v.as_str())
            .and_then(Self::parse_relaxed_argument_string);

        if let Some(parsed) = parsed_from_raw {
            match (&mut normalized, parsed) {
                (serde_json::Value::Object(obj), serde_json::Value::Object(parsed_obj)) => {
                    for (k, v) in parsed_obj {
                        obj.entry(k).or_insert(v);
                    }
                    obj.remove("raw");
                }
                (_, parsed_non_object) => {
                    normalized = parsed_non_object;
                }
            }
        }

        if canonical_name == "fs"
            && let Some(obj) = normalized.as_object_mut()
        {
            Self::infer_fs_op(obj);
        }

        normalized
    }

    /// Validate tool call arguments against the tool's schema.
    ///
    /// Returns `Ok(())` if valid, or an error message describing what is wrong.
    pub fn validate_tool_call(&self, name: &str, arguments: &serde_json::Value) -> Result<()> {
        let canonical_name = Self::canonical_tool_name(name);
        let arguments = Self::normalize_tool_arguments(canonical_name, arguments);
        let tool = match self.get(canonical_name) {
            Some(t) => t,
            None => bail!("Tool '{name}' not found"),
        };

        let params = tool.parameters();

        // Get required parameters
        let required = params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        // Check if arguments is an object
        let args_obj = if let Some(obj) = arguments.as_object() {
            obj
        } else {
            if !required.is_empty() {
                bail!("Tool '{name}' requires arguments to be an object, but got: {arguments:?}");
            }
            return Ok(()); // No required params, empty args are OK
        };

        // Check required parameters
        for param in &required {
            if !args_obj.contains_key(param) {
                bail!("Tool '{name}' requires parameter '{param}' but it was not provided");
            }

            // Check if the value is null or empty string (which are effectively missing)
            if let Some(value) = args_obj.get(param) {
                if value.is_null() {
                    bail!("Tool '{name}' requires parameter '{param}' but it was null");
                }
                if let Some(s) = value.as_str()
                    && s.is_empty()
                {
                    bail!("Tool '{name}' requires parameter '{param}' but it was empty");
                }
            }
        }

        Ok(())
    }

    /// Get cached tool schemas for LLM function calling
    /// Builds and caches schemas on first call for efficient reuse
    /// Uses `RwLock` with double-checked locking for fast reads and clearable cache
    pub fn get_tool_schemas(&self) -> Arc<Vec<crate::messages::ToolSchema>> {
        let current_version = self.version.load(Ordering::Acquire);

        // Fast path: read lock to check cache
        {
            let cache = self.cached_schemas.read();
            if let Some((schemas, ver)) = cache.as_ref()
                && *ver == current_version
            {
                return Arc::clone(schemas);
            }
        }

        // Slow path: write lock to rebuild cache
        let mut cache = self.cached_schemas.write();

        // Double-check after acquiring write lock (another thread may have rebuilt)
        if let Some((schemas, ver)) = cache.as_ref()
            && *ver == current_version
        {
            return Arc::clone(schemas);
        }

        // Build schemas while holding the tools read lock, then release it
        // before acquiring the schema cache write lock (avoids lock ordering issues).
        let schema_vec: Vec<crate::messages::ToolSchema> = {
            let tools_guard = self.tools.read();
            let active_guard = self.active.read();
            let mut v = Vec::with_capacity(tools_guard.len());
            for tool in tools_guard.values() {
                if !Self::is_name_active_locked(&active_guard, tool.name()) {
                    continue;
                }
                let params = tool.parameters();
                let schema_type_str = params
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("object");
                let schema_type = if schema_type_str == "object" {
                    beef::lean::Cow::borrowed("object")
                } else {
                    beef::lean::Cow::owned(schema_type_str.to_string())
                };

                let mut properties = params
                    .get("properties")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                inline_local_schema_refs(&mut properties, &params, 0);

                let required: Vec<beef::lean::Cow<'static, str>> = params
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .map(beef::lean::Cow::owned)
                            .collect()
                    })
                    .unwrap_or_default();

                v.push(crate::messages::ToolSchema {
                    name: tool.name_as_cow(),
                    description: tool.description_as_cow(),
                    parameters: crate::messages::ParametersSchema {
                        schema_type,
                        properties,
                        required,
                    },
                });
            }
            v
        }; // tools_guard / active_guard dropped here

        let schemas = Arc::new(schema_vec);

        // Store in cache (cache write lock re-acquired after tools read lock released)
        *cache = Some((Arc::clone(&schemas), current_version));

        schemas
    }

    /// Estimate tokens consumed by tool schemas when sent via native/text-based tool calling APIs.
    ///
    /// Note: This is an estimate to improve prompt budgeting. Exact tokenization varies by provider.
    /// Result is cached and invalidated when tools are added/removed.
    pub fn estimate_schema_tokens(&self, token_counter: &dyn TokenCounter) -> usize {
        let current_version = self.version.load(Ordering::Acquire);
        {
            let cache = self.cached_schema_tokens.read();
            if let Some((tokens, version)) = *cache
                && version == current_version
            {
                return tokens;
            }
        }

        // Avoid cloning schema data; Arc clone is cheap.
        let schemas = self.get_tool_schemas();
        let tokens: usize = schemas
            .iter()
            .map(|schema| {
                let mut total = 0usize;
                total = total.saturating_add(token_counter.estimate_text_tokens(&schema.name));
                total =
                    total.saturating_add(token_counter.estimate_text_tokens(&schema.description));

                // Parameters are JSON schema; serialize for approximate sizing.
                if let Ok(params_json) = serde_json::to_string(&schema.parameters) {
                    total = total.saturating_add(token_counter.estimate_text_tokens(&params_json));
                }

                // Rough structural overhead for function schema wrappers.
                total.saturating_add(20)
            })
            .sum();

        *self.cached_schema_tokens.write() = Some((tokens, current_version));
        tokens
    }
}

fn inline_local_schema_refs(schema: &mut Value, root_schema: &Value, depth: usize) {
    if depth >= MAX_SCHEMA_REF_INLINE_DEPTH {
        return;
    }

    match schema {
        Value::Object(object) => {
            if let Some(reference) = object.get("$ref").and_then(Value::as_str)
                && let Some(resolved) = resolve_local_schema_ref(root_schema, reference)
            {
                let mut replacement = resolved.clone();
                inline_local_schema_refs(&mut replacement, root_schema, depth + 1);
                if let Value::Object(replacement_object) = &mut replacement {
                    for (key, value) in object.iter() {
                        if key != "$ref" {
                            replacement_object.insert(key.clone(), value.clone());
                        }
                    }
                }
                *schema = replacement;
                return;
            }

            object.remove("$schema");
            object.remove("$defs");
            object.remove("definitions");
            for value in object.values_mut() {
                inline_local_schema_refs(value, root_schema, depth + 1);
            }
        }
        Value::Array(items) => {
            for item in items {
                inline_local_schema_refs(item, root_schema, depth + 1);
            }
        }
        _ => {}
    }
}

fn resolve_local_schema_ref<'a>(root_schema: &'a Value, reference: &str) -> Option<&'a Value> {
    let key = reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))?;
    let key = decode_json_pointer_segment(key);
    root_schema
        .get("$defs")
        .or_else(|| root_schema.get("definitions"))
        .and_then(|definitions| definitions.get(key.as_ref()))
}

fn decode_json_pointer_segment(segment: &str) -> std::borrow::Cow<'_, str> {
    if !segment.contains('~') {
        return std::borrow::Cow::Borrowed(segment);
    }

    std::borrow::Cow::Owned(segment.replace("~1", "/").replace("~0", "~"))
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::ToolMeta;
    use async_trait::async_trait;

    #[derive(Debug)]
    struct DummyTool {
        name: &'static str,
    }

    #[derive(Debug)]
    struct RefTool;

    impl ToolMeta for DummyTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "dummy"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }
    }

    #[async_trait]
    impl Tool for DummyTool {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    impl ToolMeta for RefTool {
        fn name(&self) -> &str {
            "ref_tool"
        }

        fn description(&self) -> &str {
            "uses local schema refs"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "request": {
                        "$ref": "#/$defs/ImageRequest",
                        "description": "Request payload"
                    }
                },
                "required": ["request"],
                "$defs": {
                    "ImageRequest": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            })
        }
    }

    #[async_trait]
    impl Tool for RefTool {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    #[test]
    fn keeps_short_aliases_but_rejects_removed_legacy_names() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool { name: "bash" }));
        registry.register(Arc::new(DummyTool { name: "http" }));
        registry.register(Arc::new(DummyTool { name: "workspace" }));

        assert!(registry.get("shell").is_some());
        assert!(registry.get(" Shell ").is_some());
        assert!(registry.get("WEB").is_some());
        assert!(registry.get("memory").is_some());
        assert!(registry.get("read_file").is_none());
        assert!(registry.get("git_status").is_none());
    }

    #[test]
    fn session_alias_activates_full_sessions_family() {
        let registry = ToolRegistry::new();
        for name in [
            "sessions_list",
            "sessions_history",
            "sessions_send",
            "sessions_spawn",
        ] {
            registry.register(Arc::new(DummyTool { name }));
        }

        registry.set_active_tools(Some(["session"]));
        assert!(registry.is_active("sessions_list"));
        assert!(registry.is_active("sessions_history"));
        assert!(registry.is_active("sessions_send"));
        assert!(registry.is_active("sessions_spawn"));
        // get/execute still map bare session -> sessions_send
        assert_eq!(
            registry.get("session").map(|tool| tool.name().to_string()),
            Some("sessions_send".into())
        );
    }

    #[tokio::test]
    async fn aliases_apply_consistently_to_active_filters_schemas_and_execution() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool { name: "bash" }));
        registry.register(Arc::new(DummyTool { name: "http" }));
        registry.register(Arc::new(DummyTool { name: "workspace" }));

        // Active set configured via aliases should resolve to canonical names.
        registry.set_active_tools(Some(["shell", "web"]));
        assert!(registry.is_active("shell"));
        assert!(registry.is_active("bash"));
        assert!(registry.is_active("web"));
        assert!(registry.is_active("http"));
        assert!(!registry.is_active("memory"));
        assert!(!registry.is_active("workspace"));

        let schemas = registry.get_tool_schemas();
        let names: Vec<&str> = schemas.iter().map(|s| s.name.as_ref()).collect();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"http"));
        assert!(!names.contains(&"workspace"));

        let ok = registry.execute("shell", serde_json::json!({})).await;
        assert!(ok.success);

        let denied = registry.execute("memory", serde_json::json!({})).await;
        assert!(!denied.success);
        assert!(
            denied
                .error
                .as_deref()
                .is_some_and(|e| e.contains("not active"))
        );
    }

    #[test]
    fn inlines_local_schema_refs_for_tool_schemas() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(RefTool));

        let schemas = registry.get_tool_schemas();
        let schema = schemas
            .iter()
            .find(|schema| schema.name.as_ref() == "ref_tool")
            .expect("ref tool schema should be present");
        let request = schema
            .parameters
            .properties
            .get("request")
            .expect("request property should exist");

        assert!(request.get("$ref").is_none());
        assert_eq!(request.get("type").and_then(Value::as_str), Some("object"));
        assert!(request.pointer("/properties/prompt").is_some());
        assert_eq!(
            request.get("description").and_then(Value::as_str),
            Some("Request payload")
        );
    }

    #[test]
    fn deferred_tools_hidden_until_activated() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool { name: "bash" }));
        registry.register_deferred(Arc::new(DummyTool { name: "http" }));

        let schemas = registry.get_tool_schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name.as_ref(), "bash");
        assert!(!registry.is_active("http"));

        registry.activate(["http"]);
        let schemas = registry.get_tool_schemas();
        assert_eq!(schemas.len(), 2);
        assert!(registry.is_active("http"));

        registry.deactivate(["http"]);
        assert!(!registry.is_active("http"));
        assert_eq!(registry.get_tool_schemas().len(), 1);

        registry.activate_all();
        assert!(registry.is_active("http"));
        assert_eq!(registry.get_tool_schemas().len(), 2);
    }

    #[tokio::test]
    async fn inactive_tools_cannot_execute() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool { name: "bash" }));
        registry.set_active_tools(Some(["bash"]));
        registry.register_deferred(Arc::new(DummyTool { name: "http" }));

        let denied = registry.execute("http", serde_json::json!({})).await;
        assert!(!denied.success);
        assert!(
            denied
                .error
                .as_deref()
                .is_some_and(|e| e.contains("not active"))
        );

        registry.activate(["http"]);
        let ok = registry.execute("http", serde_json::json!({})).await;
        assert!(ok.success);
    }

    #[derive(Debug)]
    struct RequiredParamTool;

    impl ToolMeta for RequiredParamTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "requires text"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            })
        }
    }

    #[async_trait]
    impl Tool for RequiredParamTool {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    #[test]
    fn normalize_tool_arguments_infers_fs_op_and_unwraps_raw() {
        let write = ToolRegistry::normalize_tool_arguments(
            "fs",
            &serde_json::json!({"path": "a.txt", "content": "hi"}),
        );
        assert_eq!(write["op"], "write");

        let unwrapped = ToolRegistry::normalize_tool_arguments(
            "echo",
            &serde_json::json!({"raw": "{\"text\":\"hello\"}", "meta": 1}),
        );
        assert_eq!(unwrapped["text"], "hello");
        assert!(unwrapped.get("raw").is_none());
        assert_eq!(unwrapped["meta"], 1);
    }

    #[test]
    fn normalize_tool_arguments_parses_relaxed_string_payloads() {
        let from_json = ToolRegistry::normalize_tool_arguments(
            "echo",
            &serde_json::Value::String(r#"{"text":"hi"}"#.into()),
        );
        assert_eq!(from_json["text"], "hi");

        let from_kv = ToolRegistry::normalize_tool_arguments(
            "fs",
            &serde_json::Value::String("path: /tmp/a.txt, op: read".into()),
        );
        assert_eq!(from_kv["path"], "/tmp/a.txt");
        assert_eq!(from_kv["op"], "read");
    }

    #[test]
    fn validate_tool_call_rejects_missing_required_params() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(RequiredParamTool));

        let err = registry
            .validate_tool_call("echo", &serde_json::json!({}))
            .expect_err("missing required param");
        assert!(err.to_string().contains("text"));

        registry
            .validate_tool_call("echo", &serde_json::json!({"text": "hi"}))
            .expect("valid call");
    }

    #[test]
    fn validate_tool_call_rejects_null_and_empty_required_params() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(RequiredParamTool));

        let null_err = registry
            .validate_tool_call("echo", &serde_json::json!({"text": null}))
            .expect_err("null required param");
        assert!(null_err.to_string().contains("null"));

        let empty_err = registry
            .validate_tool_call("echo", &serde_json::json!({"text": ""}))
            .expect_err("empty required param");
        assert!(empty_err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn execute_unknown_tool_returns_not_found() {
        let registry = ToolRegistry::new();
        let result = registry
            .execute("no_such_tool", serde_json::json!({}))
            .await;
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|e| e.contains("not found")
                    || e.contains("No such tool")
                    || e.contains("Unknown"))
                || result.content.to_lowercase().contains("not found")
                || result.content.to_lowercase().contains("unknown")
        );
    }
}
