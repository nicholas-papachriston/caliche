//! Provider-backed JSON-RPC method router for reusable MCP servers.
//!
//! Hosts implement [`crate::providers`] traits and call [`ProviderRouter::handle`].
//! Transports stay outside this crate so the dispatch loop stays lean and reusable.

use crate::pagination::{DEFAULT_PAGE_SIZE, PaginationError, cmp_name, paginate_owned};
use crate::protocol::{
    ContentPart, JsonRpcError, JsonRpcRequest, JsonRpcResponse, McpPrompt, McpResource,
    PromptsCapability, PromptsGetParams, PromptsGetResult, PromptsListParams, PromptsListResult,
    ResourceTemplate, ResourceTemplatesListParams, ResourceTemplatesListResult,
    ResourcesCapability, ResourcesListParams, ResourcesListResult, ResourcesReadParams,
    ResourcesReadResult, ServerCapabilities, ServerInfo, ToolsCallParams, ToolsCallResult,
    ToolsCapability, ToolsListParams, ToolsListResult,
};
use crate::providers::{PromptProvider, ResourceProvider, ToolProvider};
use crate::revision::{
    admit, cache_hint, cancelled_request_key, listen_response, request_cancelled, request_id_key,
    shape_result,
};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

/// Configuration for [`ProviderRouter`].
#[derive(Debug, Clone)]
pub struct ProviderRouterConfig {
    pub server_info: ServerInfo,
    pub instructions: Option<String>,
    pub page_size: usize,
    pub advertise_tools: bool,
    pub advertise_prompts: bool,
    pub advertise_resources: bool,
}

impl ProviderRouterConfig {
    pub fn new(server_info: ServerInfo) -> Self {
        Self {
            server_info,
            instructions: None,
            page_size: DEFAULT_PAGE_SIZE,
            advertise_tools: true,
            advertise_prompts: true,
            advertise_resources: true,
        }
    }
}

/// Engine-free MCP method router backed by provider traits.
pub struct ProviderRouter<T, P, R> {
    tools: Arc<T>,
    prompts: Arc<P>,
    resources: Arc<R>,
    config: ProviderRouterConfig,
    cancelled: HashSet<String>,
}

impl<T, P, R> ProviderRouter<T, P, R>
where
    T: ToolProvider + 'static,
    P: PromptProvider + 'static,
    R: ResourceProvider + 'static,
{
    pub fn new(
        tools: Arc<T>,
        prompts: Arc<P>,
        resources: Arc<R>,
        config: ProviderRouterConfig,
    ) -> Self {
        Self {
            tools,
            prompts,
            resources,
            config,
            cancelled: HashSet::new(),
        }
    }

    pub fn capabilities(&self) -> ServerCapabilities {
        ServerCapabilities {
            tools: self.config.advertise_tools.then_some(ToolsCapability {
                list_changed: Some(false),
            }),
            prompts: self.config.advertise_prompts.then_some(PromptsCapability {
                list_changed: Some(false),
            }),
            resources: self
                .config
                .advertise_resources
                .then_some(ResourcesCapability {
                    subscribe: Some(false),
                    list_changed: Some(false),
                }),
            logging: None,
            completions: None,
            tasks: None,
            experimental: None,
        }
    }

    /// Handle one JSON-RPC request. Returns `None` for notifications.
    pub async fn handle(&mut self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        if request.jsonrpc.as_ref() != "2.0" {
            return Some(JsonRpcResponse::error(
                request.id,
                JsonRpcError::invalid_request(Some(Value::String(
                    "Invalid jsonrpc version".into(),
                ))),
            ));
        }

        if request.id.is_none() {
            self.record_cancel(&request);
            return None;
        }

        if self.take_cancelled(request.id.as_ref()) {
            return Some(JsonRpcResponse::error(request.id, request_cancelled()));
        }

        if let Err(err) = admit(request.method.as_ref(), request.params.as_ref()) {
            return Some(JsonRpcResponse::error(request.id, err));
        }
        let method = request.method.clone();
        let method_name = method.as_ref();
        let hint = cache_hint(method_name);
        let response = if method_name == "server/discover" {
            self.discover(request)
        } else {
            self.dispatch_method(request).await
        };
        Some(shape_result(response, hint))
    }

    async fn dispatch_method(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_ref() {
            "tools/list" => self.tools_list(request).await,
            "tools/call" => self.tools_call(request).await,
            "prompts/list" => self.prompts_list(request).await,
            "prompts/get" => self.prompts_get(request).await,
            "resources/list" => self.resources_list(request).await,
            "resources/templates/list" => self.resources_templates_list(request).await,
            "resources/read" => self.resources_read(request).await,
            "subscriptions/listen" => listen_response(request.id),
            other => JsonRpcResponse::error(request.id, JsonRpcError::method_not_found(other)),
        }
    }

    fn record_cancel(&mut self, request: &JsonRpcRequest) {
        if request.method.as_ref() != "notifications/cancelled" {
            return;
        }
        if let Some(key) = cancelled_request_key(request.params.as_ref()) {
            self.cancelled.insert(key);
        }
    }

    fn take_cancelled(&mut self, id: Option<&Value>) -> bool {
        let Some(id) = id else {
            return false;
        };
        self.cancelled.remove(&request_id_key(id))
    }

    fn discover(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let info = &self.config.server_info;
        let mut body = serde_json::json!({
            "supportedVersions": crate::protocol::SUPPORTED_PROTOCOL_VERSIONS,
            "capabilities": self.capabilities(),
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": info.name.as_ref(),
                    "version": info.version.as_deref(),
                }
            }
        });
        if let Some(instructions) = &self.config.instructions
            && let Some(map) = body.as_object_mut()
        {
            map.insert(
                "instructions".to_string(),
                Value::String(instructions.clone()),
            );
        }
        JsonRpcResponse::success(request.id, body)
    }

    async fn tools_list(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: Option<ToolsListParams> =
            request.params.and_then(|v| serde_json::from_value(v).ok());
        let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
        match self.tools.list_tools().await {
            Ok(descriptors) => {
                let mut tools: Vec<_> = descriptors.into_iter().map(|d| d.tool).collect();
                tools.sort_by(|a, b| cmp_name(a.name.as_ref(), b.name.as_ref()));
                match paginate_owned(tools, cursor, Some(self.config.page_size)) {
                    Ok(page) => respond_ok(
                        request.id,
                        ToolsListResult {
                            tools: page.items,
                            next_cursor: page.next_cursor,
                        },
                    ),
                    Err(err) => map_page_err(request.id, err),
                }
            }
            Err(e) => internal(request.id, e),
        }
    }

    async fn tools_call(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: ToolsCallParams = match parse_required(request.params) {
            Ok(p) => p,
            Err(err) => return JsonRpcResponse::error(request.id, err),
        };
        let args = params
            .arguments
            .unwrap_or_else(|| Value::Object(Default::default()));
        match self.tools.call_tool(&params.name, args).await {
            Ok(result) => {
                let text = result
                    .content
                    .into_iter()
                    .find_map(|part| match part {
                        ContentPart::Text { text } => Some(text),
                        _ => None,
                    })
                    .unwrap_or_default();
                respond_ok(
                    request.id,
                    ToolsCallResult::from_text(text, result.structured_content, result.is_error),
                )
            }
            Err(e) => failure(request.id, e),
        }
    }

    async fn prompts_list(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: Option<PromptsListParams> =
            request.params.and_then(|v| serde_json::from_value(v).ok());
        let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
        match self.prompts.list_prompts().await {
            Ok(mut prompts) => {
                prompts.sort_by(|a: &McpPrompt, b: &McpPrompt| cmp_name(&a.name, &b.name));
                match paginate_owned(prompts, cursor, Some(self.config.page_size)) {
                    Ok(page) => respond_ok(
                        request.id,
                        PromptsListResult {
                            prompts: page.items,
                            next_cursor: page.next_cursor,
                        },
                    ),
                    Err(err) => map_page_err(request.id, err),
                }
            }
            Err(e) => internal(request.id, e),
        }
    }

    async fn prompts_get(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: PromptsGetParams = match parse_required(request.params) {
            Ok(p) => p,
            Err(err) => return JsonRpcResponse::error(request.id, err),
        };
        match self
            .prompts
            .get_prompt(&params.name, params.arguments)
            .await
        {
            Ok((messages, description)) => respond_ok(
                request.id,
                PromptsGetResult {
                    messages,
                    description,
                },
            ),
            Err(e) => failure(request.id, e),
        }
    }

    async fn resources_list(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: Option<ResourcesListParams> =
            request.params.and_then(|v| serde_json::from_value(v).ok());
        let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
        match self.resources.list_resources().await {
            Ok(mut resources) => {
                resources.sort_by(|a: &McpResource, b: &McpResource| {
                    cmp_name(
                        a.name.as_deref().unwrap_or(a.uri.as_str()),
                        b.name.as_deref().unwrap_or(b.uri.as_str()),
                    )
                });
                match paginate_owned(resources, cursor, Some(self.config.page_size)) {
                    Ok(page) => respond_ok(
                        request.id,
                        ResourcesListResult {
                            resources: page.items,
                            next_cursor: page.next_cursor,
                        },
                    ),
                    Err(err) => map_page_err(request.id, err),
                }
            }
            Err(e) => internal(request.id, e),
        }
    }

    async fn resources_templates_list(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: Option<ResourceTemplatesListParams> =
            request.params.and_then(|v| serde_json::from_value(v).ok());
        let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
        match self.resources.list_templates().await {
            Ok(mut templates) => {
                templates.sort_by(|a: &ResourceTemplate, b: &ResourceTemplate| {
                    cmp_name(
                        a.name.as_deref().unwrap_or(a.uri_template.as_str()),
                        b.name.as_deref().unwrap_or(b.uri_template.as_str()),
                    )
                });
                match paginate_owned(templates, cursor, Some(self.config.page_size)) {
                    Ok(page) => respond_ok(
                        request.id,
                        ResourceTemplatesListResult {
                            resource_templates: page.items,
                            next_cursor: page.next_cursor,
                        },
                    ),
                    Err(err) => map_page_err(request.id, err),
                }
            }
            Err(e) => internal(request.id, e),
        }
    }

    async fn resources_read(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let params: ResourcesReadParams = match parse_required(request.params) {
            Ok(p) => p,
            Err(err) => return JsonRpcResponse::error(request.id, err),
        };
        match self.resources.read_resource(&params.uri).await {
            Ok(contents) => respond_ok(request.id, ResourcesReadResult { contents }),
            Err(e) => failure(request.id, e),
        }
    }
}

fn parse_required<T: DeserializeOwned>(params: Option<Value>) -> Result<T, JsonRpcError> {
    match params {
        Some(Value::Object(obj)) => serde_json::from_value(Value::Object(obj))
            .map_err(|e| JsonRpcError::invalid_params(Some(Value::String(e.to_string())))),
        _ => Err(JsonRpcError::invalid_params(Some(Value::String(
            "Invalid params".into(),
        )))),
    }
}

fn respond_ok<T: serde::Serialize>(id: Option<Value>, value: T) -> JsonRpcResponse {
    match serde_json::to_value(value) {
        Ok(v) => JsonRpcResponse::success(id, v),
        Err(e) => JsonRpcResponse::error(
            id,
            JsonRpcError::internal_error(Some(Value::String(e.to_string()))),
        ),
    }
}

fn map_page_err(id: Option<Value>, err: PaginationError) -> JsonRpcResponse {
    JsonRpcResponse::error(
        id,
        JsonRpcError::invalid_params(Some(Value::String(err.to_string()))),
    )
}

fn internal(id: Option<Value>, err: impl std::fmt::Display) -> JsonRpcResponse {
    JsonRpcResponse::error(
        id,
        JsonRpcError::internal_error(Some(Value::String(err.to_string()))),
    )
}

fn failure(id: Option<Value>, err: impl std::fmt::Display) -> JsonRpcResponse {
    let text = err.to_string();
    if text.contains("not found") {
        return JsonRpcResponse::error(id, JsonRpcError::invalid_params(Some(Value::String(text))));
    }
    internal(id, text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::McpTool;
    use crate::providers::{NoPrompts, NoResources, ToolDescriptor, ToolInvokeResult};
    use async_trait::async_trait;
    use beef::lean::Cow;

    fn current_meta(mut body: Value) -> Value {
        let Some(map) = body.as_object_mut() else {
            return body;
        };
        map.insert(
            "_meta".to_string(),
            serde_json::json!({
                "io.modelcontextprotocol/protocolVersion": "2026-07-28"
            }),
        );
        body
    }

    struct FakeTools;

    #[async_trait]
    impl ToolProvider for FakeTools {
        async fn list_tools(&self) -> anyhow::Result<Vec<ToolDescriptor>> {
            Ok(vec![
                ToolDescriptor {
                    tool: McpTool::new("b", "second", serde_json::json!({"type":"object"})),
                },
                ToolDescriptor {
                    tool: McpTool::new("a", "first", serde_json::json!({"type":"object"})),
                },
            ])
        }

        async fn call_tool(
            &self,
            name: &str,
            _arguments: Value,
        ) -> anyhow::Result<ToolInvokeResult> {
            Ok(ToolInvokeResult::text_ok(format!("ok:{name}"))
                .with_structured(serde_json::json!({"name": name})))
        }
    }

    #[tokio::test]
    async fn paginated_tools_list() {
        let mut router = ProviderRouter::new(
            Arc::new(FakeTools),
            Arc::new(NoPrompts),
            Arc::new(NoResources),
            ProviderRouterConfig {
                page_size: 1,
                ..ProviderRouterConfig::new(ServerInfo {
                    name: Cow::borrowed("test"),
                    version: Some(Cow::borrowed("0.0.1")),
                    title: None,
                    description: None,
                    icons: None,
                })
            },
        );

        let list = JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("tools/list"),
            params: Some(current_meta(serde_json::json!({}))),
            id: Some(serde_json::json!(2)),
        };
        let list_resp = router.handle(list).await.expect("resp");
        let result = list_resp.result.expect("result");
        assert_eq!(result["resultType"], "complete");
        let page: ToolsListResult = serde_json::from_value(result).expect("page");
        assert_eq!(page.tools.len(), 1);
        assert_eq!(page.tools[0].name.as_ref(), "a");
        assert!(page.next_cursor.is_some());
    }

    #[tokio::test]
    async fn invalid_cursor_is_invalid_params() {
        let mut router = ProviderRouter::new(
            Arc::new(FakeTools),
            Arc::new(NoPrompts),
            Arc::new(NoResources),
            ProviderRouterConfig::new(ServerInfo {
                name: Cow::borrowed("test"),
                version: None,
                title: None,
                description: None,
                icons: None,
            }),
        );
        let resp = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("tools/list"),
                params: Some(current_meta(serde_json::json!({"cursor": "!!!"}))),
                id: Some(serde_json::json!(2)),
            })
            .await
            .expect("resp");
        let err = resp.error.expect("error");
        assert_eq!(err.code, -32602);
    }

    fn test_router() -> ProviderRouter<FakeTools, NoPrompts, NoResources> {
        ProviderRouter::new(
            Arc::new(FakeTools),
            Arc::new(NoPrompts),
            Arc::new(NoResources),
            ProviderRouterConfig::new(ServerInfo {
                name: Cow::borrowed("test"),
                version: Some(Cow::borrowed("0.0.1")),
                title: None,
                description: None,
                icons: None,
            }),
        )
    }

    #[tokio::test]
    async fn modern_client_lists_tools_without_initialize() {
        let mut router = test_router();
        let resp = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("tools/list"),
                params: Some(serde_json::json!({
                    "_meta": { "io.modelcontextprotocol/protocolVersion": "2026-07-28" }
                })),
                id: Some(serde_json::json!(1)),
            })
            .await
            .expect("resp");
        let result = resp.result.expect("result");
        assert_eq!(result["resultType"], "complete");
        assert_eq!(result["ttlMs"], 60_000);
        assert_eq!(result["cacheScope"], "private");
        assert!(
            result["tools"]
                .as_array()
                .is_some_and(|tools| tools.len() == 2)
        );
    }

    #[tokio::test]
    async fn discover_reports_supported_versions() {
        let mut router = test_router();
        let resp = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("server/discover"),
                params: None,
                id: Some(serde_json::json!(1)),
            })
            .await
            .expect("resp");
        let result = resp.result.expect("result");
        assert_eq!(result["resultType"], "complete");
        assert_eq!(result["cacheScope"], "public");
        let versions = result["supportedVersions"].as_array().expect("versions");
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0], "2026-07-28");
        assert_eq!(
            result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
            "test"
        );
    }

    #[tokio::test]
    async fn missing_resource_is_invalid_params() {
        let mut router = test_router();
        let resp = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("resources/read"),
                params: Some(current_meta(serde_json::json!({"uri": "missing://doc"}))),
                id: Some(serde_json::json!(2)),
            })
            .await
            .expect("resp");
        assert_eq!(resp.error.expect("error").code, -32602);
    }

    #[tokio::test]
    async fn subscriptions_listen_returns_an_id() {
        let mut router = test_router();
        let resp = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("subscriptions/listen"),
                params: Some(serde_json::json!({
                    "_meta": { "io.modelcontextprotocol/protocolVersion": "2026-07-28" }
                })),
                id: Some(serde_json::json!(4)),
            })
            .await
            .expect("resp");
        let result = resp.result.expect("result");
        assert_eq!(result["resultType"], "complete");
        assert_eq!(result["subscriptionId"], "4");
    }

    #[tokio::test]
    async fn cancel_before_start_skips_the_request() {
        let mut router = test_router();
        let note = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("notifications/cancelled"),
                params: Some(serde_json::json!({"requestId": 9})),
                id: None,
            })
            .await;
        assert!(note.is_none());
        let resp = router
            .handle(JsonRpcRequest {
                jsonrpc: Cow::borrowed("2.0"),
                method: Cow::borrowed("tools/list"),
                params: Some(serde_json::json!({
                    "_meta": { "io.modelcontextprotocol/protocolVersion": "2026-07-28" }
                })),
                id: Some(serde_json::json!(9)),
            })
            .await
            .expect("resp");
        assert_eq!(resp.error.expect("cancelled").code, -32800);
    }
}
