#![expect(
    clippy::expect_used,
    reason = "static Regex/LazyLock literals are reviewed constants; failure is a programmer bug"
)]

use crate::agent::tools::common::{http, params, schema, text};
use crate::agent::tools::credentials::{CredentialResolver, ToolCredential};
use crate::agent::tools::http::{DomainProbeCache, ensure_domain_probed, new_probe_cache};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::utils::user_context::UserContext;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Cache entry type: (`response_body`, `cached_at_timestamp`)
type CacheEntry = (String, chrono::DateTime<chrono::Utc>);
/// Cache storage type
type ResponseCache = Arc<RwLock<HashMap<String, CacheEntry>>>;

/// Unified HTTP tool
#[derive(Debug)]
pub struct HttpTool {
    /// Optional default base URL for API calls
    default_base_url: Option<String>,
    /// Default headers to include in all requests
    default_headers: HashMap<String, String>,
    /// Optional response cache
    cache: Option<ResponseCache>,
    /// Cache TTL in seconds
    cache_ttl_seconds: u64,
    /// Construction-time credentials (highest priority)
    credentials: Option<ToolCredential>,
    /// Per-domain llms.txt / llm.txt probe cache; probed once per hostname.
    domain_probe_cache: DomainProbeCache,
}

impl HttpTool {
    /// Create a new `HttpTool` without credentials
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_base_url: None,
            default_headers: HashMap::new(),
            cache: Some(Arc::new(RwLock::new(HashMap::new()))),
            cache_ttl_seconds: 300, // 5 minutes default
            credentials: None,
            domain_probe_cache: new_probe_cache(),
        }
    }

    /// Create a new `HttpTool` with construction-time credentials
    #[must_use]
    pub fn with_credentials(credentials: ToolCredential) -> Self {
        Self {
            default_base_url: None,
            default_headers: HashMap::new(),
            cache: Some(Arc::new(RwLock::new(HashMap::new()))),
            cache_ttl_seconds: 300,
            credentials: Some(credentials),
            domain_probe_cache: new_probe_cache(),
        }
    }

    /// Set default base URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.default_base_url = Some(base_url.into());
        self
    }

    /// Set default headers
    #[must_use]
    pub fn with_default_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.default_headers = headers;
        self
    }

    /// Set cache TTL
    #[must_use]
    pub const fn with_cache_ttl(mut self, ttl_seconds: u64) -> Self {
        self.cache_ttl_seconds = ttl_seconds;
        self
    }

    /// Disable caching
    #[must_use]
    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

const NAME: &str = "http";
const DESCRIPTION: &str =
    "Make HTTP requests with custom headers, authentication, and query parameters.";

impl ToolMeta for HttpTool {
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
        let mut props = json!({
            "operation": {
                "type": "string",
                "enum": ["request", "fetch_html", "api_call"],
                "description": "Operation mode: 'request' for general HTTP calls, 'fetch_html' for HTML content extraction, 'api_call' for REST API calls with caching",
                "default": "request"
            },
            "url": {
                "type": "string",
                "description": "The URL to make the request to (must be http:// or https://). For 'api_call', can use 'endpoint' with base_url instead."
            },
            "base_url": {
                "type": "string",
                "description": "Base URL for API calls (optional if default_base_url is configured, used with 'endpoint' parameter)"
            },
            "endpoint": {
                "type": "string",
                "description": "API endpoint path (e.g., '/v1/users', used with 'api_call' operation and 'base_url')"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                "description": "HTTP method to use (default: GET)",
                "default": "GET"
            },
            "headers": {
                "type": "object",
                "description": "Custom HTTP headers as key-value pairs"
            },
            "body": {
                "type": ["string", "object"],
                "description": "Request body. Can be a JSON object (will be serialized), form data object, or raw string."
            },
            "body_type": {
                "type": "string",
                "enum": ["json", "form", "text", "raw"],
                "description": "Type of request body: 'json' (default) serializes object to JSON, 'form' sends as form-urlencoded, 'text' sends as plain text, 'raw' sends body as-is",
                "default": "json"
            },
            "query_params": {
                "type": "object",
                "description": "Query parameters as key-value pairs"
            },
            "follow_redirects": {
                "type": "boolean",
                "description": "Whether to follow redirects (default: true)",
                "default": true
            },
            "forward_user_token": {
                "type": "boolean",
                "default": false,
                "description": "Forward JWT token from user context (if available, for 'api_call' operation)"
            },
            "cache_ttl_seconds": {
                "type": "integer",
                "default": 300,
                "description": "Cache TTL in seconds (0 to disable caching, used for 'api_call' operation)"
            }
        });

        // Add timeout parameter
        props["timeout"] = schema::timeout_param(30);
        props["max_length"] = schema::max_length_param(
            50_000,
            Some("Maximum length of cleaned text to return per URL (for 'fetch_html' operation)"),
        );

        // Merge credential parameters into the props object.
        // By pattern-matching on the owned `Value`, we can move each key-value
        // pair directly via `Map::extend` without any intermediate clones.
        let credential_params = crate::agent::tools::credential_parameters();
        if let (Some(props_obj), serde_json::Value::Object(cred_obj)) =
            (props.as_object_mut(), credential_params)
        {
            props_obj.extend(cred_obj);
        }

        json!({
            "type": "object",
            "properties": props,
            "required": []
        })
    }
}

#[async_trait]
impl Tool for HttpTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        self.execute_with_context(arguments, None).await
    }

    async fn execute_with_context(
        &self,
        arguments: serde_json::Value,
        context: Option<&UserContext>,
    ) -> ToolResult {
        let operation = params::extract_string_or(&arguments, "operation", "request");

        match operation.as_str() {
            "request" => self.handle_request(&arguments, context).await,
            "fetch_html" => self.handle_fetch_html(&arguments, context).await,
            "api_call" => self.handle_api_call(&arguments, context).await,
            _ => ToolResult::error(format!("Unknown operation: {operation}")),
        }
    }
}

impl HttpTool {
    fn reject_local_file_url(url: &str) -> Option<ToolResult> {
        if url.starts_with("file://") {
            return Some(ToolResult::error(
                "Local filesystem paths are not valid for the http tool. Use fs with op=read/list/grep for local files.",
            ));
        }
        None
    }

    fn reject_forbidden_headers(
        default_headers: &HashMap<String, String>,
        arguments: &serde_json::Value,
    ) -> Option<ToolResult> {
        for key in default_headers.keys() {
            if crate::security::http_policy::is_forbidden_egress_header(key) {
                return Some(ToolResult::error(format!("HTTP header not allowed: {key}")));
            }
        }
        if let Some(headers_obj) = arguments.get("headers").and_then(|v| v.as_object()) {
            for key in headers_obj.keys() {
                if crate::security::http_policy::is_forbidden_egress_header(key) {
                    return Some(ToolResult::error(format!("HTTP header not allowed: {key}")));
                }
            }
        }
        None
    }

    fn ssrf_client(
        url: &str,
        timeout_secs: u64,
        follow_redirects: bool,
    ) -> Result<reqwest::Client> {
        let policy = crate::gateway::egress::current_policy()?;
        let validated = policy.validate(crate::gateway::egress::EgressClass::Tool, url)?;
        let policy = policy.http_policy(crate::gateway::egress::EgressClass::Tool);
        let redirect = if follow_redirects {
            policy.ssrf_safe_redirect_policy()
        } else {
            reqwest::redirect::Policy::none()
        };
        validated
            .pinned_client_builder()?
            .timeout(Duration::from_secs(timeout_secs))
            .user_agent("Mozilla/5.0 (compatible; Caliche/1.0)")
            .redirect(redirect)
            .build()
            .context("Failed to create HTTP client")
    }

    /// Probe the domain for llms.txt / llm.txt and return warnings to surface to the agent.
    async fn probe_url_domain(&self, url: &str) -> Vec<String> {
        let result = ensure_domain_probed(&self.domain_probe_cache, url).await;
        result.as_warning().into_iter().collect()
    }

    /// Merge probe warnings into an existing `ToolResult`, preserving its success/failure status.
    fn with_probe_warnings(mut result: ToolResult, mut probe_warnings: Vec<String>) -> ToolResult {
        if probe_warnings.is_empty() {
            return result;
        }
        if let Some(ref mut existing) = result.warnings {
            existing.append(&mut probe_warnings);
        } else {
            result.warnings = Some(probe_warnings);
        }
        result
    }

    /// Handle general HTTP request operation
    async fn handle_request(
        &self,
        arguments: &serde_json::Value,
        context: Option<&UserContext>,
    ) -> ToolResult {
        let url_str = match params::extract_string(arguments, "url", None) {
            Ok(u) => u,
            Err(e) => return ToolResult::error(e),
        };
        if let Some(err) = Self::reject_forbidden_headers(&self.default_headers, arguments) {
            return err;
        }
        if let Some(err) = Self::reject_local_file_url(&url_str) {
            return err;
        }

        if let Err(e) = http::validate_http_url(&url_str) {
            return ToolResult::error(e);
        }

        let probe_warnings = self.probe_url_domain(&url_str).await;

        let method_str = params::extract_string_or(arguments, "method", "GET").to_uppercase();
        let timeout_secs = params::extract_number_or(arguments, "timeout", 30);
        let follow_redirects = params::extract_bool(arguments, "follow_redirects", true);
        let body_type = params::extract_string_or(arguments, "body_type", "json");

        let client = match Self::ssrf_client(&url_str, timeout_secs, follow_redirects) {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::error(format!("Failed to create HTTP client: {e}"));
            }
        };

        let method = match http_method_from_str(&method_str) {
            Ok(m) => m,
            Err(e) => return e,
        };

        let mut request_builder = client.request(method, &url_str);
        match apply_request_headers(request_builder, &self.default_headers, arguments) {
            Ok(rb) => request_builder = rb,
            Err(e) => return e,
        }

        let resolved_credential =
            CredentialResolver::resolve(self.credentials.as_ref(), arguments, context);
        if let Some(credential) = resolved_credential {
            match credential.apply_to_request(request_builder) {
                Ok(rb) => request_builder = rb,
                Err(e) => return ToolResult::error(format!("Failed to apply credentials: {e}")),
            }
        }

        request_builder = apply_query_params(request_builder, arguments);
        match apply_request_body(request_builder, arguments, &body_type) {
            Ok(rb) => request_builder = rb,
            Err(e) => return e,
        }
        request_builder = apply_default_content_type(request_builder, arguments, &body_type);

        match request_builder.send().await {
            Ok(response) => Self::format_http_response(response, &url_str, probe_warnings).await,
            Err(e) => ToolResult::error(format!("HTTP request failed: {e}")),
        }
    }

    async fn format_http_response(
        response: reqwest::Response,
        url_str: &str,
        probe_warnings: Vec<String>,
    ) -> ToolResult {
        let status = response.status();
        let status_code = status.as_u16();
        let status_text = status.canonical_reason().unwrap_or("Unknown");
        let mut headers_map = HashMap::new();
        for (key, value) in response.headers() {
            headers_map.insert(
                key.to_string(),
                value.to_str().unwrap_or("<invalid>").to_string(),
            );
        }
        let body_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                return ToolResult::error(format!("Failed to read response body: {e}"));
            }
        };
        let mut output = format!("HTTP {status_code} {status_text}\nURL: {url_str}\n\n");
        if !headers_map.is_empty() {
            output.push_str("Response Headers:\n");
            for (key, value) in &headers_map {
                output.push_str(&format!("  {key}: {value}\n"));
            }
            output.push('\n');
        }
        output.push_str("Response Body:\n");
        output.push_str(&body_text);
        if status.is_success() {
            Self::with_probe_warnings(ToolResult::success(output), probe_warnings)
        } else {
            ToolResult::error(format!(
                "HTTP {status_code} {status_text} - {url_str}\n\nResponse Body:\n{body_text}"
            ))
        }
    }

    /// Handle fetch HTML operation (from http)
    async fn handle_fetch_html(
        &self,
        arguments: &serde_json::Value,
        context: Option<&UserContext>,
    ) -> ToolResult {
        let urls: Vec<String> = if let Some(url_array) = arguments["urls"].as_array() {
            url_array
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        } else if let Some(url_str) = arguments["url"].as_str() {
            vec![url_str.to_owned()]
        } else if let Some(url_str) = arguments["urls"].as_str() {
            vec![url_str.to_owned()]
        } else {
            return ToolResult::error("'url' or 'urls' parameter must be provided");
        };

        if urls.is_empty() {
            return ToolResult::error("At least one URL is required");
        }
        if let Some(local_url) = urls.iter().find(|u| u.starts_with("file://")) {
            return ToolResult::error(format!(
                "Local filesystem URL '{local_url}' is not valid for http fetch_html. Use fs with op=read/list/grep."
            ));
        }

        let timeout_secs = params::extract_number_or(arguments, "timeout", 10);
        let max_length = params::extract_number_or(arguments, "max_length", 50_000);

        // Probe each unique domain once before spawning fetches.
        let mut probe_warnings: Vec<String> = Vec::new();
        let mut probed_domains = HashSet::new();
        for url in &urls {
            if let Some(domain) = crate::agent::tools::http::extract_domain(url)
                && probed_domains.insert(domain)
            {
                probe_warnings.extend(self.probe_url_domain(url).await);
            }
        }

        let resolved_credential =
            CredentialResolver::resolve(self.credentials.as_ref(), arguments, context);

        let mut handles = Vec::new();
        for url in urls {
            let url_clone = url.clone();
            let cred_clone = resolved_credential.clone();
            let handle = tokio::spawn(async move {
                Self::fetch_and_process_url(
                    &url_clone,
                    timeout_secs,
                    max_length,
                    cred_clone.as_ref(),
                )
                .await
            });
            handles.push((url, handle));
        }

        let mut results = Vec::new();
        let mut all_found_urls = HashSet::new();

        for (original_url, handle) in handles {
            match handle.await {
                Ok(Ok((text, urls))) => {
                    all_found_urls.extend(urls.iter().cloned());
                    results.push(format!(
                        "URL: {}\nText ({} chars): {}\nFound {} links",
                        original_url,
                        text.len(),
                        if text.len() > max_length as usize {
                            caliche_contracts::text::truncate_utf8(&text, max_length as usize)
                        } else {
                            &text
                        },
                        urls.len()
                    ));
                }
                Ok(Err(e)) => {
                    results.push(format!("URL: {original_url}\nError: {e}"));
                }
                Err(e) => {
                    results.push(format!("URL: {original_url}\nError: Task join error: {e}"));
                }
            }
        }

        let mut output = results.join("\n\n---\n\n");

        if !all_found_urls.is_empty() {
            let mut url_list: Vec<String> = all_found_urls.into_iter().collect();
            url_list.sort();
            output.push_str("\n\n=== Discovered URLs ===\n");
            for url in url_list.iter().take(50) {
                output.push_str(&format!("- {url}\n"));
            }
            if url_list.len() > 50 {
                output.push_str(&format!("... and {} more URLs\n", url_list.len() - 50));
            }
        }

        Self::with_probe_warnings(ToolResult::success(output), probe_warnings)
    }

    /// Handle API call operation (from `rest_api_client`)
    async fn handle_api_call(
        &self,
        arguments: &serde_json::Value,
        context: Option<&UserContext>,
    ) -> ToolResult {
        let endpoint = match arguments.get("endpoint").and_then(|v| v.as_str()) {
            Some(e) => e,
            None => {
                return ToolResult::error("Missing required parameter: endpoint");
            }
        };

        let base_url = match arguments
            .get("base_url")
            .and_then(|v| v.as_str())
            .or(self.default_base_url.as_deref())
        {
            Some(url) => url,
            None => {
                return ToolResult::error(
                    "Missing base_url parameter and no default_base_url configured".to_string(),
                );
            }
        };

        let url = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("{}{endpoint}", base_url.trim_end_matches('/'))
        };

        let method = arguments
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET")
            .to_uppercase();

        let cache_ttl = arguments
            .get("cache_ttl_seconds")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(self.cache_ttl_seconds);

        // Check cache for GET requests
        if method == "GET"
            && cache_ttl > 0
            && let Some(ref cache) = self.cache
        {
            let cache_key = format!("{method}:{url}");
            let cache_guard = cache.read().await;
            if let Some((cached_response, cached_at)) = cache_guard.get(&cache_key) {
                let age = chrono::Utc::now()
                    .signed_duration_since(*cached_at)
                    .num_seconds();
                if age < cache_ttl as i64 {
                    return ToolResult::success_with_warnings(
                        cached_response.clone(),
                        vec!["Response served from cache".to_string()],
                    );
                }
            }
        }

        // Build request arguments for handle_request
        let mut request_args = json!({
            "url": url,
            "method": method,
        });

        if let Some(headers) = arguments.get("headers") {
            request_args["headers"] = headers.clone();
        }
        if let Some(body) = arguments.get("body") {
            request_args["body"] = body.clone();
        }
        if let Some(body_type) = arguments.get("body_type") {
            request_args["body_type"] = body_type.clone();
        }
        if let Some(query_params) = arguments.get("query_params") {
            request_args["query_params"] = query_params.clone();
        }
        if let Some(timeout) = arguments.get("timeout") {
            request_args["timeout"] = timeout.clone();
        }

        let probe_warnings = self.probe_url_domain(&url).await;

        let mut result = self.handle_request(&request_args, context).await;
        result = Self::with_probe_warnings(result, probe_warnings);

        // Cache successful GET responses
        if result.success
            && method == "GET"
            && cache_ttl > 0
            && let Some(ref cache) = self.cache
        {
            let cache_key = format!("{method}:{url}");
            let mut cache_guard = cache.write().await;
            cache_guard.insert(cache_key, (result.content.clone(), chrono::Utc::now()));
        }

        result
    }

    async fn fetch_and_process_url(
        url: &str,
        timeout_secs: u64,
        max_length: u64,
        credential: Option<&ToolCredential>,
    ) -> Result<(String, Vec<String>)> {
        http::validate_http_url(url)?;

        let client = Self::ssrf_client(url, timeout_secs, false)?;

        let mut request_builder = client.get(url);
        if let Some(cred) = credential {
            request_builder = cred
                .apply_to_request(request_builder)
                .context("Failed to apply credentials")?;
        }

        let response = request_builder
            .send()
            .await
            .with_context(|| format!("Failed to fetch URL: {url}"))?;

        if !response.status().is_success() {
            bail!("HTTP error: {}", response.status());
        }

        let html = response
            .text()
            .await
            .context("Failed to read response body")?;

        let found_urls = Self::extract_urls(&html, url);
        let cleaned_text = text::clean_html_text(&html);

        let final_text = if cleaned_text.len() > max_length as usize {
            caliche_contracts::text::truncate_utf8(&cleaned_text, max_length as usize).to_string()
        } else {
            cleaned_text
        };

        Ok((final_text, found_urls))
    }

    fn extract_urls(html: &str, base_url: &str) -> Vec<String> {
        use std::sync::LazyLock;

        static HREF_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r#"(?i)href\s*=\s*["']([^"']+)["']"#)
                .expect("HREF_PATTERN: constant regex must compile")
        });
        static SRC_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r#"(?i)src\s*=\s*["']([^"']+)["']"#)
                .expect("SRC_PATTERN: constant regex must compile")
        });

        let mut urls = HashSet::new();

        for cap in HREF_PATTERN.captures_iter(html) {
            if let Some(url_str) = cap.get(1) {
                let url = url_str.as_str();
                if let Ok(resolved) = http::resolve_url(url, base_url)
                    && http::is_valid_http_url(&resolved)
                {
                    urls.insert(resolved);
                }
            }
        }

        for cap in SRC_PATTERN.captures_iter(html) {
            if let Some(url_str) = cap.get(1) {
                let url = url_str.as_str();
                if let Ok(resolved) = http::resolve_url(url, base_url)
                    && http::is_valid_http_url(&resolved)
                {
                    urls.insert(resolved);
                }
            }
        }

        urls.into_iter().collect()
    }
}

fn http_method_from_str(method_str: &str) -> Result<reqwest::Method, ToolResult> {
    match method_str {
        "GET" => Ok(reqwest::Method::GET),
        "POST" => Ok(reqwest::Method::POST),
        "PUT" => Ok(reqwest::Method::PUT),
        "DELETE" => Ok(reqwest::Method::DELETE),
        "PATCH" => Ok(reqwest::Method::PATCH),
        "HEAD" => Ok(reqwest::Method::HEAD),
        "OPTIONS" => Ok(reqwest::Method::OPTIONS),
        _ => Err(ToolResult::error(format!(
            "Unsupported HTTP method: {method_str}"
        ))),
    }
}

fn apply_request_headers(
    mut request_builder: reqwest::RequestBuilder,
    default_headers: &HashMap<String, String>,
    arguments: &serde_json::Value,
) -> Result<reqwest::RequestBuilder, ToolResult> {
    for (key, value) in default_headers {
        if crate::security::http_policy::is_forbidden_egress_header(key) {
            return Err(ToolResult::error(format!("HTTP header not allowed: {key}")));
        }
        request_builder = request_builder.header(key, value);
    }
    let Some(headers_obj) = arguments.get("headers").and_then(|v| v.as_object()) else {
        return Ok(request_builder);
    };
    for (key, value) in headers_obj {
        if crate::security::http_policy::is_forbidden_egress_header(key) {
            return Err(ToolResult::error(format!("HTTP header not allowed: {key}")));
        }
        if let Some(val_str) = value.as_str()
            && let Ok(header_name) = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            && let Ok(header_value) = reqwest::header::HeaderValue::from_str(val_str)
        {
            request_builder = request_builder.header(header_name, header_value);
        }
    }
    Ok(request_builder)
}

fn apply_query_params(
    mut request_builder: reqwest::RequestBuilder,
    arguments: &serde_json::Value,
) -> reqwest::RequestBuilder {
    let Some(query_obj) = arguments.get("query_params").and_then(|v| v.as_object()) else {
        return request_builder;
    };
    for (key, value) in query_obj {
        if let Some(val_str) = value.as_str() {
            request_builder = request_builder.query(&[(key, val_str)]);
        } else {
            request_builder = request_builder.query(&[(key, &value.to_string())]);
        }
    }
    request_builder
}

fn apply_request_body(
    request_builder: reqwest::RequestBuilder,
    arguments: &serde_json::Value,
    body_type: &str,
) -> Result<reqwest::RequestBuilder, ToolResult> {
    let Some(body) = arguments.get("body") else {
        return Ok(request_builder);
    };
    match body_type {
        "json" => Ok(request_builder.json(body)),
        "form" => apply_form_body(request_builder, body),
        "text" | "raw" => {
            let body_str = body
                .as_str()
                .map_or_else(|| body.to_string(), str::to_owned);
            Ok(request_builder.body(body_str))
        }
        _ => Err(ToolResult::error(format!("Unknown body_type: {body_type}"))),
    }
}

fn apply_form_body(
    request_builder: reqwest::RequestBuilder,
    body: &serde_json::Value,
) -> Result<reqwest::RequestBuilder, ToolResult> {
    if let Some(form_obj) = body.as_object() {
        let mut form_data = HashMap::new();
        for (key, value) in form_obj {
            let val_str = value
                .as_str()
                .map_or_else(|| value.to_string(), str::to_owned);
            form_data.insert(key.clone(), val_str);
        }
        Ok(request_builder.form(&form_data))
    } else if body.is_string() {
        Err(ToolResult::error(
            "Form body must be an object with key-value pairs",
        ))
    } else {
        Ok(request_builder)
    }
}

fn apply_default_content_type(
    request_builder: reqwest::RequestBuilder,
    arguments: &serde_json::Value,
    body_type: &str,
) -> reqwest::RequestBuilder {
    if arguments.get("body").is_none()
        || arguments
            .get("headers")
            .and_then(|h| h.get("Content-Type"))
            .is_some()
    {
        return request_builder;
    }
    match body_type {
        "json" => request_builder.header("Content-Type", "application/json"),
        "form" => request_builder.header("Content-Type", "application/x-www-form-urlencoded"),
        "text" => request_builder.header("Content-Type", "text/plain"),
        _ => request_builder,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn custom_host_header_is_rejected() {
        let tool = HttpTool::new();
        let result = tool
            .execute(json!({
                "url": "https://example.invalid/health",
                "headers": { "Host": "169.254.169.254" }
            }))
            .await;
        assert!(!result.success, "Host header must fail closed");
        let err = result.error.as_deref().unwrap_or_default();
        assert!(
            err.to_lowercase().contains("header"),
            "unexpected error: {err}"
        );
    }
}
