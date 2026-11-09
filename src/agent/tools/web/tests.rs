use super::{
    WebFetchTool, WebSearchTool,
    error::WebError,
    provider::auto_fetch_providers,
    transport::{MockWebTransport, WebHttpResponse, WebHttpTransport},
    types::{
        DEFAULT_FIRECRAWL_API_URL, DEFAULT_TINYFISH_FETCH_API_URL, DEFAULT_TINYFISH_SEARCH_API_URL,
        PROVIDER_EXA, PROVIDER_FIRECRAWL, PROVIDER_HTTP, PROVIDER_TAVILY, PROVIDER_TINYFISH,
        WebProviderConfig,
    },
};
use crate::agent::tools::{Tool, ToolMeta};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn firecrawl_search_url() -> String {
    format!("{DEFAULT_FIRECRAWL_API_URL}/v2/search")
}

fn firecrawl_scrape_url() -> String {
    format!("{DEFAULT_FIRECRAWL_API_URL}/v2/scrape")
}

fn tinyfish_search_url(query: &str) -> String {
    let Ok(mut parsed) = url::Url::parse(DEFAULT_TINYFISH_SEARCH_API_URL) else {
        return format!("{DEFAULT_TINYFISH_SEARCH_API_URL}?query={query}");
    };
    parsed.query_pairs_mut().append_pair("query", query);
    parsed.to_string()
}

fn tinyfish_fetch_url() -> String {
    DEFAULT_TINYFISH_FETCH_API_URL.to_owned()
}

fn tavily_url() -> &'static str {
    "https://api.tavily.com/search"
}

fn exa_url() -> &'static str {
    "https://api.exa.ai/search"
}

fn search_tool(config: WebProviderConfig, transport: MockWebTransport) -> WebSearchTool {
    WebSearchTool::from_config(Arc::new(config), Arc::new(transport))
}

fn fetch_tool(config: WebProviderConfig, transport: MockWebTransport) -> WebFetchTool {
    WebFetchTool::from_config(Arc::new(config), Arc::new(transport))
}

fn firecrawl_search_body() -> Value {
    json!({
        "success": true,
        "data": {
            "web": [
                {
                    "url": "https://example.com/docs",
                    "title": "Example Docs",
                    "description": "Clean markdown extraction."
                }
            ]
        }
    })
}

#[tokio::test]
async fn web_search_requires_query() {
    let tool = search_tool(WebProviderConfig::default(), MockWebTransport::new());
    let result = tool.execute(json!({})).await;
    assert!(!result.success);
    assert!(
        result
            .error
            .as_deref()
            .is_some_and(|err| err.contains("query"))
    );
}

#[tokio::test]
async fn web_search_uses_firecrawl_keyless_when_no_keys() {
    let transport =
        MockWebTransport::new().with_post(firecrawl_search_url(), 200, firecrawl_search_body());
    let tool = search_tool(WebProviderConfig::default(), transport);
    let result = tool
        .execute(json!({ "query": "firecrawl markdown", "max_results": 3 }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_FIRECRAWL);
    assert_eq!(payload["results"][0]["url"], "https://example.com/docs");
    assert_eq!(
        payload["results"][0]["snippet"],
        "Clean markdown extraction."
    );
}

#[tokio::test]
async fn web_search_prefers_tavily_then_falls_back_to_firecrawl() {
    let tavily_ok = MockWebTransport::new().with_post(
        tavily_url(),
        200,
        json!({
            "answer": "Tavily summary",
            "results": [{
                "title": "Tavily Hit",
                "url": "https://tavily.example/page",
                "content": "snippet",
                "score": 0.9
            }]
        }),
    );
    let mut config = WebProviderConfig::default();
    config.tavily_api_key = Some("tvly-test".to_owned());
    let tool = search_tool(config.clone(), tavily_ok);
    let result = tool.execute(json!({ "query": "example" })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_TAVILY);
    assert_eq!(payload["answer"], "Tavily summary");

    let fallback = MockWebTransport::new()
        .with_post(tavily_url(), 401, json!({ "detail": "unauthorized" }))
        .with_post(firecrawl_search_url(), 200, firecrawl_search_body());
    let tool = search_tool(config, fallback);
    let result = tool.execute(json!({ "query": "example" })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert_eq!(
        result.typed_payload.expect("payload")["provider"],
        PROVIDER_FIRECRAWL
    );
}

#[tokio::test]
async fn web_fetch_uses_firecrawl_markdown() {
    let transport = MockWebTransport::new().with_post(
        firecrawl_scrape_url(),
        200,
        json!({
            "success": true,
            "data": {
                "markdown": "# Hello",
                "metadata": { "title": "Hello", "sourceURL": "https://example.com/hello" }
            }
        }),
    );
    let mut config = WebProviderConfig::default();
    config.firecrawl_api_key = Some("fc-test-key".to_owned());
    let tool = fetch_tool(config, transport);
    let result = tool
        .execute(json!({ "url": "https://example.com/hello" }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_FIRECRAWL);
    assert_eq!(payload["markdown"], "# Hello");
    assert_eq!(payload["title"], "Hello");
}

#[tokio::test]
async fn web_fetch_falls_back_to_http_extract() {
    let transport = MockWebTransport::new().with_get(
        "https://example.com/hello",
        "<html><body><h1>Hello</h1><p>World</p></body></html>",
    );
    let tool = fetch_tool(WebProviderConfig::default(), transport);
    let result = tool
        .execute(json!({ "url": "https://example.com/hello" }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_HTTP);
    assert!(
        payload["markdown"]
            .as_str()
            .is_some_and(|text| text.contains("Hello"))
    );
}

#[tokio::test]
async fn web_fetch_rejects_empty_url() {
    let tool = fetch_tool(WebProviderConfig::default(), MockWebTransport::new());
    let result = tool.execute(json!({ "url": "  " })).await;
    assert!(!result.success);
}

#[tokio::test]
async fn web_fetch_rejects_loopback_url() {
    let tool = fetch_tool(WebProviderConfig::default(), MockWebTransport::new());
    let result = tool
        .execute(json!({ "url": "http://127.0.0.1/secret" }))
        .await;
    assert!(!result.success);
    let error = result.error.unwrap_or_default();
    assert!(
        error.contains("127.0.0.1") || error.contains("blocked") || error.contains("private"),
        "{error}"
    );
}

#[tokio::test]
async fn web_tools_are_readonly() {
    let search = search_tool(WebProviderConfig::default(), MockWebTransport::new());
    let fetch = fetch_tool(WebProviderConfig::default(), MockWebTransport::new());
    assert!(search.is_readonly());
    assert!(fetch.is_readonly());
    assert!(!crate::agent::modes::is_write_tool(search.name()));
    assert!(!crate::agent::modes::is_write_tool(fetch.name()));
}

#[tokio::test]
async fn web_search_auto_order_with_tinyfish() {
    let query = "caliche";
    let tinyfish_body = json!({
        "query": query,
        "results": [{
            "position": 1,
            "title": "TinyFish Result",
            "url": "https://tinyfish.example/hit",
            "snippet": "TinyFish snippet"
        }],
        "total_results": 1,
        "page": 0
    });

    let mut config = WebProviderConfig::default();
    config.tinyfish_api_key = Some("tf-test-key".to_owned());

    // When only TinyFish is configured, auto chooses TinyFish over Firecrawl
    let transport = MockWebTransport::new().with_get_json(
        tinyfish_search_url(query),
        200,
        tinyfish_body.clone(),
    );
    let tool = search_tool(config.clone(), transport);
    let result = tool.execute(json!({ "query": query })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_TINYFISH);
    assert_eq!(payload["results"][0]["title"], "TinyFish Result");

    // When Tavily + Exa + TinyFish are configured, Tavily is preferred first
    let mut all_config = config.clone();
    all_config.tavily_api_key = Some("tvly-key".to_owned());
    all_config.exa_api_key = Some("exa-key".to_owned());

    let transport_tavily = MockWebTransport::new()
        .with_post(
            tavily_url(),
            200,
            json!({
                "results": [{
                    "title": "Tavily Hit",
                    "url": "https://tavily.example/hit",
                    "content": "Tavily snippet"
                }]
            }),
        )
        .with_get_json(tinyfish_search_url(query), 200, tinyfish_body.clone());
    let tool_tavily = search_tool(all_config.clone(), transport_tavily);
    let result_tavily = tool_tavily.execute(json!({ "query": query })).await;
    assert!(result_tavily.success);
    assert_eq!(
        result_tavily.typed_payload.expect("payload")["provider"],
        PROVIDER_TAVILY
    );

    // If Tavily fails, Exa is tried before TinyFish
    let transport_exa = MockWebTransport::new()
        .with_post(tavily_url(), 500, json!({ "detail": "tavily down" }))
        .with_post(
            exa_url(),
            200,
            json!({
                "results": [{
                    "title": "Exa Hit",
                    "url": "https://exa.example/hit",
                    "text": "Exa snippet"
                }]
            }),
        )
        .with_get_json(tinyfish_search_url(query), 200, tinyfish_body.clone());
    let tool_exa = search_tool(all_config.clone(), transport_exa);
    let result_exa = tool_exa.execute(json!({ "query": query })).await;
    assert!(result_exa.success);
    assert_eq!(
        result_exa.typed_payload.expect("payload")["provider"],
        PROVIDER_EXA
    );

    // If both Tavily and Exa fail, TinyFish is tried before Firecrawl
    let transport_tinyfish = MockWebTransport::new()
        .with_post(tavily_url(), 500, json!({ "detail": "tavily down" }))
        .with_post(exa_url(), 500, json!({ "error": "exa down" }))
        .with_get_json(tinyfish_search_url(query), 200, tinyfish_body.clone())
        .with_post(firecrawl_search_url(), 200, firecrawl_search_body());
    let tool_tf = search_tool(all_config.clone(), transport_tinyfish);
    let result_tf = tool_tf.execute(json!({ "query": query })).await;
    assert!(result_tf.success);
    assert_eq!(
        result_tf.typed_payload.expect("payload")["provider"],
        PROVIDER_TINYFISH
    );

    // If TinyFish also fails, fallback to Firecrawl
    let transport_fc = MockWebTransport::new()
        .with_post(tavily_url(), 500, json!({ "detail": "tavily down" }))
        .with_post(exa_url(), 500, json!({ "error": "exa down" }))
        .with_get_json(
            tinyfish_search_url(query),
            401,
            json!({ "error": { "code": "INVALID_API_KEY", "message": "invalid key" } }),
        )
        .with_post(firecrawl_search_url(), 200, firecrawl_search_body());
    let tool_fc = search_tool(all_config, transport_fc);
    let result_fc = tool_fc.execute(json!({ "query": query })).await;
    assert!(result_fc.success);
    assert_eq!(
        result_fc.typed_payload.expect("payload")["provider"],
        PROVIDER_FIRECRAWL
    );
}

#[tokio::test]
async fn web_search_explicit_tinyfish_provider() {
    let query = "rust tokio";
    let transport = MockWebTransport::new().with_get_json(
        tinyfish_search_url(query),
        200,
        json!({
            "query": query,
            "results": [
                {
                    "position": 1,
                    "title": "Tokio Docs",
                    "url": "https://tokio.rs",
                    "snippet": "Async runtime for Rust"
                },
                {
                    "position": 2,
                    "title": "Tokio Crate",
                    "url": "https://crates.io/crates/tokio",
                    "snippet": "Crates.io page"
                }
            ],
            "total_results": 2,
            "page": 0
        }),
    );
    let mut config = WebProviderConfig::default();
    config.tinyfish_api_key = Some("tf-key".to_owned());
    let tool = search_tool(config, transport);
    let result = tool
        .execute(json!({ "query": query, "provider": "tinyfish", "max_results": 1 }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_TINYFISH);
    assert_eq!(payload["results"].as_array().map(Vec::len), Some(1));
    assert_eq!(payload["results"][0]["url"], "https://tokio.rs");
}

#[tokio::test]
async fn web_fetch_auto_order_with_tinyfish() {
    let target_url = "https://example.com/article";
    let tinyfish_fetch_body = json!({
        "results": [{
            "url": target_url,
            "final_url": "https://example.com/article",
            "title": "TinyFish Extracted Article",
            "text": "# TinyFish Article\n\nContent here.",
            "format": "markdown"
        }],
        "errors": []
    });

    let mut config = WebProviderConfig::default();
    config.tinyfish_api_key = Some("tf-test-key".to_owned());
    let mut firecrawl_config = config.clone();
    firecrawl_config.firecrawl_api_key = Some("fc-test-key".to_owned());

    // Auto prefers TinyFish over Firecrawl and HTTP when tinyfish key is set
    let transport =
        MockWebTransport::new().with_post(tinyfish_fetch_url(), 200, tinyfish_fetch_body.clone());
    let tool = fetch_tool(config.clone(), transport);
    let result = tool.execute(json!({ "url": target_url })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_TINYFISH);
    assert_eq!(payload["title"], "TinyFish Extracted Article");
    assert_eq!(payload["markdown"], "# TinyFish Article\n\nContent here.");

    // If TinyFish fails, fallback to Firecrawl
    let fallback_firecrawl = MockWebTransport::new()
        .with_post(
            tinyfish_fetch_url(),
            500,
            json!({ "error": { "code": "INTERNAL_ERROR", "message": "server error" } }),
        )
        .with_post(
            firecrawl_scrape_url(),
            200,
            json!({
                "success": true,
                "data": {
                    "markdown": "# Firecrawl Fallback",
                    "metadata": { "title": "Firecrawl Fallback", "sourceURL": target_url }
                }
            }),
        );
    let tool_fc = fetch_tool(firecrawl_config.clone(), fallback_firecrawl);
    let result_fc = tool_fc.execute(json!({ "url": target_url })).await;
    assert!(result_fc.success, "{}", result_fc.error.unwrap_or_default());
    assert_eq!(
        result_fc.typed_payload.expect("payload")["provider"],
        PROVIDER_FIRECRAWL
    );

    // If TinyFish and Firecrawl both fail, fallback to HTTP
    let fallback_http = MockWebTransport::new()
        .with_post(
            tinyfish_fetch_url(),
            500,
            json!({ "error": { "message": "failed" } }),
        )
        .with_post(
            firecrawl_scrape_url(),
            429,
            json!({ "error": "rate limited" }),
        )
        .with_get(
            target_url,
            "<html><body><h1>HTTP Fallback</h1><p>Cleaned</p></body></html>",
        );
    let tool_http = fetch_tool(firecrawl_config, fallback_http);
    let result_http = tool_http.execute(json!({ "url": target_url })).await;
    assert!(
        result_http.success,
        "{}",
        result_http.error.unwrap_or_default()
    );
    assert_eq!(
        result_http.typed_payload.expect("payload")["provider"],
        PROVIDER_HTTP
    );
}

#[tokio::test]
async fn web_fetch_explicit_tinyfish_provider() {
    let target_url = "https://example.com/page";
    let transport = MockWebTransport::new().with_post(
        tinyfish_fetch_url(),
        200,
        json!({
            "results": [{
                "url": target_url,
                "title": "Specific Page",
                "text": "Extracted text content"
            }],
            "errors": []
        }),
    );
    let mut config = WebProviderConfig::default();
    config.tinyfish_api_key = Some("tf-key".to_owned());
    let tool = fetch_tool(config, transport);
    let result = tool
        .execute(json!({ "url": target_url, "provider": "tinyfish" }))
        .await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_TINYFISH);
    assert_eq!(payload["markdown"], "Extracted text content");
}

#[test]
fn auto_fetch_order_requires_keys_before_third_parties() {
    assert_eq!(auto_fetch_providers(false, false), &[PROVIDER_HTTP]);
    assert_eq!(
        auto_fetch_providers(false, true),
        &[PROVIDER_FIRECRAWL, PROVIDER_HTTP]
    );
    assert_eq!(
        auto_fetch_providers(true, false),
        &[PROVIDER_TINYFISH, PROVIDER_HTTP]
    );
    assert_eq!(
        auto_fetch_providers(true, true),
        &[PROVIDER_TINYFISH, PROVIDER_FIRECRAWL, PROVIDER_HTTP]
    );
}

struct LocalFallbackTransport {
    addr: SocketAddr,
    posts: Mutex<Vec<String>>,
    gets: Mutex<Vec<String>>,
    tinyfish_ok: Option<Value>,
}

impl LocalFallbackTransport {
    fn new(addr: SocketAddr, tinyfish_ok: Option<Value>) -> Self {
        Self {
            addr,
            posts: Mutex::new(Vec::new()),
            gets: Mutex::new(Vec::new()),
            tinyfish_ok,
        }
    }

    fn posts(&self) -> Vec<String> {
        self.posts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }

    fn gets(&self) -> Vec<String> {
        self.gets
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }
}

fn record(slot: &Mutex<Vec<String>>, url: &str) {
    slot.lock()
        .unwrap_or_else(|err| err.into_inner())
        .push(url.to_owned());
}

fn spawn_html_server(html: &str) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local html server");
    let addr = listener.local_addr().expect("local html server addr");
    let html = html.to_owned();
    std::thread::spawn(move || {
        for _ in 0..8 {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut buf = [0_u8; 2048];
            let _ = stream.read(&mut buf);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                html.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(html.as_bytes());
            let _ = stream.flush();
        }
    });
    addr
}

fn read_local_page(addr: SocketAddr) -> Result<String, WebError> {
    let mut stream = TcpStream::connect(addr).map_err(|err| WebError::message(err.to_string()))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|err| WebError::message(err.to_string()))?;
    let request = format!("GET /page HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|err| WebError::message(err.to_string()))?;
    let mut raw = String::new();
    stream
        .read_to_string(&mut raw)
        .map_err(|err| WebError::message(err.to_string()))?;
    let page = match raw.find("\r\n\r\n") {
        Some(index) => raw[index + 4..].to_owned(),
        None => raw,
    };
    Ok(page)
}

#[async_trait]
impl WebHttpTransport for LocalFallbackTransport {
    async fn post_json(
        &self,
        url: &str,
        _headers: &[(&str, &str)],
        _body: &Value,
    ) -> Result<WebHttpResponse, WebError> {
        record(&self.posts, url);
        if url == tinyfish_fetch_url() {
            return match &self.tinyfish_ok {
                Some(body) => Ok(WebHttpResponse {
                    status: 200,
                    body: body.clone(),
                }),
                None => Err(WebError::message(
                    "tinyfish was called without a configured key response",
                )),
            };
        }
        if url == firecrawl_scrape_url() {
            return Ok(WebHttpResponse {
                status: 503,
                body: json!({ "error": "unavailable" }),
            });
        }
        Err(WebError::message(format!("unexpected POST {url}")))
    }

    async fn get_json(
        &self,
        url: &str,
        _headers: &[(&str, &str)],
    ) -> Result<WebHttpResponse, WebError> {
        record(&self.posts, url);
        Err(WebError::message(format!("unexpected GET JSON {url}")))
    }

    async fn get_text(&self, url: &str) -> Result<String, WebError> {
        record(&self.gets, url);
        let addr = self.addr;
        match tokio::task::spawn_blocking(move || read_local_page(addr)).await {
            Ok(result) => result,
            Err(err) => Err(WebError::message(err.to_string())),
        }
    }
}

const LOCAL_PAGE_HTML: &str = "<html><body><h1>Local Page</h1><p>Plain HTTP body</p></body></html>";
const PUBLIC_PAGE_URL: &str = "https://example.com/article";

/// Zero-key fetch falls through to plain HTTP. The requested URL stays public
/// because loopback is SSRF-blocked; the fallback transport reads a local server.
#[tokio::test]
async fn web_fetch_no_keys_uses_local_http_server() {
    let addr = spawn_html_server(LOCAL_PAGE_HTML);
    let transport = Arc::new(LocalFallbackTransport::new(addr, None));
    let tool = WebFetchTool::from_config(
        Arc::new(WebProviderConfig::default()),
        Arc::clone(&transport) as Arc<dyn WebHttpTransport>,
    );
    let result = tool.execute(json!({ "url": PUBLIC_PAGE_URL })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_HTTP);
    let markdown = payload["markdown"].as_str().unwrap_or_default();
    assert!(markdown.contains("Local Page"), "{markdown}");
    assert!(markdown.contains("Plain HTTP body"), "{markdown}");
    let posts = transport.posts();
    assert!(
        !posts.iter().any(|url| url.contains("tinyfish")),
        "missing TinyFish key must skip that provider: {posts:?}"
    );
    assert!(
        !posts.iter().any(|url| url.contains("firecrawl")),
        "missing Firecrawl key must not POST the URL: {posts:?}"
    );
    assert!(
        posts.is_empty(),
        "zero-key fetch stays on plain HTTP: {posts:?}"
    );
    assert!(!transport.gets().is_empty());
}

#[tokio::test]
async fn web_fetch_prefers_tinyfish_when_key_is_present() {
    let addr = spawn_html_server(LOCAL_PAGE_HTML);
    let transport = Arc::new(LocalFallbackTransport::new(
        addr,
        Some(json!({
            "results": [{
                "url": PUBLIC_PAGE_URL,
                "title": "TinyFish Preferred",
                "text": "# Preferred provider"
            }],
            "errors": []
        })),
    ));
    let mut config = WebProviderConfig::default();
    config.tinyfish_api_key = Some("tf-test-key".to_owned());
    let tool = WebFetchTool::from_config(
        Arc::new(config),
        Arc::clone(&transport) as Arc<dyn WebHttpTransport>,
    );
    let result = tool.execute(json!({ "url": PUBLIC_PAGE_URL })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    let payload = result.typed_payload.expect("payload");
    assert_eq!(payload["provider"], PROVIDER_TINYFISH);
    assert_eq!(payload["markdown"], "# Preferred provider");
    assert!(
        transport.gets().is_empty(),
        "preferred provider must not fall through to plain HTTP"
    );
    assert!(
        !transport
            .posts()
            .iter()
            .any(|url| url.contains("firecrawl")),
        "TinyFish success must not call Firecrawl"
    );
}

#[tokio::test]
async fn web_fetch_uses_firecrawl_when_key_is_present() {
    let transport = Arc::new(LocalFallbackTransport::new(
        spawn_html_server(LOCAL_PAGE_HTML),
        None,
    ));
    let mut config = WebProviderConfig::default();
    config.firecrawl_api_key = Some("fc-test-key".to_owned());
    // The local transport returns 503 for Firecrawl, so a keyed request must
    // attempt that POST before plain HTTP. A success path is covered by
    // `web_fetch_uses_firecrawl_markdown`.
    let tool = WebFetchTool::from_config(
        Arc::new(config),
        Arc::clone(&transport) as Arc<dyn WebHttpTransport>,
    );
    let result = tool.execute(json!({ "url": PUBLIC_PAGE_URL })).await;
    assert!(result.success, "{}", result.error.unwrap_or_default());
    assert_eq!(
        result.typed_payload.expect("payload")["provider"],
        PROVIDER_HTTP
    );
    assert!(
        transport
            .posts()
            .iter()
            .any(|url| url.contains("firecrawl")),
        "FIRECRAWL_API_KEY must contact Firecrawl before plain HTTP"
    );
}

#[tokio::test]
async fn web_fetch_explicit_firecrawl_without_key_is_unavailable() {
    let transport = MockWebTransport::new().with_post(
        firecrawl_scrape_url(),
        200,
        json!({
            "success": true,
            "data": { "markdown": "# Should Not Run" }
        }),
    );
    let tool = fetch_tool(WebProviderConfig::default(), transport);
    let result = tool
        .execute(json!({ "url": PUBLIC_PAGE_URL, "provider": "firecrawl" }))
        .await;
    assert!(!result.success);
    let error = result.error.unwrap_or_default();
    assert!(
        error.contains("not configured") || error.contains("firecrawl"),
        "{error}"
    );
}

#[tokio::test]
async fn web_fetch_explicit_tinyfish_without_key_is_unavailable() {
    let transport = MockWebTransport::new();
    let tool = fetch_tool(WebProviderConfig::default(), transport);
    let result = tool
        .execute(json!({ "url": PUBLIC_PAGE_URL, "provider": "tinyfish" }))
        .await;
    assert!(!result.success);
    let error = result.error.unwrap_or_default();
    assert!(
        error.contains("not configured") || error.contains("tinyfish"),
        "{error}"
    );
}
