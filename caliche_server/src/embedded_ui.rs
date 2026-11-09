//! Embedded Dioxus web UI asset serving.

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "../caliche_web/dist"]
struct UiAssets;

const CACHE_CONTROL_IMMUTABLE: &str = "public, max-age=31536000, immutable";
const CACHE_CONTROL_NO_CACHE: &str = "no-cache";

/// Serve the embedded Dioxus web build, falling back to `index.html` for SPA routes.
pub async fn serve_embedded_ui(request: Request) -> Response {
    let path = request.uri().path();
    if is_api_like_path(path) {
        return StatusCode::NOT_FOUND.into_response();
    }

    let asset_path = normalize_asset_path(path);
    match UiAssets::get(&asset_path) {
        Some(asset) => asset_response(&asset_path, asset.data.into_owned()),
        None if should_fallback_to_index(path) => match UiAssets::get("index.html") {
            Some(asset) => asset_response("index.html", asset.data.into_owned()),
            None => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "embedded UI is missing index.html",
            )
                .into_response(),
        },
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn normalize_asset_path(path: &str) -> String {
    let trimmed = path.trim_start_matches('/');
    if trimmed.is_empty() {
        "index.html".to_string()
    } else {
        trimmed.to_string()
    }
}

fn should_fallback_to_index(path: &str) -> bool {
    let trimmed = path.trim_start_matches('/');
    trimmed.is_empty()
        || !trimmed
            .rsplit('/')
            .next()
            .is_some_and(|segment| segment.contains('.'))
}

fn is_api_like_path(path: &str) -> bool {
    path.starts_with("/api/")
        || matches!(
            path,
            "/api" | "/rpc" | "/ws" | "/health" | "/metrics" | "/mcp"
        )
        || path.starts_with("/mcp/")
}

fn asset_response(path: &str, body: Vec<u8>) -> Response {
    let cache_control = if path == "index.html" || !path.starts_with("assets/") {
        CACHE_CONTROL_NO_CACHE
    } else {
        CACHE_CONTROL_IMMUTABLE
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type(path))
        .header(header::CACHE_CONTROL, cache_control)
        .body(Body::from(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn content_type(path: &str) -> HeaderValue {
    let value = content_type_value(path);
    HeaderValue::from_static(value)
}

fn content_type_value(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or_default() {
        "css" => "text/css; charset=utf-8",
        "htm" => "text/html; charset=utf-8",
        "html" => "text/html; charset=utf-8",
        "ico" => "image/x-icon",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json",
        "map" => "application/json",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "webmanifest" => "application/manifest+json",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}
