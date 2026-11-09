//! Request logging middleware that logs request details with content truncation.
//!
//! Sensitive HTTP headers and query parameters are redacted before the log is
//! emitted. The request body is truncated and (when it parses as JSON) the
//! high-risk top-level keys listed in [`truncate_json_for_log`] are shortened.

use axum::body::Body;
use axum::extract::Request;
use axum::http::{Method, StatusCode, Uri, header};
use axum::middleware::Next;
use axum::response::Response;
use caliche_contracts::text::truncate_utf8;
use serde_json::Value;
use std::time::Instant;
use tracing::{Instrument, error, info, warn};
use uuid::Uuid;

/// Maximum length for logged content fields before truncation
const MAX_LOG_LENGTH: usize = 200;
/// Maximum request body size we will buffer for logging.
const MAX_LOG_BODY_BYTES: usize = 64 * 1024;
/// Query parameter names that are masked when logging request URIs.
const REDACTED_QUERY_PARAMS: &[&str] = &[
    "token",
    "api_key",
    "apikey",
    "access_token",
    "id_token",
    "password",
    "secret",
];
/// HTTP header names (lowercase) whose values we never log directly.
pub const REDACTED_HEADER_NAMES: &[&str] = &[
    "authorization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-api-key",
    "x-auth-token",
    "x-cluster-register-token",
];

/// Return a URI where sensitive query parameters are replaced with `***`.
pub fn redact_uri(uri: &Uri) -> String {
    let Some(query) = uri.query() else {
        return uri.to_string();
    };
    let mut redacted_pairs: Vec<String> = Vec::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, _value) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        let key_lower = key.to_ascii_lowercase();
        if REDACTED_QUERY_PARAMS
            .iter()
            .any(|sensitive| key_lower == *sensitive)
        {
            redacted_pairs.push(format!("{key}=***"));
        } else {
            redacted_pairs.push(pair.to_string());
        }
    }
    let base = uri
        .path_and_query()
        .map(|p| p.path())
        .unwrap_or_else(|| uri.path());
    let scheme = uri.scheme_str();
    let authority = uri.authority().map(|a| a.as_str());
    let query_str = redacted_pairs.join("&");
    match (scheme, authority) {
        (Some(scheme), Some(auth)) => format!("{scheme}://{auth}{base}?{query_str}"),
        _ => format!("{base}?{query_str}"),
    }
}

/// Returns true if the given header name is in the redacted list.
pub fn is_redacted_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    REDACTED_HEADER_NAMES.iter().any(|h| lower == *h)
}

/// Truncate a string to a maximum length, appending "..." if truncated
pub fn truncate_for_log(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!(
            "{}... (truncated, {} chars)",
            truncate_utf8(s, max_len),
            s.len()
        )
    }
}

fn is_sensitive_json_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "api_key"
            | "apikey"
            | "password"
            | "passwd"
            | "secret"
            | "token"
            | "access_token"
            | "refresh_token"
            | "authorization"
            | "auth_token"
            | "client_secret"
    )
}

/// Truncate a JSON value for logging, handling nested structures
pub fn truncate_json_for_log(value: &Value, max_len: usize) -> Value {
    match value {
        Value::String(s) => Value::String(truncate_for_log(s, max_len)),
        Value::Object(map) => {
            let mut truncated = serde_json::Map::new();
            for (key, val) in map.iter() {
                if is_sensitive_json_key(key) {
                    truncated.insert(key.clone(), Value::String("[REDACTED]".to_string()));
                    continue;
                }
                truncated.insert(key.clone(), truncate_json_for_log(val, max_len));
            }
            Value::Object(truncated)
        }
        Value::Array(arr) => {
            let truncated: Vec<Value> = arr
                .iter()
                .take(10) // Limit array size in logs
                .map(|v| truncate_json_for_log(v, max_len))
                .collect();
            if arr.len() > 10 {
                let mut result = truncated;
                result.push(Value::String(format!(
                    "... ({} more items)",
                    arr.len() - 10
                )));
                Value::Array(result)
            } else {
                Value::Array(truncated)
            }
        }
        _ => value.clone(),
    }
}

/// Request logging middleware.
///
/// Wraps the downstream handler in a tracing span named `http_request` with
/// request-level fields (request_id, method, path) so any inner instrumented
/// handlers inherit those fields in their own spans. This lets operators
/// correlate structured logs and OTLP traces without adding per-handler
/// boilerplate.
struct RequestLog<'a> {
    request_id: &'a str,
    method: &'a Method,
    redacted_uri: &'a str,
    path: &'a str,
    user_agent: &'a str,
    content_type: &'a str,
}

fn header_str<'a>(headers: &'a axum::http::HeaderMap, name: &str) -> &'a str {
    headers
        .get(name)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
}

fn should_buffer_body(method: &Method, content_type: &str, content_length: Option<usize>) -> bool {
    matches!(method, &Method::POST | &Method::PUT | &Method::PATCH)
        && !content_type.starts_with("multipart/")
        && content_length
            .map(|len| len <= MAX_LOG_BODY_BYTES)
            .unwrap_or(false)
}

fn parse_body_value_for_log(body_bytes: &[u8]) -> Option<Value> {
    if body_bytes.is_empty() {
        return None;
    }
    match serde_json::from_slice::<Value>(body_bytes) {
        Ok(json) => Some(truncate_json_for_log(&json, MAX_LOG_LENGTH)),
        Err(_) => match String::from_utf8(body_bytes.to_vec()) {
            Ok(s) if !s.is_empty() => Some(Value::String(truncate_for_log(&s, MAX_LOG_LENGTH))),
            _ => None,
        },
    }
}

fn log_incoming_with_body(log: &RequestLog<'_>, body_value: Option<&Value>) {
    if let Some(body_val) = body_value {
        info!(
            request_id = %log.request_id,
            method = %log.method,
            uri = %log.redacted_uri,
            user_agent = %log.user_agent,
            content_type = %log.content_type,
            body = %serde_json::to_string(body_val).unwrap_or_else(|_| "{}".to_string()),
            "Incoming request"
        );
    } else {
        info!(
            request_id = %log.request_id,
            method = %log.method,
            uri = %log.redacted_uri,
            user_agent = %log.user_agent,
            content_type = %log.content_type,
            "Incoming request"
        );
    }
}

fn log_incoming_basic(log: &RequestLog<'_>, content_length: Option<usize>) {
    info!(
        request_id = %log.request_id,
        method = %log.method,
        uri = %log.redacted_uri,
        user_agent = %log.user_agent,
        content_type = %log.content_type,
        content_length = content_length,
        "Incoming request"
    );
}

fn is_health_check_path(path: &str) -> bool {
    path == "/metrics" || path == "/health"
}

fn log_request_completion_simple(log: &RequestLog<'_>, status: StatusCode, duration_ms: u128) {
    if status.is_client_error() || status.is_server_error() {
        error!(
            request_id = %log.request_id,
            method = %log.method,
            uri = %log.redacted_uri,
            status = %status.as_u16(),
            duration_ms = duration_ms,
            "Request completed with error"
        );
    } else {
        info!(
            request_id = %log.request_id,
            method = %log.method,
            uri = %log.redacted_uri,
            status = %status.as_u16(),
            duration_ms = duration_ms,
            "Request completed"
        );
    }
}

fn log_request_completion(log: &RequestLog<'_>, status: StatusCode, duration_ms: u128) {
    let is_health_check = is_health_check_path(log.path);
    if status.is_client_error() || status.is_server_error() {
        if is_health_check && status == StatusCode::NOT_FOUND {
            tracing::debug!(
                request_id = %log.request_id,
                method = %log.method,
                uri = %log.redacted_uri,
                status = %status.as_u16(),
                duration_ms = duration_ms,
                "Health check endpoint not found"
            );
        } else {
            error!(
                request_id = %log.request_id,
                method = %log.method,
                uri = %log.redacted_uri,
                status = %status.as_u16(),
                duration_ms = duration_ms,
                "Request completed with error"
            );
        }
    } else if is_health_check {
        tracing::debug!(
            request_id = %log.request_id,
            method = %log.method,
            uri = %log.redacted_uri,
            status = %status.as_u16(),
            duration_ms = duration_ms,
            "Health check completed"
        );
    } else {
        info!(
            request_id = %log.request_id,
            method = %log.method,
            uri = %log.redacted_uri,
            status = %status.as_u16(),
            duration_ms = duration_ms,
            "Request completed"
        );
    }
}

async fn log_request_with_body(
    request: Request,
    next: Next,
    log: &RequestLog<'_>,
    request_span: tracing::Span,
    start_time: Instant,
) -> Response {
    let (parts, body) = request.into_parts();
    let body_bytes = match axum::body::to_bytes(body, MAX_LOG_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!(
                request_id = %log.request_id,
                error = %e,
                "Failed to read request body for logging"
            );
            let request = Request::from_parts(parts, Body::empty());
            let response = next.run(request).instrument(request_span.clone()).await;
            let status = response.status();
            log_request_completion_simple(log, status, start_time.elapsed().as_millis());
            return response;
        }
    };

    let body_value = parse_body_value_for_log(&body_bytes);
    let request = Request::from_parts(parts, Body::from(body_bytes));
    log_incoming_with_body(log, body_value.as_ref());
    let response = next.run(request).instrument(request_span.clone()).await;
    let status = response.status();
    log_request_completion(log, status, start_time.elapsed().as_millis());
    response
}

async fn log_request_without_body(
    request: Request,
    next: Next,
    log: &RequestLog<'_>,
    request_span: tracing::Span,
    start_time: Instant,
    content_length: Option<usize>,
) -> Response {
    log_incoming_basic(log, content_length);
    let response = next.run(request).instrument(request_span.clone()).await;
    let status = response.status();
    log_request_completion(log, status, start_time.elapsed().as_millis());
    response
}

/// Request logging middleware.
///
/// Wraps the downstream handler in a tracing span named `http_request` with
/// request-level fields (request_id, method, path) so any inner instrumented
/// handlers inherit those fields in their own spans. This lets operators
/// correlate structured logs and OTLP traces without adding per-handler
/// boilerplate.
pub async fn request_logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let redacted_uri = redact_uri(&uri);
    let headers = request.headers().clone();
    let start_time = Instant::now();
    let request_id = Uuid::now_v7().to_string();
    let request_path = uri.path().to_string();
    let request_span = tracing::info_span!(
        "http_request",
        request_id = %request_id,
        method = %method,
        path = %request_path,
    );

    let user_agent = header_str(&headers, "user-agent");
    let content_type = header_str(&headers, "content-type");
    let content_length = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.parse::<usize>().ok());
    let log = RequestLog {
        request_id: &request_id,
        method: &method,
        redacted_uri: &redacted_uri,
        path: &request_path,
        user_agent,
        content_type,
    };

    if should_buffer_body(&method, content_type, content_length) {
        log_request_with_body(request, next, &log, request_span, start_time).await
    } else {
        log_request_without_body(
            request,
            next,
            &log,
            request_span,
            start_time,
            content_length,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_uri_leaves_non_sensitive_params() {
        let uri: Uri = "/api/thing?foo=bar&limit=10".parse().expect("uri");
        let out = redact_uri(&uri);
        assert_eq!(out, "/api/thing?foo=bar&limit=10");
    }

    #[test]
    fn redact_uri_masks_token_and_api_key() {
        let uri: Uri = "/api/thing?token=abc&api_key=xyz&normal=ok"
            .parse()
            .expect("uri");
        let out = redact_uri(&uri);
        assert!(out.contains("token=***"));
        assert!(out.contains("api_key=***"));
        assert!(out.contains("normal=ok"));
    }

    #[test]
    fn redact_uri_preserves_absolute_form() {
        let uri: Uri = "https://host.example/path?password=p&x=1"
            .parse()
            .expect("uri");
        let out = redact_uri(&uri);
        assert!(out.starts_with("https://host.example"));
        assert!(out.contains("password=***"));
    }

    #[test]
    fn is_redacted_header_is_case_insensitive() {
        assert!(is_redacted_header("Authorization"));
        assert!(is_redacted_header("COOKIE"));
        assert!(is_redacted_header("X-Api-Key"));
        assert!(!is_redacted_header("content-type"));
    }

    #[test]
    fn redact_uri_masks_token_case_variants() {
        let uri: Uri = "/x?Token=abc&API_KEY=secret&keep=1".parse().expect("uri");
        let out = redact_uri(&uri);
        assert!(out.contains("Token=***") || out.contains("token=***"));
        assert!(out.contains("API_KEY=***") || out.contains("api_key=***"));
        assert!(out.contains("keep=1"));
    }

    #[test]
    fn truncate_json_for_log_redacts_sensitive_keys() {
        let value = serde_json::json!({
            "api_key": "sk-live",
            "Password": "hunter2",
            "nested": { "token": "abc", "ok": "value" },
            "message": "x".repeat(20)
        });
        let redacted = truncate_json_for_log(&value, 8);
        assert_eq!(redacted["api_key"], "[REDACTED]");
        assert_eq!(redacted["Password"], "[REDACTED]");
        assert_eq!(redacted["nested"]["token"], "[REDACTED]");
        assert_eq!(redacted["nested"]["ok"], "value");
        assert!(
            redacted["message"]
                .as_str()
                .expect("message")
                .contains("truncated")
        );
    }
}
