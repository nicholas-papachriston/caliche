//! Simple in-process token-bucket rate limiting per client key.

use crate::security::SecurityPolicy;
use axum::extract::ConnectInfo;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const DEFAULT_REQUESTS_PER_MINUTE: u32 = 120;

struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

struct RateLimiter {
    capacity: f64,
    refill_per_sec: f64,
    buckets: DashMap<String, Bucket>,
}

impl RateLimiter {
    fn from_env() -> Self {
        let rpm = std::env::var("CALICHE_RATE_LIMIT_RPM")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_REQUESTS_PER_MINUTE);
        let capacity = rpm as f64;
        Self {
            capacity,
            refill_per_sec: capacity / 60.0,
            buckets: DashMap::new(),
        }
    }

    fn check(&self, key: &str) -> bool {
        let now = Instant::now();
        let mut entry = self.buckets.entry(key.to_string()).or_insert(Bucket {
            tokens: self.capacity,
            last_refill: now,
        });
        let bucket = entry.value_mut();
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        bucket.last_refill = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

fn limiter() -> &'static RateLimiter {
    static LIMITER: OnceLock<RateLimiter> = OnceLock::new();
    LIMITER.get_or_init(RateLimiter::from_env)
}

pub fn rate_limit_enabled() -> bool {
    crate::security::env_bool("CALICHE_RATE_LIMIT_ENABLED")
}

fn forwarded_client_ip(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(forwarded) = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
    {
        let trimmed = forwarded.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    headers
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn peer_addr<B>(request: &axum::http::Request<B>) -> SocketAddr {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| *addr)
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 0)))
}

fn client_key(
    peer: SocketAddr,
    headers: &axum::http::HeaderMap,
    policy: Option<&SecurityPolicy>,
) -> String {
    if policy.is_some_and(|p| p.peer_is_trusted_proxy(peer.ip()))
        && let Some(forwarded) = forwarded_client_ip(headers)
    {
        return forwarded;
    }
    peer.ip().to_string()
}

/// Reject requests when the per-client token bucket is exhausted.
pub async fn rate_limit_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if !rate_limit_enabled() {
        return next.run(request).await;
    }

    let policy = request.extensions().get::<Arc<SecurityPolicy>>();
    let key = client_key(
        peer_addr(&request),
        request.headers(),
        policy.map(AsRef::as_ref),
    );
    if limiter().check(&key) {
        next.run(request).await
    } else {
        tracing::warn!(
            target: "caliche_server::rate_limit",
            client = %key,
            "rate limit exceeded"
        );
        (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::ConnectInfo;
    use axum::http::HeaderMap;
    use std::net::IpAddr;

    #[test]
    fn token_bucket_allows_burst_then_blocks() {
        let limiter = RateLimiter {
            capacity: 2.0,
            refill_per_sec: 0.0,
            buckets: DashMap::new(),
        };
        assert!(limiter.check("client-a"));
        assert!(limiter.check("client-a"));
        assert!(!limiter.check("client-a"));
        assert!(limiter.check("client-b"));
    }

    #[test]
    fn peer_addr_reads_connect_info_extension() {
        let mut request = axum::extract::Request::new(axum::body::Body::empty());
        let addr = SocketAddr::from(([10, 0, 0, 1], 1234));
        request.extensions_mut().insert(ConnectInfo(addr));
        assert_eq!(peer_addr(&request), addr);
    }

    #[test]
    fn client_key_uses_peer_ip_by_default() {
        let peer = SocketAddr::new(IpAddr::from([10, 0, 0, 1]), 1234);
        let headers = HeaderMap::new();
        let policy = SecurityPolicy {
            jwt_enabled: false,
            require_auth: false,
            jwt_state: None,
            trusted_proxy: false,
            trusted_proxy_cidrs: Vec::new(),
            metrics_token: None,
            allow_unauth_orchestration: false,
            public_chat: false,
        };
        assert_eq!(
            client_key(peer, &headers, Some(&policy)),
            "10.0.0.1".to_string()
        );
    }

    #[test]
    fn client_key_honors_forwarded_headers_when_trusted() {
        let peer = SocketAddr::new(IpAddr::from([10, 0, 0, 1]), 1234);
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "203.0.113.5, 10.0.0.1".parse().expect("header"),
        );
        let policy = SecurityPolicy {
            jwt_enabled: false,
            require_auth: false,
            jwt_state: None,
            trusted_proxy: true,
            trusted_proxy_cidrs: vec!["10.0.0.0/8".parse().expect("cidr")],
            metrics_token: None,
            allow_unauth_orchestration: false,
            public_chat: false,
        };
        assert_eq!(
            client_key(peer, &headers, Some(&policy)),
            "203.0.113.5".to_string()
        );
    }

    #[test]
    fn client_key_ignores_forwarded_headers_from_untrusted_peer() {
        let peer = SocketAddr::new(IpAddr::from([198, 51, 100, 9]), 1234);
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.5".parse().expect("header"));
        let policy = SecurityPolicy {
            jwt_enabled: false,
            require_auth: false,
            jwt_state: None,
            trusted_proxy: true,
            trusted_proxy_cidrs: vec!["10.0.0.0/8".parse().expect("cidr")],
            metrics_token: None,
            allow_unauth_orchestration: false,
            public_chat: false,
        };
        assert_eq!(
            client_key(peer, &headers, Some(&policy)),
            "198.51.100.9".to_string()
        );
    }
}
