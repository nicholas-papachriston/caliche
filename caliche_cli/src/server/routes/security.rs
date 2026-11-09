//! Security middleware for sensitive CLI parity HTTP routes.

use axum::extract::ConnectInfo;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::net::{IpAddr, SocketAddr};

/// Paths that expose or mutate credentials and must not be reachable off-loopback.
pub fn is_credential_route(path: &str) -> bool {
    path.starts_with("/api/cli/config") || path.starts_with("/api/cli/vault")
}

/// Whether `peer_ip` may access `path` (credential routes require loopback).
pub fn is_credential_route_accessible(peer_ip: IpAddr, path: &str) -> bool {
    !is_credential_route(path) || peer_ip.is_loopback()
}

/// Reject credential/vault routes when the TCP peer is not loopback.
///
/// Complements JWT auth (when enabled) so binding `--host 0.0.0.0` does not expose
/// keychain CRUD to the LAN when JWT is unset.
pub async fn require_loopback_for_credential_routes(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if !is_credential_route_accessible(peer.ip(), path) {
        return (
            StatusCode::FORBIDDEN,
            "credential and vault routes are only available from loopback (127.0.0.1 / ::1)",
        )
            .into_response();
    }
    next.run(request).await
}
