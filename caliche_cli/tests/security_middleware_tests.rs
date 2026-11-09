//! Loopback-only guard for credential and vault HTTP routes.

use axum::Router;
use axum::http::StatusCode;
use axum::middleware;
use axum::routing::get;
use caliche_cli::server::routes::{
    is_credential_route, is_credential_route_accessible, require_loopback_for_credential_routes,
};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

#[test]
fn credential_route_prefixes() {
    assert!(is_credential_route("/api/cli/config"));
    assert!(is_credential_route("/api/cli/config/providers"));
    assert!(is_credential_route("/api/cli/vault/import"));
    assert!(!is_credential_route("/api/cli/doctor"));
    assert!(!is_credential_route("/api/chat/stream"));
}

#[test]
fn credential_route_accessible_from_loopback() {
    assert!(is_credential_route_accessible(
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        "/api/cli/config"
    ));
    assert!(is_credential_route_accessible(
        IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
        "/api/cli/vault/import"
    ));
}

#[test]
fn credential_route_blocked_off_loopback() {
    assert!(!is_credential_route_accessible(
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10)),
        "/api/cli/config"
    ));
    assert!(!is_credential_route_accessible(
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)),
        "/api/cli/vault/import"
    ));
}

#[test]
fn non_credential_routes_allowed_off_loopback() {
    assert!(is_credential_route_accessible(
        IpAddr::V4(Ipv4Addr::new(192, 168, 1, 10)),
        "/api/cli/doctor"
    ));
}

#[tokio::test]
async fn loopback_middleware_allows_local_peer() {
    let app = Router::new()
        .route("/api/cli/config", get(|| async { "ok" }))
        .route_layer(middleware::from_fn(require_loopback_for_credential_routes));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .expect("serve");
    });

    let client = reqwest::Client::new();
    let response = client
        .get(format!("http://{addr}/api/cli/config"))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), StatusCode::OK);
    server.abort();
}
