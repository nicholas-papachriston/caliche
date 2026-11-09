//! Webhook signature and Teams bearer verification tests.

use axum::http::{HeaderMap, Uri};
use caliche_server::webhook_verify::{
    PublicUrlOptions, discord_signing_payload, inbound_public_url_with, slack_signature_v0,
    slack_timestamp_is_fresh, twilio_expected_signature, verify_teams_hmac_bearer,
};
use std::collections::HashMap;

#[test]
fn slack_signature_v0_vector() {
    let secret = "8f734221a0442d528b49ea0a6afb2804";
    let ts = "1531420618";
    let body = b"token=ZZZZZZWSxiJJG1ARDvVav7f88888WJ8lz1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1z1&team_id=T1DC2JH3J&team_domain=test-team&channel_id=C8W9YJUFR&channel_name=foobar&user_id=U2OY0QW8F&user_name=Steve+Krenzel&command=%2Ftest&text=some+text+goes+here&response_url=https%3A%2F%2Fhooks.slack.com%2Factions%2FT1DC2JH3J%2F399973967235%2F0ActionsBeGone%2FXXXXXXXXXXXX";
    let got = slack_signature_v0(secret, ts, body).expect("hmac");
    assert_eq!(
        got.as_str(),
        "v0=5b9bb725e8a9d20a6bd65f1b0da0ec37e83b95e66b561e894f24d2491364b811"
    );
}

#[test]
fn twilio_expected_signature_sorted_params() {
    let url = "https://example.com/webhook";
    let mut form = HashMap::new();
    form.insert("Digits".to_string(), "1234".to_string());
    form.insert("CallSid".to_string(), "CA123".to_string());
    let token = "my_auth_token";
    let sig = twilio_expected_signature(url, &form, token).expect("sig");
    assert_eq!(sig.len(), 28);
    let again = twilio_expected_signature(url, &form, token).expect("sig2");
    assert_eq!(sig, again);
}

#[test]
fn discord_signing_payload_concat() {
    let ts = "1234567890";
    let body = b"{\"type\":1}";
    let p = discord_signing_payload(ts, body);
    assert_eq!(p.len(), ts.len() + body.len());
    assert!(p.starts_with(ts.as_bytes()));
    assert_eq!(&p[ts.len()..], body);
}

#[test]
fn inbound_public_url_ignores_forwarded_headers_by_default() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-forwarded-proto",
        "https".parse().expect("proto header value"),
    );
    headers.insert(
        "x-forwarded-host",
        "evil.example.com".parse().expect("host header value"),
    );
    headers.insert("host", "api.example.com".parse().expect("host"));
    let uri: Uri = "/ingress/slack?x=1".parse().expect("uri");
    assert_eq!(
        inbound_public_url_with(&headers, &uri, &PublicUrlOptions::default()),
        "http://api.example.com/ingress/slack?x=1"
    );
}

#[test]
fn inbound_public_url_uses_forwarded_headers_when_trusted() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-forwarded-proto",
        "https".parse().expect("proto header value"),
    );
    headers.insert(
        "x-forwarded-host",
        "api.example.com".parse().expect("host header value"),
    );
    let uri: Uri = "/ingress/slack?x=1".parse().expect("uri");
    let options = PublicUrlOptions {
        public_base_url: None,
        trust_forwarded_headers: true,
    };
    assert_eq!(
        inbound_public_url_with(&headers, &uri, &options),
        "https://api.example.com/ingress/slack?x=1"
    );
}

#[test]
fn inbound_public_url_fallback_host() {
    let headers = HeaderMap::new();
    let uri: Uri = "/path".parse().expect("uri");
    assert_eq!(
        inbound_public_url_with(&headers, &uri, &PublicUrlOptions::default()),
        "http://localhost/path"
    );
}

#[test]
fn slack_timestamp_is_fresh_rejects_garbage() {
    assert!(!slack_timestamp_is_fresh("nope", 1_700_000_000, 300));
}

#[test]
fn teams_hmac_bearer_accepts_matching_secret() {
    assert!(verify_teams_hmac_bearer(
        "shared-secret",
        Some("Bearer shared-secret")
    ));
}

#[test]
fn teams_hmac_bearer_rejects_mismatch() {
    assert!(!verify_teams_hmac_bearer(
        "shared-secret",
        Some("Bearer wrong-secret")
    ));
    assert!(!verify_teams_hmac_bearer("shared-secret", None));
    assert!(!verify_teams_hmac_bearer(
        "shared-secret",
        Some("Token shared-secret")
    ));
}
