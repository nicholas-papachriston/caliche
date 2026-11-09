//! Pure helpers for webhook signature verification.
//!
//! Host layers (API ingress, future gateway crate) should call these helpers
//! rather than re-implementing Slack/Twilio/Discord/Teams crypto.

use base64::{Engine, engine::general_purpose::STANDARD};
use hmac::{Hmac, Mac};
use http::{HeaderMap, Uri};
use sha1::Sha1;
use sha2::Sha256;
use std::collections::HashMap;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;
type HmacSha1 = Hmac<Sha1>;

/// When true, enabled webhook adapters must have verification secrets configured.
///
/// Production (`CALICHE_ENV=production`) requires verification unless
/// `CALICHE_ALLOW_INSECURE_PRODUCTION` is set.
#[must_use]
pub fn webhook_verification_required() -> bool {
    env_bool("CALICHE_WEBHOOK_REQUIRE_VERIFICATION")
        || (crate::security::is_production_profile()
            && !crate::security::allow_insecure_production())
}

/// Resolve a webhook secret from the environment, failing when verification is enforced.
pub fn resolve_webhook_secret(env_var: &str) -> anyhow::Result<Option<String>> {
    match caliche_contracts::credential_env::var(env_var) {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        _ if webhook_verification_required() => {
            anyhow::bail!("{env_var} is required when CALICHE_WEBHOOK_REQUIRE_VERIFICATION is set")
        }
        _ => Ok(None),
    }
}

/// Slack `v0` request signature: `v0:{timestamp}:{raw_body}` keyed by the signing secret.
pub fn slack_signature_v0(
    signing_secret: &str,
    timestamp: &str,
    body: &[u8],
) -> anyhow::Result<String> {
    let signed_payload = format!("v0:{timestamp}:{}", String::from_utf8_lossy(body));
    let mut mac = HmacSha256::new_from_slice(signing_secret.as_bytes())?;
    mac.update(signed_payload.as_bytes());
    let digest = mac.finalize().into_bytes();
    Ok(format!("v0={}", hex::encode(digest)))
}

/// Twilio webhook HMAC-SHA1 (base64), per Twilio validation algorithm.
pub fn twilio_expected_signature(
    url: &str,
    form: &HashMap<String, String>,
    auth_token: &str,
) -> anyhow::Result<String> {
    let mut pairs = form.iter().collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.cmp(b.0));

    let mut data = String::from(url);
    for (k, v) in pairs {
        data.push_str(k);
        data.push_str(v);
    }

    let mut mac = HmacSha1::new_from_slice(auth_token.as_bytes())?;
    mac.update(data.as_bytes());
    let bytes = mac.finalize().into_bytes();
    Ok(STANDARD.encode(bytes))
}

/// Bytes signed by Discord (Ed25519) for incoming webhooks: `timestamp || body`.
#[must_use]
pub fn discord_signing_payload(timestamp: &str, body: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(timestamp.len() + body.len());
    message.extend_from_slice(timestamp.as_bytes());
    message.extend_from_slice(body);
    message
}

fn env_bool(name: &str) -> bool {
    caliche_contracts::credential_env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Constant-time compare for webhook signatures and shared secrets.
#[must_use]
pub fn signatures_equal(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    left.len() == right.len() && bool::from(left.ct_eq(right))
}

/// Slack replay window: timestamp must parse as unix seconds and be within `max_skew_secs`.
///
/// Non-numeric timestamps are rejected. Callers must not skip this check on parse failure.
#[must_use]
pub fn slack_timestamp_is_fresh(timestamp: &str, now_unix: i64, max_skew_secs: u64) -> bool {
    let Ok(ts) = timestamp.parse::<i64>() else {
        return false;
    };
    now_unix.abs_diff(ts) <= max_skew_secs
}

/// Options for reconstructing the public URL used in webhook signatures.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PublicUrlOptions {
    /// Scheme+host used instead of request headers, for example `https://api.example.com`.
    pub public_base_url: Option<String>,
    /// When true, honor `X-Forwarded-Proto` / `X-Forwarded-Host`. Default is false.
    pub trust_forwarded_headers: bool,
}

impl PublicUrlOptions {
    /// Load from `CALICHE_PUBLIC_BASE_URL` and `CALICHE_TRUST_FORWARDED_HEADERS`.
    #[must_use]
    pub fn from_env() -> Self {
        let public_base_url = std::env::var("CALICHE_PUBLIC_BASE_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty());
        Self {
            public_base_url,
            trust_forwarded_headers: env_bool("CALICHE_TRUST_FORWARDED_HEADERS"),
        }
    }
}

fn env_var_nonempty(name: &str) -> bool {
    caliche_contracts::credential_env::var(name)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// Whether Teams ingress verification should run (global flag or credentials present).
#[must_use]
pub fn teams_verification_required() -> bool {
    webhook_verification_required()
        || env_var_nonempty("TEAMS_WEBHOOK_SECRET")
        || env_var_nonempty("MICROSOFT_APP_ID")
}

/// Constant-time compare for `TEAMS_WEBHOOK_SECRET` bearer tokens.
#[must_use]
pub fn verify_teams_hmac_bearer(expected_secret: &str, authorization: Option<&str>) -> bool {
    let Some(token) = authorization
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
    else {
        return false;
    };
    let expected = expected_secret.as_bytes();
    let presented = token.as_bytes();
    expected.len() == presented.len() && bool::from(expected.ct_eq(presented))
}

/// Reconstruct the public URL Twilio/Slack used when behind `X-Forwarded-*` proxies.
///
/// `X-Forwarded-*` is ignored unless [`PublicUrlOptions::trust_forwarded_headers`] is set.
/// Prefer [`PublicUrlOptions::public_base_url`] in production.
#[must_use]
pub fn inbound_public_url(headers: &HeaderMap, uri: &Uri) -> String {
    inbound_public_url_with(headers, uri, &PublicUrlOptions::from_env())
}

/// Reconstruct the public webhook URL using explicit options.
#[must_use]
pub fn inbound_public_url_with(
    headers: &HeaderMap,
    uri: &Uri,
    options: &PublicUrlOptions,
) -> String {
    if let Some(base) = options
        .public_base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let base = base.trim_end_matches('/');
        return format!("{base}{uri}");
    }

    let proto = if options.trust_forwarded_headers {
        headers
            .get("x-forwarded-proto")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("http")
    } else {
        "http"
    };
    let host = if options.trust_forwarded_headers {
        headers
            .get("x-forwarded-host")
            .or_else(|| headers.get("host"))
            .and_then(|v| v.to_str().ok())
            .unwrap_or("localhost")
    } else {
        headers
            .get("host")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("localhost")
    };
    format!("{proto}://{host}{uri}")
}

/// Verify Microsoft Teams / Bot Framework ingress authentication.
///
/// When `TEAMS_WEBHOOK_SECRET` is set, validates the bearer token with a
/// constant-time compare. When `MICROSOFT_APP_ID` is set (and no shared secret),
/// validates the Bot Framework JWT against Microsoft's JWKS. Verification also
/// runs when [`webhook_verification_required`] is true.
#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
pub async fn verify_teams_signature(headers: &HeaderMap) -> anyhow::Result<()> {
    if !teams_verification_required() {
        return Ok(());
    }

    let authorization = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"))
        .and_then(|value| value.to_str().ok());

    if let Some(secret) = std::env::var("TEAMS_WEBHOOK_SECRET")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        if verify_teams_hmac_bearer(&secret, authorization) {
            return Ok(());
        }
        anyhow::bail!("Teams webhook bearer token mismatch");
    }

    if let Some(app_id) = std::env::var("MICROSOFT_APP_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let token = authorization
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("missing Teams Authorization bearer token"))?;
        verify_bot_framework_jwt(token, app_id.trim()).await?;
        return Ok(());
    }

    if webhook_verification_required() {
        anyhow::bail!("missing Teams Authorization bearer token");
    }

    anyhow::bail!("Teams webhook verification not configured");
}

#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
mod teams_jwt {
    use super::BOT_FRAMEWORK_ISSUER;
    use crate::security::JwksCache;
    use jsonwebtoken::{Algorithm, Validation, decode, decode_header};
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::sync::OnceCell;

    const BOT_FRAMEWORK_JWKS_URL: &str = "https://login.botframework.com/v1/.well-known/keys";

    static BOT_FRAMEWORK_JWKS: OnceCell<Arc<JwksCache>> = OnceCell::const_new();

    pub(super) async fn verify_bot_framework_jwt(token: &str, app_id: &str) -> anyhow::Result<()> {
        let header = decode_header(token)?;
        let kid = header
            .kid
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Teams JWT missing kid"))?;

        let cache = BOT_FRAMEWORK_JWKS
            .get_or_try_init(|| async {
                JwksCache::try_new(BOT_FRAMEWORK_JWKS_URL.to_string(), 3600).map(Arc::new)
            })
            .await?;
        let decoding_key = cache
            .get_key(kid)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Teams JWT kid not found in Bot Framework JWKS"))?;

        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[BOT_FRAMEWORK_ISSUER]);
        validation.set_audience(&[app_id]);
        validation.validate_exp = true;
        validation.required_spec_claims = HashSet::new();

        decode::<serde_json::Value>(token, &decoding_key, &validation)?;
        Ok(())
    }
}

#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
use teams_jwt::verify_bot_framework_jwt;

#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
const BOT_FRAMEWORK_ISSUER: &str = "https://api.botframework.com";

#[cfg(test)]
mod tests {
    use super::*;

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
    fn inbound_public_url_prefers_configured_base() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-host",
            "evil.example.com".parse().expect("host header value"),
        );
        let uri: Uri = "/ingress/twilio".parse().expect("uri");
        let options = PublicUrlOptions {
            public_base_url: Some("https://hooks.example.com".to_string()),
            trust_forwarded_headers: true,
        };
        assert_eq!(
            inbound_public_url_with(&headers, &uri, &options),
            "https://hooks.example.com/ingress/twilio"
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
    fn slack_timestamp_rejects_non_numeric_and_stale() {
        assert!(!slack_timestamp_is_fresh(
            "not-a-number",
            1_531_420_618,
            300
        ));
        assert!(!slack_timestamp_is_fresh("1000", 1_531_420_618, 300));
        assert!(slack_timestamp_is_fresh("1531420618", 1_531_420_618, 300));
    }

    #[test]
    fn signatures_equal_rejects_mismatch() {
        assert!(signatures_equal("abc", "abc"));
        assert!(!signatures_equal("abc", "abd"));
        assert!(!signatures_equal("abc", "ab"));
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
}
