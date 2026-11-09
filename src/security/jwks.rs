//! Shared JWKS fetch/cache for JWT verification.

use jsonwebtoken::DecodingKey;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// JWKS cache for storing fetched public keys.
#[derive(Debug, Clone)]
pub struct JwksCache {
    keys: Arc<RwLock<HashMap<String, DecodingKey>>>,
    jwks_url: String,
    last_fetch: Arc<RwLock<Option<Instant>>>,
    /// Last forced refresh triggered by an unknown `kid` (rate-limited).
    last_miss_refresh: Arc<RwLock<Option<Instant>>>,
    cache_ttl: Duration,
    miss_refresh_cooldown: Duration,
    refresh_lock: Arc<Mutex<()>>,
    http_client: reqwest::Client,
    /// Test-only counter of HTTP refresh attempts.
    refresh_attempts: Arc<AtomicU64>,
}

impl JwksCache {
    pub fn try_new(jwks_url: String, cache_ttl_seconds: u64) -> anyhow::Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build JWKS HTTP client: {e}"))?;
        Ok(Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            jwks_url,
            last_fetch: Arc::new(RwLock::new(None)),
            last_miss_refresh: Arc::new(RwLock::new(None)),
            cache_ttl: Duration::from_secs(cache_ttl_seconds),
            // Bound unknown-kid refresh storms while still supporting key rotation.
            miss_refresh_cooldown: Duration::from_secs(5),
            refresh_lock: Arc::new(Mutex::new(())),
            http_client,
            refresh_attempts: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Override the unknown-kid refresh cooldown (tests / specialized deployments).
    #[must_use]
    pub fn with_miss_refresh_cooldown(mut self, cooldown: Duration) -> Self {
        self.miss_refresh_cooldown = cooldown;
        self
    }

    /// Number of HTTP JWKS refresh attempts observed by this cache.
    pub fn refresh_attempts(&self) -> u64 {
        self.refresh_attempts.load(Ordering::Relaxed)
    }

    /// JWKS endpoint URL this cache refreshes from.
    pub fn url(&self) -> &str {
        &self.jwks_url
    }

    fn cache_is_fresh(last_fetch: Option<Instant>, ttl: Duration) -> bool {
        last_fetch
            .map(|instant| instant.elapsed() <= ttl)
            .unwrap_or(false)
    }

    pub async fn get_key(&self, kid: &str) -> anyhow::Result<Option<DecodingKey>> {
        let kid = kid.trim();
        if kid.is_empty() {
            return Ok(None);
        }

        let last_fetch = *self.last_fetch.read().await;
        let fresh = Self::cache_is_fresh(last_fetch, self.cache_ttl);
        if let Some(existing) = self.keys.read().await.get(kid).cloned()
            && fresh
        {
            return Ok(Some(existing));
        }

        let unknown_kid = !self.keys.read().await.contains_key(kid);
        let miss_cooldown_elapsed = {
            let last_miss = *self.last_miss_refresh.read().await;
            last_miss
                .map(|instant| instant.elapsed() >= self.miss_refresh_cooldown)
                .unwrap_or(true)
        };
        let should_refresh = !fresh || (unknown_kid && miss_cooldown_elapsed);

        if should_refresh {
            let _refresh_guard = self.refresh_lock.lock().await;
            let last_fetch = *self.last_fetch.read().await;
            let fresh = Self::cache_is_fresh(last_fetch, self.cache_ttl);
            if let Some(existing) = self.keys.read().await.get(kid).cloned()
                && fresh
            {
                return Ok(Some(existing));
            }

            let unknown_kid = !self.keys.read().await.contains_key(kid);
            let miss_cooldown_elapsed = {
                let last_miss = *self.last_miss_refresh.read().await;
                last_miss
                    .map(|instant| instant.elapsed() >= self.miss_refresh_cooldown)
                    .unwrap_or(true)
            };
            let still_should_refresh = !fresh || (unknown_kid && miss_cooldown_elapsed);
            if still_should_refresh {
                if unknown_kid {
                    *self.last_miss_refresh.write().await = Some(Instant::now());
                }
                if let Err(err) = self.refresh_keys().await {
                    // Keep previously loaded keys when a rotation refetch fails.
                    if self.keys.read().await.is_empty() {
                        return Err(err);
                    }
                    tracing::warn!(
                        error = %err,
                        url = %self.jwks_url,
                        "JWKS refresh failed; retaining previously cached keys"
                    );
                }
            }
        }

        Ok(self.keys.read().await.get(kid).cloned())
    }

    /// Parse a JWKS JSON document and replace the in-memory key map.
    ///
    /// Useful for tests and offline injection without an HTTP round-trip.
    pub async fn load_jwks_document(&self, jwks: &serde_json::Value) -> anyhow::Result<()> {
        let decoded = decode_jwks_keys(jwks);
        if decoded.is_empty() {
            tracing::warn!(
                "JWKS document contained no usable keys for {}",
                self.jwks_url
            );
            let had_keys = !self.keys.read().await.is_empty();
            if had_keys {
                return Ok(());
            }
        } else {
            *self.keys.write().await = decoded;
        }
        *self.last_fetch.write().await = Some(Instant::now());
        Ok(())
    }

    async fn refresh_keys(&self) -> anyhow::Result<()> {
        self.refresh_attempts.fetch_add(1, Ordering::Relaxed);
        tracing::debug!("Refreshing JWKS from {}", self.jwks_url);
        let response = self
            .http_client
            .get(&self.jwks_url)
            .send()
            .await?
            .error_for_status()?;
        let jwks: serde_json::Value = response.json().await?;
        self.load_jwks_document(&jwks).await
    }
}

fn decode_jwks_keys(jwks: &serde_json::Value) -> HashMap<String, DecodingKey> {
    let mut decoded_keys: HashMap<String, DecodingKey> = HashMap::new();

    let Some(keys_array) = jwks.get("keys").and_then(|k| k.as_array()) else {
        tracing::warn!("JWKS response missing 'keys' array");
        return decoded_keys;
    };

    for key_obj in keys_array {
        let kid = match key_obj.get("kid").and_then(|k| k.as_str()) {
            Some(kid) if !kid.is_empty() => kid.to_string(),
            _ => {
                tracing::warn!("Skipping JWKS key without 'kid'");
                continue;
            }
        };

        if let Some(decoding_key) = jwk_x5c_to_decoding_key(key_obj) {
            decoded_keys.insert(kid, decoding_key);
            continue;
        }

        match jwk_to_decoding_key(key_obj) {
            Ok(Some(decoding_key)) => {
                decoded_keys.insert(kid, decoding_key);
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("Failed to parse JWKS key: {e}");
            }
        }
    }

    decoded_keys
}

/// Convert a JWKS JWK to a `DecodingKey` using x5c certificate chain (if present).
pub fn jwk_x5c_to_decoding_key(jwk: &serde_json::Value) -> Option<DecodingKey> {
    let x5c_first = jwk
        .get("x5c")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|v| v.as_str())?;

    let pem = pem_wrap_certificate(x5c_first);

    DecodingKey::from_rsa_pem(pem.as_bytes())
        .or_else(|_| DecodingKey::from_ec_pem(pem.as_bytes()))
        .ok()
}

/// Convert a JWKS JWK to a `DecodingKey` using raw components.
pub fn jwk_to_decoding_key(jwk: &serde_json::Value) -> anyhow::Result<Option<DecodingKey>> {
    let kty = jwk.get("kty").and_then(|v| v.as_str()).unwrap_or_default();

    match kty {
        "RSA" => {
            let n = match jwk.get("n").and_then(|v| v.as_str()) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    tracing::warn!("Skipping RSA JWK without 'n' component");
                    return Ok(None);
                }
            };
            let e = match jwk.get("e").and_then(|v| v.as_str()) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    tracing::warn!("Skipping RSA JWK without 'e' component");
                    return Ok(None);
                }
            };

            let key = DecodingKey::from_rsa_components(n, e)
                .map_err(|e| anyhow::anyhow!("Failed to build RSA decoding key: {e}"))?;
            Ok(Some(key))
        }
        "EC" => {
            let x = match jwk.get("x").and_then(|v| v.as_str()) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    tracing::warn!("Skipping EC JWK without 'x' component");
                    return Ok(None);
                }
            };
            let y = match jwk.get("y").and_then(|v| v.as_str()) {
                Some(v) if !v.is_empty() => v,
                _ => {
                    tracing::warn!("Skipping EC JWK without 'y' component");
                    return Ok(None);
                }
            };

            let key = DecodingKey::from_ec_components(x, y)
                .map_err(|e| anyhow::anyhow!("Failed to build EC decoding key: {e}"))?;
            Ok(Some(key))
        }
        other => {
            if !other.is_empty() {
                tracing::warn!("Skipping unsupported JWK kty='{other}'");
            } else {
                tracing::warn!("Skipping JWK without 'kty'");
            }
            Ok(None)
        }
    }
}

/// Wrap a base64-encoded DER certificate into PEM format.
pub fn pem_wrap_certificate(x5c_b64_der: &str) -> String {
    let mut pem = String::from("-----BEGIN CERTIFICATE-----\n");
    let bytes = x5c_b64_der.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let end = x5c_b64_der.floor_char_boundary(i + 64);
        pem.push_str(&x5c_b64_der[i..end]);
        pem.push('\n');
        i = end;
    }
    pem.push_str("-----END CERTIFICATE-----\n");
    pem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pem_wrap_certificate_keeps_multibyte_characters_whole() {
        let input = format!("a{}", "é".repeat(40));
        assert_eq!(
            pem_wrap_certificate(&input),
            format!(
                "-----BEGIN CERTIFICATE-----\na{}\n{}\n-----END CERTIFICATE-----\n",
                "é".repeat(31),
                "é".repeat(9)
            )
        );
    }

    #[test]
    fn pem_wrap_certificate_chunks_at_64() {
        let input = "A".repeat(70);
        let pem = pem_wrap_certificate(&input);
        assert!(pem.starts_with("-----BEGIN CERTIFICATE-----\n"));
        assert!(pem.ends_with("-----END CERTIFICATE-----\n"));
        let body: String = pem
            .lines()
            .filter(|l| !l.starts_with("-----"))
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(body, input);
        let lines: Vec<&str> = pem.lines().filter(|l| !l.starts_with("-----")).collect();
        assert_eq!(lines[0].len(), 64);
        assert_eq!(lines[1].len(), 6);
    }

    #[test]
    fn jwk_to_decoding_key_builds_rsa_from_components() {
        // Well-known RSA JWK test vector (n/e only).
        let jwk = serde_json::json!({
            "kty": "RSA",
            "kid": "test-rsa",
            "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
            "e": "AQAB"
        });
        let key = jwk_to_decoding_key(&jwk).unwrap();
        assert!(key.is_some());
    }

    #[test]
    fn jwk_to_decoding_key_skips_missing_components() {
        let jwk = serde_json::json!({ "kty": "RSA", "kid": "x" });
        assert!(jwk_to_decoding_key(&jwk).unwrap().is_none());
    }

    #[tokio::test]
    async fn load_jwks_document_indexes_by_kid() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        let doc = serde_json::json!({
            "keys": [{
                "kty": "RSA",
                "kid": "kid-1",
                "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
                "e": "AQAB"
            }]
        });
        cache.load_jwks_document(&doc).await.unwrap();
        assert!(cache.get_key("kid-1").await.unwrap().is_some());
        // Unknown kid forces a refresh attempt; HTTP fails against the dummy URL,
        // but previously loaded keys are retained and the miss returns None.
        assert!(cache.get_key("missing").await.unwrap().is_none());
        assert!(cache.get_key("kid-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn unknown_kid_within_ttl_can_pick_up_rotated_key() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        let first = sample_rsa_jwks("kid-1");
        cache.load_jwks_document(&first).await.unwrap();
        assert!(cache.get_key("kid-2").await.unwrap().is_none());

        let rotated = sample_rsa_jwks("kid-2");
        // Simulate a successful rotation refresh without waiting for TTL.
        cache.load_jwks_document(&rotated).await.unwrap();
        assert!(cache.get_key("kid-2").await.unwrap().is_some());
    }

    fn sample_rsa_jwks(kid: &str) -> serde_json::Value {
        serde_json::json!({
            "keys": [{
                "kty": "RSA",
                "kid": kid,
                "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
                "e": "AQAB"
            }]
        })
    }

    #[tokio::test]
    async fn empty_jwks_document_retains_existing_keys() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        cache
            .load_jwks_document(&sample_rsa_jwks("kid-1"))
            .await
            .unwrap();
        assert!(cache.get_key("kid-1").await.unwrap().is_some());

        cache
            .load_jwks_document(&serde_json::json!({ "keys": [] }))
            .await
            .unwrap();
        assert!(cache.get_key("kid-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn empty_jwks_document_on_empty_cache_marks_fetched() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        cache
            .load_jwks_document(&serde_json::json!({ "keys": [] }))
            .await
            .unwrap();
        // Fresh empty cache should not force another refresh for known misses
        // until TTL expires; unknown kid still attempts once under miss cooldown.
        assert_eq!(cache.refresh_attempts(), 0);
        let _ = cache.get_key("missing").await;
        assert_eq!(cache.refresh_attempts(), 1);
    }

    #[tokio::test]
    async fn unknown_kid_refresh_respects_miss_cooldown() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600)
            .unwrap()
            .with_miss_refresh_cooldown(Duration::from_secs(60));
        cache
            .load_jwks_document(&sample_rsa_jwks("kid-1"))
            .await
            .unwrap();

        assert!(cache.get_key("missing-a").await.unwrap().is_none());
        assert_eq!(cache.refresh_attempts(), 1);
        assert!(cache.get_key("kid-1").await.unwrap().is_some());

        // Second unknown kid within cooldown must not storm the JWKS endpoint.
        assert!(cache.get_key("missing-b").await.unwrap().is_none());
        assert_eq!(cache.refresh_attempts(), 1);
        assert!(cache.get_key("kid-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn failed_unknown_kid_refresh_retains_cached_keys() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        cache
            .load_jwks_document(&sample_rsa_jwks("kid-1"))
            .await
            .unwrap();
        assert!(cache.get_key("missing").await.unwrap().is_none());
        assert!(cache.get_key("kid-1").await.unwrap().is_some());
        assert_eq!(cache.refresh_attempts(), 1);
    }

    #[tokio::test]
    async fn empty_kid_is_ignored_without_refresh() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        cache
            .load_jwks_document(&sample_rsa_jwks("kid-1"))
            .await
            .unwrap();
        assert!(cache.get_key("").await.unwrap().is_none());
        assert!(cache.get_key("   ").await.unwrap().is_none());
        assert_eq!(cache.refresh_attempts(), 0);
        assert!(cache.get_key("kid-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn jwks_document_skips_empty_kid_entries() {
        let cache = JwksCache::try_new("https://example.test/jwks".into(), 3600).unwrap();
        let doc = serde_json::json!({
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "",
                    "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
                    "e": "AQAB"
                },
                {
                    "kty": "RSA",
                    "kid": "kid-ok",
                    "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw",
                    "e": "AQAB"
                }
            ]
        });
        cache.load_jwks_document(&doc).await.unwrap();
        assert!(cache.get_key("").await.unwrap().is_none());
        assert!(cache.get_key("kid-ok").await.unwrap().is_some());
    }
}
