//! JWT validation and JWKS-backed auth.
#![cfg(feature = "auth-jwt")]

use anyhow::{Context, Result, bail};
#[cfg(feature = "reqwest")]
use jsonwebtoken::decode_header;
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::auth::AuthContext;
#[cfg(feature = "reqwest")]
use super::jwks::JwksCache;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Claims {
    sub: String,
    #[serde(default)]
    tenant_id: Option<String>,
    #[serde(default)]
    roles: Vec<String>,
    exp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    aud: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    iss: Option<String>,
}

/// JWT configuration for enterprise multi-tenant deployments.
///
/// Supports HS256 shared secrets, RS256 static public keys, and JWKS discovery
/// via [`JwtValidator::from_config`] / [`JwtValidator::validate_token_async`].
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// Enable JWT authentication
    pub enabled: bool,
    /// JWT algorithm (e.g., "RS256", "HS256")
    pub algorithm: String,
    /// Path to public key file (for RS256)
    pub public_key_path: Option<PathBuf>,
    /// Public key as string (alternative to public_key_path)
    pub public_key: Option<String>,
    /// JWKS URL for key discovery (resolved asynchronously via kid)
    pub jwks_url: Option<String>,
    /// Claim name for tenant ID
    pub tenant_claim: Option<String>,
    /// Required claims that must be present
    pub required_claims: HashSet<String>,
    /// Validate expiration (exp claim)
    pub validate_exp: bool,
    /// Accepted `aud` values. Empty is rejected by [`JwtValidator::from_config`].
    pub audience: Vec<String>,
    /// Accepted `iss` values. Empty is rejected by [`JwtValidator::from_config`].
    pub issuer: Vec<String>,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: "RS256".to_owned(),
            public_key_path: None,
            public_key: None,
            jwks_url: None,
            tenant_claim: Some("tenant_id".to_owned()),
            required_claims: HashSet::from(["sub".to_owned()]),
            validate_exp: true,
            audience: Vec::new(),
            issuer: Vec::new(),
        }
    }
}

impl JwtConfig {
    /// Create a builder for JWT configuration
    pub fn builder() -> JwtConfigBuilder {
        JwtConfigBuilder::default()
    }

    /// Create configuration from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `JWT_ENABLED`: Enable JWT authentication (default: false)
    /// - `JWT_ALGORITHM`: JWT algorithm (default: "RS256")
    /// - `JWT_PUBLIC_KEY_PATH`: Path to public key file
    /// - `JWT_PUBLIC_KEY`: Public key as string (alternative to path)
    /// - `JWT_JWKS_URL`: JWKS URL for key discovery (use `validate_token_async`)
    /// - `JWT_TENANT_CLAIM`: Claim name for tenant ID (default: "tenant_id")
    /// - `JWT_REQUIRED_CLAIMS`: Comma-separated list of required claims (default: "sub")
    /// - `JWT_VALIDATE_EXP`: Validate expiration (default: true)
    /// - `JWT_AUDIENCE`: Comma-separated accepted audiences (required when enabled)
    /// - `JWT_ISSUER`: Comma-separated accepted issuers (required when enabled)
    pub fn from_env() -> Self {
        let enabled = std::env::var("JWT_ENABLED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        if !enabled {
            return Self {
                enabled: false,
                ..Default::default()
            };
        }

        let algorithm = std::env::var("JWT_ALGORITHM").unwrap_or_else(|_| "RS256".to_owned());

        let public_key_path = std::env::var("JWT_PUBLIC_KEY_PATH").ok().map(PathBuf::from);

        let public_key = std::env::var("JWT_PUBLIC_KEY").ok();

        let jwks_url = std::env::var("JWT_JWKS_URL").ok();

        let tenant_claim = std::env::var("JWT_TENANT_CLAIM")
            .ok()
            .or_else(|| Some("tenant_id".to_owned()));

        let required_claims_str =
            std::env::var("JWT_REQUIRED_CLAIMS").unwrap_or_else(|_| "sub".to_owned());

        let required_claims: HashSet<String> = required_claims_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let validate_exp = std::env::var("JWT_VALIDATE_EXP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(true);

        let audience = csv_env("JWT_AUDIENCE");
        let issuer = csv_env("JWT_ISSUER");

        Self {
            enabled: true,
            algorithm,
            public_key_path,
            public_key,
            jwks_url,
            tenant_claim,
            required_claims,
            validate_exp,
            audience,
            issuer,
        }
    }
}

/// Builder for JWT configuration
#[derive(Debug, Default)]
pub struct JwtConfigBuilder {
    enabled: Option<bool>,
    algorithm: Option<String>,
    public_key_path: Option<PathBuf>,
    public_key: Option<String>,
    jwks_url: Option<String>,
    tenant_claim: Option<String>,
    required_claims: Option<HashSet<String>>,
    validate_exp: Option<bool>,
    audience: Option<Vec<String>>,
    issuer: Option<Vec<String>>,
}

impl JwtConfigBuilder {
    /// Enable or disable JWT authentication
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    /// Set the JWT algorithm
    pub fn algorithm(mut self, algorithm: impl Into<String>) -> Self {
        self.algorithm = Some(algorithm.into());
        self
    }

    /// Set the public key file path
    pub fn public_key_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.public_key_path = Some(path.into());
        self
    }

    /// Set the public key as string
    pub fn public_key(mut self, key: impl Into<String>) -> Self {
        self.public_key = Some(key.into());
        self
    }

    /// Set the JWKS URL
    pub fn jwks_url(mut self, url: impl Into<String>) -> Self {
        self.jwks_url = Some(url.into());
        self
    }

    /// Set the tenant claim name
    pub fn tenant_claim(mut self, claim: impl Into<String>) -> Self {
        self.tenant_claim = Some(claim.into());
        self
    }

    /// Set required claims
    pub fn required_claims(mut self, claims: HashSet<String>) -> Self {
        self.required_claims = Some(claims);
        self
    }

    /// Enable or disable expiration validation
    pub fn validate_exp(mut self, validate: bool) -> Self {
        self.validate_exp = Some(validate);
        self
    }

    /// Set accepted token audiences.
    pub fn audience(mut self, audience: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.audience = Some(audience.into_iter().map(Into::into).collect());
        self
    }

    /// Set accepted token issuers.
    pub fn issuer(mut self, issuer: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.issuer = Some(issuer.into_iter().map(Into::into).collect());
        self
    }

    /// Build the configuration
    pub fn build(self) -> JwtConfig {
        let defaults = JwtConfig::default();
        JwtConfig {
            enabled: self.enabled.unwrap_or(defaults.enabled),
            algorithm: self.algorithm.unwrap_or(defaults.algorithm),
            public_key_path: self.public_key_path.or(defaults.public_key_path),
            public_key: self.public_key.or(defaults.public_key),
            jwks_url: self.jwks_url.or(defaults.jwks_url),
            tenant_claim: self.tenant_claim.or(defaults.tenant_claim),
            required_claims: self.required_claims.unwrap_or(defaults.required_claims),
            validate_exp: self.validate_exp.unwrap_or(defaults.validate_exp),
            audience: self.audience.unwrap_or(defaults.audience),
            issuer: self.issuer.unwrap_or(defaults.issuer),
        }
    }
}

pub struct JwtValidator {
    decoding_key: Option<DecodingKey>,
    encoding_key: Option<EncodingKey>,
    validation: Validation,
    tenant_claim: Option<String>,
    audience: Vec<String>,
    issuer: Vec<String>,
    #[cfg(feature = "reqwest")]
    jwks_cache: Option<Arc<JwksCache>>,
}

impl JwtValidator {
    /// Build a validator from [`JwtConfig`].
    ///
    /// Accepts a static public key / HS256 secret and/or a JWKS URL. When only
    /// JWKS is configured, use [`Self::validate_token_async`] so kids can be
    /// resolved from the remote key set.
    pub fn from_config(config: &JwtConfig) -> Result<Self> {
        let algorithm = parse_algorithm(&config.algorithm)?;
        if config.audience.is_empty() || config.issuer.is_empty() {
            bail!("JWT_AUDIENCE and JWT_ISSUER are required");
        }
        let mut validation = Validation::new(algorithm);
        validation.validate_exp = config.validate_exp;
        validation.set_audience(
            &config
                .audience
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        validation.set_issuer(&config.issuer.iter().map(String::as_str).collect::<Vec<_>>());
        validation.set_required_spec_claims(
            &config
                .required_claims
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );

        #[cfg(feature = "reqwest")]
        let jwks_cache = match config.jwks_url.as_deref() {
            Some(url) => Some(Arc::new(JwksCache::try_new(url.to_string(), 3600)?)),
            None => None,
        };
        #[cfg(not(feature = "reqwest"))]
        if config.jwks_url.is_some() {
            bail!(
                "JWT JWKS URL requires the `reqwest` feature; rebuild with a provider feature or `reqwest`"
            );
        }

        let (decoding_key, encoding_key) = match algorithm {
            Algorithm::HS256 => match load_hs256_secret(config) {
                Ok(secret_bytes) => (
                    Some(DecodingKey::from_secret(&secret_bytes)),
                    Some(EncodingKey::from_secret(&secret_bytes)),
                ),
                Err(e) => {
                    if config.jwks_url.is_some() {
                        (None, None)
                    } else {
                        return Err(e);
                    }
                }
            },
            Algorithm::RS256 => match load_rs256_public_key_pem(config) {
                Ok(pem) => (
                    Some(
                        DecodingKey::from_rsa_pem(pem.as_bytes())
                            .context("invalid RS256 public key PEM")?,
                    ),
                    None,
                ),
                Err(e) => {
                    if config.jwks_url.is_some() {
                        (None, None)
                    } else {
                        return Err(e);
                    }
                }
            },
            other => bail!("unsupported JWT algorithm for validator: {other:?}"),
        };

        #[cfg(feature = "reqwest")]
        if decoding_key.is_none() && jwks_cache.is_none() {
            bail!("JWT_PUBLIC_KEY, JWT_PUBLIC_KEY_PATH, or JWT_JWKS_URL is required");
        }
        #[cfg(not(feature = "reqwest"))]
        if decoding_key.is_none() {
            bail!("JWT_PUBLIC_KEY or JWT_PUBLIC_KEY_PATH is required");
        }

        Ok(Self {
            decoding_key,
            encoding_key,
            validation,
            tenant_claim: config.tenant_claim.clone(),
            audience: config.audience.clone(),
            issuer: config.issuer.clone(),
            #[cfg(feature = "reqwest")]
            jwks_cache,
        })
    }

    /// Optional JWKS cache when configured via [`JwtConfig::jwks_url`].
    #[cfg(feature = "reqwest")]
    pub fn jwks_cache(&self) -> Option<&Arc<JwksCache>> {
        self.jwks_cache.as_ref()
    }

    pub fn generate_token(&self, context: &AuthContext, ttl_secs: u64) -> Result<String> {
        let Some(encoding_key) = &self.encoding_key else {
            bail!("JWT token generation is only supported for HS256 validators");
        };

        let exp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() + ttl_secs;

        let claims = Claims {
            sub: context.user_id.clone(),
            tenant_id: context.tenant_id.clone(),
            roles: context.roles.clone(),
            exp,
            aud: self.audience.first().cloned(),
            iss: self.issuer.first().cloned(),
        };

        encode(&Header::new(Algorithm::HS256), &claims, encoding_key)
            .context("Failed to generate JWT")
    }

    /// Validate a JWT with a statically configured decoding key.
    ///
    /// For JWKS-backed validators, use [`Self::validate_token_async`].
    pub fn validate_token(&self, token: &str) -> Result<AuthContext> {
        let Some(decoding_key) = &self.decoding_key else {
            bail!(
                "JWT validator has no static key; call validate_token_async for JWKS-backed tokens"
            );
        };
        let token_data = decode::<serde_json::Value>(token, decoding_key, &self.validation)
            .context("Invalid JWT")?;
        Self::auth_context_from_claims(&token_data.claims, self.tenant_claim.as_deref())
    }

    /// Validate a JWT, resolving `kid` via JWKS when configured.
    pub async fn validate_token_async(&self, token: &str) -> Result<AuthContext> {
        #[cfg(feature = "reqwest")]
        if let Some(jwks) = &self.jwks_cache {
            let header = decode_header(token).context("Invalid JWT header")?;
            let kid = header
                .kid
                .as_deref()
                .map(str::trim)
                .filter(|kid| !kid.is_empty())
                .context("JWT missing kid (key ID) required for JWKS validation")?;
            let decoding_key = jwks
                .get_key(kid)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Key ID {kid} not found in JWKS"))?;
            let token_data = decode::<serde_json::Value>(token, &decoding_key, &self.validation)
                .context("Invalid JWT")?;
            return Self::auth_context_from_claims(
                &token_data.claims,
                self.tenant_claim.as_deref(),
            );
        }

        self.validate_token(token)
    }

    fn auth_context_from_claims(
        claims: &serde_json::Value,
        tenant_claim: Option<&str>,
    ) -> Result<AuthContext> {
        let user_id = claims
            .get("sub")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .context("JWT missing sub claim")?;

        let tenant_id = tenant_claim.and_then(|claim_name| {
            claims.get(claim_name).and_then(|value| match value {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
        });

        let roles = claims
            .get("roles")
            .and_then(|value| serde_json::from_value::<Vec<String>>(value.clone()).ok())
            .unwrap_or_default();

        Ok(AuthContext {
            user_id,
            tenant_id,
            roles,
            permissions: vec![],
        })
    }
}

fn csv_env(name: &str) -> Vec<String> {
    std::env::var(name)
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_algorithm(raw: &str) -> Result<Algorithm> {
    match raw.trim().to_ascii_uppercase().as_str() {
        "HS256" => Ok(Algorithm::HS256),
        "RS256" => Ok(Algorithm::RS256),
        other => bail!("unsupported JWT algorithm: {other}"),
    }
}

fn load_hs256_secret(config: &JwtConfig) -> Result<Vec<u8>> {
    if let Some(key) = &config.public_key {
        return Ok(key.as_bytes().to_vec());
    }
    if let Some(path) = &config.public_key_path {
        return fs::read(path)
            .with_context(|| format!("failed to read JWT secret at {}", path.display()));
    }
    std::env::var("JWT_SECRET")
        .map(String::into_bytes)
        .context("JWT_SECRET is required for HS256 JWT validation")
}

fn load_rs256_public_key_pem(config: &JwtConfig) -> Result<String> {
    if let Some(key) = &config.public_key {
        return Ok(key.clone());
    }
    if let Some(path) = &config.public_key_path {
        return fs::read_to_string(path)
            .with_context(|| format!("failed to read JWT public key at {}", path.display()));
    }
    bail!("JWT_PUBLIC_KEY or JWT_PUBLIC_KEY_PATH is required for RS256 JWT validation");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hs256_round_trip_validates_claims() {
        let config = JwtConfig::builder()
            .algorithm("HS256")
            .public_key("unit-test-secret")
            .audience(["caliche"])
            .issuer(["caliche"])
            .build();
        let validator = JwtValidator::from_config(&config).expect("validator");
        let context = AuthContext {
            user_id: "user-1".to_string(),
            tenant_id: Some("tenant-a".to_string()),
            roles: vec!["admin".to_string()],
            permissions: vec![],
        };
        let token = validator
            .generate_token(&context, 60)
            .expect("generate token");
        let parsed = validator.validate_token(&token).expect("validate token");
        assert_eq!(parsed.user_id, "user-1");
        assert_eq!(parsed.tenant_id.as_deref(), Some("tenant-a"));
        assert_eq!(parsed.roles, vec!["admin".to_string()]);
    }

    #[test]
    fn missing_audience_or_issuer_fails_closed() {
        let mut config = JwtConfig::builder()
            .algorithm("HS256")
            .public_key("unit-test-secret")
            .audience(["caliche"])
            .issuer(["caliche"])
            .build();
        config.audience.clear();
        let err = JwtValidator::from_config(&config)
            .err()
            .expect("empty audience");
        assert!(err.to_string().contains("JWT_AUDIENCE"));
        config.audience = vec!["caliche".to_string()];
        config.issuer.clear();
        let err = JwtValidator::from_config(&config)
            .err()
            .expect("empty issuer");
        assert!(err.to_string().contains("JWT_ISSUER"));
    }

    #[test]
    fn wrong_audience_is_rejected() {
        let config = JwtConfig::builder()
            .algorithm("HS256")
            .public_key("unit-test-secret")
            .audience(["caliche"])
            .issuer(["caliche"])
            .build();
        let validator = JwtValidator::from_config(&config).expect("validator");
        let other = JwtConfig::builder()
            .algorithm("HS256")
            .public_key("unit-test-secret")
            .audience(["other-service"])
            .issuer(["caliche"])
            .build();
        let other_validator = JwtValidator::from_config(&other).expect("other validator");
        let token = other_validator
            .generate_token(
                &AuthContext {
                    user_id: "user-1".to_string(),
                    tenant_id: None,
                    roles: vec![],
                    permissions: vec![],
                },
                60,
            )
            .expect("token");
        let err = validator.validate_token(&token).expect_err("wrong aud");
        assert!(
            err.to_string().to_ascii_lowercase().contains("invalid")
                || err.to_string().to_ascii_lowercase().contains("aud"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn default_and_builder_do_not_invent_audience() {
        let default = JwtConfig::default();
        assert!(default.audience.is_empty());
        assert!(default.issuer.is_empty());
        let config = JwtConfig::builder()
            .algorithm("HS256")
            .public_key("unit-test-secret")
            .enabled(true)
            .build();
        assert!(config.audience.is_empty());
        assert!(config.issuer.is_empty());
        let err = JwtValidator::from_config(&config)
            .err()
            .expect("silent caliche audience");
        assert!(err.to_string().contains("JWT_AUDIENCE"));
    }
}
