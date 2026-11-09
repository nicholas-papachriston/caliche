use crate::jwks_cache::JwksCache;
use crate::middleware::context::UserContext;
use crate::security::{SecurityPolicy, env_bool, is_production_profile};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use jsonwebtoken::{DecodingKey, Validation, decode, decode_header};
use std::collections::HashSet;
use std::sync::Arc;

/// Generic client-facing auth failure (details are logged server-side).
pub const AUTH_REQUIRED_MESSAGE: &str = "authentication required";
pub const INVALID_TOKEN_MESSAGE: &str = "invalid or missing authentication token";
/// Refusal for orchestration routes, which never run unauthenticated by default.
pub const ORCHESTRATION_AUTH_REQUIRED_MESSAGE: &str =
    "authentication required for orchestration routes";

/// JWT authentication middleware configuration
#[derive(Debug, Clone)]
pub struct JwtAuthConfig {
    pub enabled: bool,
    pub algorithm: jsonwebtoken::Algorithm,
    pub public_key: Option<String>,
    pub public_key_path: Option<String>,
    pub jwks_url: Option<String>,
    pub tenant_claim: String,
    pub required_claims: Vec<String>,
    pub validate_exp: bool,
    pub audience: Vec<String>,
    pub issuer: Vec<String>,
}

impl Default for JwtAuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: jsonwebtoken::Algorithm::RS256,
            public_key: None,
            public_key_path: None,
            jwks_url: None,
            tenant_claim: "tenant_id".to_string(),
            required_claims: vec!["sub".to_string()],
            validate_exp: true,
            audience: Vec::new(),
            issuer: Vec::new(),
        }
    }
}

impl JwtAuthConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        let enabled = std::env::var("JWT_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        if !enabled {
            return Ok(Self {
                enabled: false,
                ..Default::default()
            });
        }

        let algorithm_str = std::env::var("JWT_ALGORITHM")
            .unwrap_or_else(|_| "RS256".to_string())
            .to_uppercase();

        let algorithm = match algorithm_str.as_str() {
            "RS256" => jsonwebtoken::Algorithm::RS256,
            "RS384" => jsonwebtoken::Algorithm::RS384,
            "RS512" => jsonwebtoken::Algorithm::RS512,
            "HS256" => jsonwebtoken::Algorithm::HS256,
            "HS384" => jsonwebtoken::Algorithm::HS384,
            "HS512" => jsonwebtoken::Algorithm::HS512,
            "ES256" => jsonwebtoken::Algorithm::ES256,
            "ES384" => jsonwebtoken::Algorithm::ES384,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported JWT algorithm: {}. Supported: RS256, RS384, RS512, HS256, HS384, HS512, ES256, ES384",
                    algorithm_str
                ));
            }
        };

        let public_key = std::env::var("JWT_PUBLIC_KEY").ok();
        let public_key_path = std::env::var("JWT_PUBLIC_KEY_PATH").ok();
        let jwks_url = std::env::var("JWT_JWKS_URL").ok();
        let tenant_claim =
            std::env::var("JWT_TENANT_CLAIM").unwrap_or_else(|_| "tenant_id".to_string());
        let required_claims_str =
            std::env::var("JWT_REQUIRED_CLAIMS").unwrap_or_else(|_| "sub".to_string());
        let required_claims: Vec<String> = required_claims_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut validate_exp = std::env::var("JWT_VALIDATE_EXP")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);

        if !validate_exp && (env_bool("CALICHE_REQUIRE_AUTH") || is_production_profile()) {
            tracing::warn!(
                target: "caliche_server::jwt_auth",
                "JWT_VALIDATE_EXP=false is ignored when CALICHE_REQUIRE_AUTH or \
                 CALICHE_ENV=production is set; expiration validation remains enabled"
            );
            validate_exp = true;
        }

        if public_key.is_none() && public_key_path.is_none() && jwks_url.is_none() {
            return Err(anyhow::anyhow!(
                "JWT authentication enabled but no key source provided. Set JWT_PUBLIC_KEY, JWT_PUBLIC_KEY_PATH, or JWT_JWKS_URL"
            ));
        }

        let audience = csv_env("JWT_AUDIENCE");
        let issuer = csv_env("JWT_ISSUER");
        if audience.is_empty() || issuer.is_empty() {
            return Err(anyhow::anyhow!(
                "JWT authentication enabled but JWT_AUDIENCE and JWT_ISSUER are required"
            ));
        }

        Ok(Self {
            enabled,
            algorithm,
            public_key,
            public_key_path,
            jwks_url,
            tenant_claim,
            required_claims,
            validate_exp,
            audience,
            issuer,
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

/// JWT authentication state (shared across requests)
#[derive(Clone, Debug)]
pub struct JwtAuthState {
    config: JwtAuthConfig,
    decoding_key: Option<DecodingKey>,
    jwks_cache: Option<Arc<JwksCache>>,
}

impl JwtAuthState {
    pub fn new(config: JwtAuthConfig) -> anyhow::Result<Self> {
        if !config.enabled {
            return Ok(Self {
                config,
                decoding_key: None,
                jwks_cache: None,
            });
        }

        if config.audience.is_empty() || config.issuer.is_empty() {
            return Err(anyhow::anyhow!(
                "JWT authentication enabled but JWT_AUDIENCE and JWT_ISSUER are required"
            ));
        }

        if config.public_key.is_none()
            && config.public_key_path.is_none()
            && config.jwks_url.is_none()
        {
            return Err(anyhow::anyhow!(
                "JWT authentication enabled but no key source provided. Set public_key, public_key_path, or jwks_url"
            ));
        }

        let decoding_key = if let Some(ref public_key) = config.public_key {
            Some(
                DecodingKey::from_rsa_pem(public_key.as_bytes())
                    .or_else(|_| DecodingKey::from_ec_pem(public_key.as_bytes()))
                    .or_else(|_| {
                        let secret_key = DecodingKey::from_secret(public_key.as_bytes());
                        Ok::<DecodingKey, jsonwebtoken::errors::Error>(secret_key)
                    })
                    .map_err(|e| anyhow::anyhow!("Failed to create decoding key: {e}"))?,
            )
        } else if let Some(ref path) = config.public_key_path {
            let key_data = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("Failed to read public key file {path}: {e}"))?;
            Some(
                DecodingKey::from_rsa_pem(key_data.as_bytes())
                    .or_else(|_| DecodingKey::from_ec_pem(key_data.as_bytes()))
                    .or_else(|_| {
                        let secret_key = DecodingKey::from_secret(key_data.as_bytes());
                        Ok::<DecodingKey, jsonwebtoken::errors::Error>(secret_key)
                    })
                    .map_err(|e| anyhow::anyhow!("Failed to create decoding key from file: {e}"))?,
            )
        } else {
            None
        };

        let jwks_cache = match config.jwks_url.as_deref() {
            Some(url) => Some(Arc::new(JwksCache::try_new(url.to_string(), 3600)?)),
            None => None,
        };

        Ok(Self {
            config,
            decoding_key,
            jwks_cache,
        })
    }

    /// Extract and validate JWT token from headers
    pub async fn extract_user_context(
        &self,
        headers: &HeaderMap,
    ) -> anyhow::Result<Option<UserContext>> {
        if !self.config.enabled {
            return Ok(None);
        }

        let auth_header = headers
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .ok_or_else(|| anyhow::anyhow!("Missing Authorization header"))?;

        if !auth_header.starts_with("Bearer ") {
            return Err(anyhow::anyhow!("Invalid Authorization header format"));
        }

        let token = &auth_header[7..];

        let header = decode_header(token)?;
        let kid = header.kid.as_deref();

        let decoding_key = if let Some(ref jwks) = self.jwks_cache {
            if let Some(kid) = kid {
                jwks.get_key(kid)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("Key ID {kid} not found in JWKS"))?
            } else {
                return Err(anyhow::anyhow!("Token missing kid (key ID)"));
            }
        } else if let Some(ref key) = self.decoding_key {
            key.clone()
        } else {
            return Err(anyhow::anyhow!("No decoding key available"));
        };

        let mut validation = Validation::new(self.config.algorithm);
        validation.validate_exp = self.config.validate_exp;
        validation.set_audience(
            &self
                .config
                .audience
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        validation.set_issuer(
            &self
                .config
                .issuer
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        validation.required_spec_claims = self
            .config
            .required_claims
            .iter()
            .cloned()
            .collect::<HashSet<String>>();

        let token_data = decode::<serde_json::Value>(token, &decoding_key, &validation)
            .map_err(|e| anyhow::anyhow!("JWT validation failed: {e}"))?;

        let user_context =
            UserContext::from_claims(&token_data.claims, Some(self.config.tenant_claim.as_str()))?;

        Ok(Some(user_context))
    }
}

/// Optional UserContext extractor - returns None if JWT is disabled or missing
pub struct OptionalUserContext(pub Option<UserContext>);

impl<S> axum::extract::FromRequestParts<S> for OptionalUserContext
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        if let Some(user) = parts.extensions.get::<UserContext>().cloned() {
            return Ok(OptionalUserContext(Some(user)));
        }

        if let Some(auth_state) = parts.extensions.get::<JwtAuthState>() {
            match auth_state.extract_user_context(&parts.headers).await {
                Ok(context) => Ok(OptionalUserContext(context)),
                Err(_) => Ok(OptionalUserContext(None)),
            }
        } else {
            Ok(OptionalUserContext(None))
        }
    }
}

impl<S> axum::extract::FromRequestParts<S> for UserContext
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        if let Some(user) = parts.extensions.get::<UserContext>().cloned() {
            return Ok(user);
        }

        let auth_state = parts.extensions.get::<JwtAuthState>().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "JWT authentication not configured".to_string(),
            )
        })?;

        match auth_state.extract_user_context(&parts.headers).await {
            Ok(Some(context)) => Ok(context),
            Ok(None) => Err((StatusCode::UNAUTHORIZED, AUTH_REQUIRED_MESSAGE.to_string())),
            Err(e) => {
                tracing::warn!(
                    target: "caliche_server::jwt_auth",
                    error = %e,
                    "JWT validation failed for required UserContext extractor"
                );
                Err((StatusCode::UNAUTHORIZED, INVALID_TOKEN_MESSAGE.to_string()))
            }
        }
    }
}

fn security_policy_from_request(request: &axum::extract::Request) -> Option<Arc<SecurityPolicy>> {
    request.extensions().get::<Arc<SecurityPolicy>>().cloned()
}

/// Middleware that forces authentication according to the startup [`SecurityPolicy`].
pub async fn require_jwt_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = request.uri().path();
    let method = request.method();

    let Some(policy) = security_policy_from_request(&request) else {
        tracing::error!(
            target: "caliche_server::jwt_auth",
            path = %path,
            "request reached auth middleware without SecurityPolicy extension"
        );
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "authentication misconfigured",
        )
            .into_response();
    };

    if !policy.path_requires_auth(path, method) {
        return next.run(request).await;
    }

    if path == "/metrics" {
        let auth_header = request
            .headers()
            .get("Authorization")
            .or_else(|| request.headers().get("authorization"))
            .and_then(|value| value.to_str().ok());
        if policy.metrics_bearer_authorized(auth_header) {
            return next.run(request).await;
        }
    }

    let Some(auth_state) = request.extensions().get::<JwtAuthState>().cloned() else {
        if policy.orchestration_requires_auth(path) {
            tracing::warn!(
                target: "caliche_server::jwt_auth",
                path = %path,
                "refusing unauthenticated orchestration request; enable JWT or set \
                 CALICHE_ALLOW_UNAUTH_ORCHESTRATION for local development"
            );
            return (
                StatusCode::UNAUTHORIZED,
                ORCHESTRATION_AUTH_REQUIRED_MESSAGE,
            )
                .into_response();
        }
        if !policy.jwt_enabled && !policy.require_auth {
            tracing::debug!(
                target: "caliche_server::jwt_auth",
                path = %path,
                "allowing protected route without JWT in development mode"
            );
            return next.run(request).await;
        }
        tracing::warn!(
            target: "caliche_server::jwt_auth",
            path = %path,
            "protected route reached without JWT middleware state"
        );
        return (StatusCode::UNAUTHORIZED, AUTH_REQUIRED_MESSAGE).into_response();
    };

    match auth_state.extract_user_context(request.headers()).await {
        Ok(Some(user)) => {
            let mut request = request;
            request.extensions_mut().insert(user);
            next.run(request).await
        }
        Ok(None) => (StatusCode::UNAUTHORIZED, AUTH_REQUIRED_MESSAGE).into_response(),
        Err(e) => {
            tracing::warn!(
                target: "caliche_server::jwt_auth",
                path = %path,
                error = %e,
                "JWT validation failed"
            );
            (StatusCode::UNAUTHORIZED, INVALID_TOKEN_MESSAGE).into_response()
        }
    }
}
