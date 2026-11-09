//! Tool credential types and resolution logic
//!
//! Provides a standardized way to handle credentials for tools with a priority chain:
//! 1. Construction-time credentials (highest priority, cannot be overridden)
//! 2. Tool arguments (runtime credentials passed in tool call)
//! 3. `UserContext.custom_claims` (credentials from JWT/user context)
//! 4. None (fail if required)

use crate::utils::user_context::UserContext;
use anyhow::{Context, Result};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Credential types supported by tools
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCredential {
    /// Bearer token authentication (Authorization: Bearer <token>)
    BearerToken(String),
    /// Basic authentication (Authorization: Basic <base64(username:password)>)
    BasicAuth { username: String, password: String },
    /// API key authentication (custom header)
    ApiKey { header_name: String, value: String },
    /// OAuth token (Authorization: <`token_type`> <token>)
    OAuth {
        token: String,
        token_type: Option<String>, // Defaults to "Bearer" if None
    },
    /// Custom headers for authentication
    Custom(HashMap<String, String>),
}

impl ToolCredential {
    /// Apply this credential to a reqwest request builder
    pub fn apply_to_request(
        &self,
        request_builder: reqwest::RequestBuilder,
    ) -> Result<reqwest::RequestBuilder> {
        match self {
            Self::BearerToken(token) => Ok(request_builder.bearer_auth(token)),
            Self::BasicAuth { username, password } => {
                Ok(request_builder.basic_auth(username, Some(password)))
            }
            Self::ApiKey { header_name, value } => {
                let header_name_h = reqwest::header::HeaderName::from_bytes(header_name.as_bytes())
                    .with_context(|| format!("Invalid header name '{header_name}'"))?;
                let header_value = reqwest::header::HeaderValue::from_str(value)
                    .context("Invalid header value")?;
                Ok(request_builder.header(header_name_h, header_value))
            }
            Self::OAuth { token, token_type } => {
                let auth_value = if let Some(tt) = token_type {
                    format!("{tt} {token}")
                } else {
                    format!("Bearer {token}")
                };
                let header_value = reqwest::header::HeaderValue::from_str(&auth_value)
                    .context("Invalid OAuth token")?;
                Ok(request_builder.header(
                    reqwest::header::HeaderName::from_static("authorization"),
                    header_value,
                ))
            }
            Self::Custom(headers) => {
                let mut builder = request_builder;
                for (key, value) in headers {
                    let header_name_h = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                        .with_context(|| format!("Invalid header name '{key}'"))?;
                    let header_value = reqwest::header::HeaderValue::from_str(value)
                        .with_context(|| format!("Invalid header value for '{key}'"))?;
                    builder = builder.header(header_name_h, header_value);
                }
                Ok(builder)
            }
        }
    }

    /// Get the credential as a `HashMap` of headers (for non-reqwest use cases)
    #[must_use]
    pub fn as_headers(&self) -> HashMap<String, String> {
        match self {
            Self::BearerToken(token) => {
                let mut map = HashMap::new();
                map.insert("Authorization".to_string(), format!("Bearer {token}"));
                map
            }
            Self::BasicAuth { username, password } => {
                let mut map = HashMap::new();
                let credentials = BASE64.encode(format!("{username}:{password}"));
                map.insert("Authorization".to_string(), format!("Basic {credentials}"));
                map
            }
            Self::ApiKey { header_name, value } => {
                let mut map = HashMap::new();
                map.insert(header_name.clone(), value.clone());
                map
            }
            Self::OAuth { token, token_type } => {
                let mut map = HashMap::new();
                let auth_value = if let Some(tt) = token_type {
                    format!("{tt} {token}")
                } else {
                    format!("Bearer {token}")
                };
                map.insert("Authorization".to_string(), auth_value);
                map
            }
            Self::Custom(headers) => headers.clone(),
        }
    }
}

/// Helper to resolve credentials from multiple sources with priority chain
pub struct CredentialResolver;

impl CredentialResolver {
    /// Resolve credentials using the priority chain:
    /// 1. Construction-time credentials (if provided, these are final)
    /// 2. Tool arguments (runtime credentials)
    /// 3. `UserContext.custom_claims` (credentials from JWT)
    /// 4. None
    ///
    /// Returns None if no credentials found, Some(credential) if found
    #[must_use]
    pub fn resolve(
        construction_credential: Option<&ToolCredential>,
        tool_arguments: &serde_json::Value,
        user_context: Option<&UserContext>,
    ) -> Option<ToolCredential> {
        // Priority 1: Construction-time credentials (highest priority, cannot be overridden)
        if let Some(cred) = construction_credential {
            return Some(cred.clone());
        }

        // Priority 2: Tool arguments
        if let Some(cred) = Self::from_tool_arguments(tool_arguments) {
            return Some(cred);
        }

        // Priority 3: UserContext.custom_claims
        if let Some(cred) = Self::from_user_context(user_context) {
            return Some(cred);
        }

        // Priority 4: None
        None
    }

    /// Extract credentials from tool arguments
    fn from_tool_arguments(arguments: &serde_json::Value) -> Option<ToolCredential> {
        // Check for credential_type field
        let cred_type = arguments.get("credential_type")?.as_str()?;

        match cred_type {
            "bearer" | "basic" | "api_key" => token_credential(cred_type, arguments),
            "oauth" | "custom" => structured_credential(cred_type, arguments),
            _ => None,
        }
    }

    /// Extract credentials from `UserContext.custom_claims`
    fn from_user_context(context: Option<&UserContext>) -> Option<ToolCredential> {
        let context = context?;

        // Look for credential information in custom_claims
        // Common patterns:
        // - api_key: string -> ApiKey with default header
        // - bearer_token: string -> BearerToken
        // - credentials: object with type and value
        // - tool_credentials: object with credential info

        if let Some(api_key) = context.custom_claims.get("api_key")
            && let Some(key_str) = api_key.as_str()
        {
            return Some(ToolCredential::ApiKey {
                header_name: "X-API-Key".to_string(),
                value: key_str.to_string(),
            });
        }

        if let Some(bearer_token) = context.custom_claims.get("bearer_token")
            && let Some(token_str) = bearer_token.as_str()
        {
            return Some(ToolCredential::BearerToken(token_str.to_string()));
        }

        if let Some(credentials) = context.custom_claims.get("tool_credentials")
            && let Some(cred_obj) = credentials.as_object()
            && let Some(cred_type) = cred_obj.get("type").and_then(|v| v.as_str())
        {
            return credential_from_claim_object(cred_type, cred_obj);
        }

        None
    }
}

fn token_credential(cred_type: &str, arguments: &serde_json::Value) -> Option<ToolCredential> {
    match cred_type {
        "bearer" => {
            let token = arguments.get("credential_token")?.as_str()?;
            Some(ToolCredential::BearerToken(token.to_string()))
        }
        "basic" => {
            let username = arguments.get("credential_username")?.as_str()?.to_string();
            let password = arguments.get("credential_token")?.as_str()?.to_string();
            Some(ToolCredential::BasicAuth { username, password })
        }
        "api_key" => {
            let header_name = arguments
                .get("credential_header_name")
                .and_then(|value| value.as_str())
                .unwrap_or("X-API-Key")
                .to_string();
            let value = arguments.get("credential_token")?.as_str()?;
            Some(ToolCredential::ApiKey {
                header_name,
                value: value.to_string(),
            })
        }
        _ => None,
    }
}

fn structured_credential(cred_type: &str, arguments: &serde_json::Value) -> Option<ToolCredential> {
    match cred_type {
        "oauth" => {
            let token = arguments.get("credential_token")?.as_str()?;
            let token_type = arguments
                .get("credential_token_type")
                .and_then(|value| value.as_str())
                .map(std::string::ToString::to_string);
            Some(ToolCredential::OAuth {
                token: token.to_string(),
                token_type,
            })
        }
        "custom" => custom_headers(arguments.get("credential_headers")?.as_object()?),
        _ => None,
    }
}

fn custom_headers(
    headers_obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<ToolCredential> {
    let mut headers = HashMap::new();
    for (key, value) in headers_obj {
        if let Some(text) = value.as_str() {
            headers.insert(key.clone(), text.to_string());
        }
    }
    if headers.is_empty() {
        None
    } else {
        Some(ToolCredential::Custom(headers))
    }
}

fn credential_from_claim_object(
    cred_type: &str,
    cred_obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<ToolCredential> {
    match cred_type {
        "bearer" => cred_obj
            .get("token")
            .and_then(|value| value.as_str())
            .map(|token| ToolCredential::BearerToken(token.to_string())),
        "basic" => {
            let username = cred_obj.get("username").and_then(|value| value.as_str())?;
            let password = cred_obj.get("password").and_then(|value| value.as_str())?;
            Some(ToolCredential::BasicAuth {
                username: username.to_string(),
                password: password.to_string(),
            })
        }
        "api_key" => {
            let value = cred_obj.get("value").and_then(|value| value.as_str())?;
            let header_name = cred_obj
                .get("header_name")
                .and_then(|value| value.as_str())
                .unwrap_or("X-API-Key")
                .to_string();
            Some(ToolCredential::ApiKey {
                header_name,
                value: value.to_string(),
            })
        }
        "oauth" => {
            let token = cred_obj.get("token").and_then(|value| value.as_str())?;
            let token_type = cred_obj
                .get("token_type")
                .and_then(|value| value.as_str())
                .map(std::string::ToString::to_string);
            Some(ToolCredential::OAuth {
                token: token.to_string(),
                token_type,
            })
        }
        "custom" => cred_obj
            .get("headers")
            .and_then(|value| value.as_object())
            .and_then(custom_headers),
        _ => None,
    }
}

/// Helper function to generate JSON schema for credential parameters
/// Tools can include this in their `parameters()` method to add credential support
#[must_use]
pub fn credential_parameters() -> serde_json::Value {
    serde_json::json!({
        "credential_type": {
            "type": "string",
            "enum": ["bearer", "basic", "api_key", "oauth", "custom"],
            "description": "Type of credential to use for authentication"
        },
        "credential_token": {
            "type": "string",
            "description": "Authentication token/password (required for bearer, basic, api_key, or oauth credential_type)"
        },
        "credential_username": {
            "type": "string",
            "description": "Username for basic authentication (required if credential_type is 'basic')"
        },
        "credential_header_name": {
            "type": "string",
            "description": "Header name for API key authentication (default: 'X-API-Key', used if credential_type is 'api_key')",
            "default": "X-API-Key"
        },
        "credential_token_type": {
            "type": "string",
            "description": "Token type for OAuth authentication (default: 'Bearer', used if credential_type is 'oauth')",
            "default": "Bearer"
        },
        "credential_headers": {
            "type": "object",
            "description": "Custom headers for authentication (required if credential_type is 'custom')"
        }
    })
}
