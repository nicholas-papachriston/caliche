//! API-key authentication and shared auth context.

use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthContext {
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
}

type KeyDigest = [u8; 32];

pub struct ApiKeyValidator {
    keys: Arc<DashMap<KeyDigest, AuthContext>>,
    pepper: Vec<u8>,
}

impl ApiKeyValidator {
    pub fn new() -> Self {
        Self::with_pepper(api_key_pepper_from_env())
    }

    fn with_pepper(pepper: Vec<u8>) -> Self {
        Self {
            keys: Arc::new(DashMap::new()),
            pepper,
        }
    }

    fn digest_key(&self, api_key: &str) -> KeyDigest {
        let mut hasher = Sha256::new();
        hasher.update(b"caliche-api-key-v1\0");
        hasher.update(&self.pepper);
        hasher.update(api_key.as_bytes());
        hasher.finalize().into()
    }

    pub fn add_key(&self, api_key: impl Into<String>, context: AuthContext) {
        let digest = self.digest_key(&api_key.into());
        self.keys.insert(digest, context);
    }

    pub fn validate(&self, api_key: &str) -> Result<AuthContext> {
        let digest = self.digest_key(api_key);
        self.keys
            .get(&digest)
            .map(|entry| entry.value().clone())
            .context("Invalid API key")
    }

    pub fn revoke(&self, api_key: &str) {
        let digest = self.digest_key(api_key);
        self.keys.remove(&digest);
    }
}

impl Default for ApiKeyValidator {
    fn default() -> Self {
        Self::new()
    }
}

fn api_key_pepper_from_env() -> Vec<u8> {
    match std::env::var("API_KEY_PEPPER") {
        Ok(pepper) => pepper.into_bytes(),
        Err(_) => {
            tracing::warn!(
                "API_KEY_PEPPER is not set; using the built-in default pepper (not suitable for production)"
            );
            b"caliche-default-api-key-pepper".to_vec()
        }
    }
}
