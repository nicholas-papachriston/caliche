//! Request-scoped identity fields shared by the core crate and API middleware.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identity attached to a request (user, tenant, roles, custom claims).
///
/// JWT claim parsing lives in `caliche_server` middleware. This type is the
/// portable payload the core library and tools consume after auth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub email: Option<String>,
    pub roles: Vec<String>,
    #[serde(flatten)]
    pub custom_claims: HashMap<String, serde_json::Value>,
}

impl UserContext {
    /// Build a context from an already-validated claims object.
    ///
    /// Prefer `caliche_server` middleware when the claims come from a JWT.
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            tenant_id: None,
            email: None,
            roles: Vec::new(),
            custom_claims: HashMap::new(),
        }
    }
}
