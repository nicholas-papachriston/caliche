use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// User context extracted from JWT and injected into requests
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
    /// Create a new [`UserContext`] from JWT claims.
    ///
    /// `tenant_claim` is the claim name from `JWT_TENANT_CLAIM` (default `tenant_id`).
    pub fn from_claims(
        claims: &serde_json::Value,
        tenant_claim: Option<&str>,
    ) -> anyhow::Result<Self> {
        let user_id = claims
            .get("sub")
            .or_else(|| claims.get("user_id"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required claim: sub or user_id"))?
            .to_string();

        let tenant_id = tenant_claim
            .and_then(|name| claims.get(name).and_then(|v| v.as_str()))
            .or_else(|| claims.get("tenant_id").and_then(|v| v.as_str()))
            .or_else(|| claims.get("org_id").and_then(|v| v.as_str()))
            .or_else(|| claims.get("organization_id").and_then(|v| v.as_str()))
            .map(str::to_string);

        let email = claims
            .get("email")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let roles = claims
            .get("roles")
            .or_else(|| claims.get("role"))
            .and_then(|v| {
                if let Some(arr) = v.as_array() {
                    Some(
                        arr.iter()
                            .filter_map(|r| r.as_str().map(|s| s.to_string()))
                            .collect(),
                    )
                } else if let Some(s) = v.as_str() {
                    Some(vec![s.to_string()])
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let mut custom_claims = HashMap::new();
        if let Some(obj) = claims.as_object() {
            for (key, value) in obj {
                if !matches!(
                    key.as_str(),
                    "sub"
                        | "user_id"
                        | "tenant_id"
                        | "org_id"
                        | "organization_id"
                        | "email"
                        | "roles"
                        | "role"
                        | "iat"
                        | "exp"
                        | "nbf"
                        | "iss"
                        | "aud"
                ) && tenant_claim.is_none_or(|claim| key != claim)
                {
                    custom_claims.insert(key.clone(), value.clone());
                }
            }
        }

        Ok(Self {
            user_id,
            tenant_id,
            email,
            roles,
            custom_claims,
        })
    }

    /// Get tenant ID, panicking if not present (for use when tenant is required)
    pub fn tenant_id_required(&self) -> anyhow::Result<&str> {
        self.tenant_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Tenant ID is required but not present in JWT"))
    }
}
