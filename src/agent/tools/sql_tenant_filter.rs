//! Tenant filtering settings for the SQL tools.
//!
//! The SQL tools run every query on one shared connection pool, so they cannot
//! keep one tenant's rows away from another. When tenant filtering is enabled,
//! [`TenantFilterConfig::check`] refuses every query instead of running it
//! without isolation.

use crate::utils::user_context::UserContext;
use anyhow::bail;

/// Isolation strategy named by `SQL_TENANT_ISOLATION_STRATEGY`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IsolationStrategy {
    /// `row_level`: not supported. Rewriting the query text is not a security boundary.
    RowLevel,
    /// `schema`: not supported. A query can name another schema.
    Schema,
    /// `database`: not supported. The SQL tools have no per-tenant connections.
    Database,
    /// `none`: no isolation.
    None,
    /// Any other value. Refused so that a typo does not run queries unisolated.
    Unrecognized(String),
}

impl IsolationStrategy {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "row_level" => Self::RowLevel,
            "schema" => Self::Schema,
            "database" => Self::Database,
            "none" => Self::None,
            _ => Self::Unrecognized(value.to_string()),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::RowLevel => "row_level",
            Self::Schema => "schema",
            Self::Database => "database",
            Self::None => "none",
            Self::Unrecognized(value) => value,
        }
    }
}

/// Configuration for SQL tenant filtering
#[derive(Debug, Clone)]
pub struct TenantFilterConfig {
    pub enabled: bool,
    pub strategy: IsolationStrategy,
    pub extract_from_claim: String,
}

impl Default for TenantFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: IsolationStrategy::RowLevel,
            extract_from_claim: "tenant_id".to_string(),
        }
    }
}

impl TenantFilterConfig {
    #[must_use]
    pub fn from_env() -> Self {
        let enabled = std::env::var("SQL_TENANT_FILTERING_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        let strategy = std::env::var("SQL_TENANT_ISOLATION_STRATEGY")
            .map_or(IsolationStrategy::RowLevel, |value| {
                IsolationStrategy::parse(&value)
            });

        let tenant_claim =
            std::env::var("SQL_TENANT_JWT_CLAIM").unwrap_or_else(|_| "tenant_id".to_string());

        Self {
            enabled,
            strategy,
            extract_from_claim: tenant_claim,
        }
    }

    /// Extract tenant ID from user context
    #[must_use]
    pub fn extract_tenant_id(&self, context: Option<&UserContext>) -> Option<String> {
        if !self.enabled {
            return None;
        }

        context.and_then(|ctx| {
            // Try tenant_id field first
            ctx.tenant_id.clone().or_else(|| {
                // Fall back to custom claims
                ctx.custom_claims
                    .get(&self.extract_from_claim)
                    .and_then(|v| v.as_str())
                    .map(std::string::ToString::to_string)
            })
        })
    }

    /// Returns `Ok` only when tenant filtering is disabled.
    ///
    /// # Errors
    ///
    /// When filtering is enabled, returns the reason the query is refused: an
    /// unrecognized strategy, a request without a tenant id, or a strategy that
    /// the SQL tools cannot enforce.
    pub fn check(&self, context: Option<&UserContext>) -> anyhow::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if let IsolationStrategy::Unrecognized(value) = &self.strategy {
            bail!(
                "SQL query refused: SQL_TENANT_ISOLATION_STRATEGY value '{value}' is not recognized"
            );
        }
        if self.extract_tenant_id(context).is_none() {
            bail!(
                "SQL query refused: tenant filtering is enabled and the request has no tenant id"
            );
        }
        bail!(
            "SQL query refused: tenant isolation strategy '{}' is not supported. \
             Tenant isolation needs a separate database connection for each tenant, \
             which the SQL tools do not provide",
            self.strategy.name()
        )
    }
}
