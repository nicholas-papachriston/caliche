//! Static router: uses request-provided or default (provider, model).

use super::{ProviderRouter, ProviderSelection, RoutingContext};
use anyhow::Result;
use async_trait::async_trait;

/// Router that returns request-provided or default provider/model.
/// Zero behavior change from today when used as the routing layer.
#[derive(Debug, Clone, Default)]
pub struct StaticRouter;

impl StaticRouter {
    /// Create a static router.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProviderRouter for StaticRouter {
    async fn select(&self, ctx: &RoutingContext) -> Result<ProviderSelection> {
        let provider = ctx.requested_provider.unwrap_or(ctx.default_provider);
        let model = ctx
            .requested_model
            .clone()
            .unwrap_or_else(|| ctx.default_model.clone());

        Ok(ProviderSelection { provider, model })
    }
}
