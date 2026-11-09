//! Advanced features command handlers.
//!
//! Handles reasoning patterns, chains, documents, multi-agent patterns, and workflows.

use crate::cancellation::CancellationToken;
use crate::commands::Args;
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use crate::traits::factory::ServiceConfig;
use anyhow::Result;
use async_trait::async_trait;
use caliche::AgentService;
use std::ops::Deref;
use std::sync::Arc;

/// Arguments for advanced features (shared reference to CLI args).
#[derive(Debug, Clone)]
pub struct AdvancedFeaturesArgs(pub Arc<Args>);

impl Deref for AdvancedFeaturesArgs {
    type Target = Args;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

/// Handler for advanced features.
pub struct AdvancedFeaturesHandler;

#[async_trait]
impl CommandHandler for AdvancedFeaturesHandler {
    type Args = AdvancedFeaturesArgs;

    async fn execute(
        &self,
        args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        // Create a service for advanced features
        let mut service = ctx
            .service_factory
            .create_service(ServiceConfig::default())
            .await
            .map_err(|e| CliError::Command(format!("Failed to create service: {}", e)))?;

        Box::pin(run_advanced_features(
            args,
            &mut service,
            ctx.cancellation_token,
        ))
        .await
        .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "advanced"
    }

    fn description(&self) -> &'static str {
        "Advanced features: reasoning patterns, chains, documents, multi-agent, workflows"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Execute advanced agent features including reasoning patterns, chains, document processing, multi-agent coordination, and workflows.",
        )
    }
}

#[path = "advanced_run.rs"]
mod advanced_run;

/// Run advanced features implementation.
///
/// `cancellation_token`, when provided, is checked between major feature
/// branches and is used to race long-running reasoning/chain/multi-agent work
/// against Ctrl+C so the user can interrupt a stuck advanced flow.
pub async fn run_advanced_features(
    args: AdvancedFeaturesArgs,
    service: &mut AgentService,
    cancellation_token: Option<&CancellationToken>,
) -> Result<()> {
    advanced_run::run_advanced_features_impl(args, service, cancellation_token).await
}

/// Race `fut` against `cancellation_token.cancelled()` so long-running advanced
/// flows can be interrupted by Ctrl+C. When cancelled, the in-flight future is
/// dropped (async cancellation); callers should treat this as best-effort.
pub(super) async fn run_with_cancellation<F, T>(
    cancellation_token: Option<&CancellationToken>,
    fut: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send,
{
    match cancellation_token {
        Some(token) => {
            tokio::select! {
                biased;
                _ = token.cancelled() => Err(anyhow::anyhow!("Operation cancelled by user")),
                result = fut => result,
            }
        }
        None => fut.await,
    }
}
