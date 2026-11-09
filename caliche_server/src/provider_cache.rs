//! Provider and tool-registry caching helpers.

use caliche::ProviderType;
#[cfg(feature = "router")]
use caliche::agent::service::resolve_model_for_provider_with_discovery;
use caliche::agent::tools::ToolRegistry;
use caliche::llm::Provider;
use std::sync::Arc;

use crate::state::{AppState, ProviderCacheKey, ToolRegistryCacheKey};

/// Select provider and model for a request (uses router when configured).
#[cfg(feature = "router")]
pub async fn select_provider(
    state: &AppState,
    requested_provider: Option<ProviderType>,
    requested_model: Option<String>,
    tenant_id: Option<&str>,
) -> anyhow::Result<caliche::llm::router::ProviderSelection> {
    use caliche::llm::router::RoutingContext;
    use std::collections::HashMap;

    let effective_requested_provider = requested_provider.unwrap_or(state.default_provider);
    let requested_model = match requested_model {
        Some(model) => Some(
            resolve_model_for_provider_with_discovery(effective_requested_provider, Some(model))
                .await?,
        ),
        None if requested_provider.is_some()
            && effective_requested_provider != state.default_provider =>
        {
            Some(
                resolve_model_for_provider_with_discovery(effective_requested_provider, None)
                    .await?,
            )
        }
        None => None,
    };

    let ctx = RoutingContext {
        requested_provider,
        requested_model,
        tenant_id: tenant_id.map(String::from),
        default_provider: state.default_provider,
        default_model: state.default_model.clone(),
        metadata: HashMap::new(),
    };

    if let Some(ref router) = state.provider_router {
        router.select(&ctx).await
    } else {
        let provider = ctx.requested_provider.unwrap_or(ctx.default_provider);
        let model = match ctx.requested_model {
            Some(model) => resolve_model_for_provider_with_discovery(provider, Some(model)).await?,
            None if provider == ctx.default_provider => ctx.default_model.clone(),
            None => resolve_model_for_provider_with_discovery(provider, None).await?,
        };
        Ok(caliche::llm::router::ProviderSelection { provider, model })
    }
}

/// Get or create a Provider instance from cache
/// This caches expensive-to-create provider instances (network clients, auth setup)
pub async fn get_or_create_provider(
    state: &AppState,
    provider_type: ProviderType,
    model: String,
) -> anyhow::Result<Arc<dyn Provider>> {
    let key = ProviderCacheKey {
        provider: provider_type,
        model: model.clone(),
    };

    if let Some(provider) = state.provider_cache.get(&key).await {
        state.cache_metrics.provider_cache.record_hit();
        tracing::debug!("Using cached provider for {:?}", key);
        return Ok(provider);
    }

    state.cache_metrics.provider_cache.record_miss();
    tracing::info!("Creating new provider for {:?}", key);
    let cred_provider = Arc::new(caliche::traits::credentials::EnvironmentCredentialProvider::new())
        as Arc<dyn caliche::traits::credentials::CredentialProvider>;

    let provider_arc = caliche::agent::service::ProviderFactory::create(
        key.provider,
        model,
        cred_provider,
        None::<&str>,
        None,
        None,
        None,
    )
    .await?;

    state
        .provider_cache
        .insert(key, Arc::clone(&provider_arc))
        .await;

    state.cache_metrics.check_hit_rate(70.0);

    Ok(provider_arc)
}

/// Get or create a ToolRegistry instance from cache
/// This caches expensive-to-build tool registries (tool initialization)
pub async fn get_or_create_tool_registry(
    state: &AppState,
    provider_type: ProviderType,
    tool_names: Option<Vec<String>>,
) -> anyhow::Result<Arc<ToolRegistry>> {
    let key = ToolRegistryCacheKey {
        provider: provider_type,
        tools: tool_names.clone(),
    };

    if let Some(registry) = state.tool_registry_cache.get(&key).await {
        state.cache_metrics.tool_registry_cache.record_hit();
        tracing::debug!("Using cached tool registry for {:?}", key);
        return Ok(registry);
    }

    state.cache_metrics.tool_registry_cache.record_miss();
    tracing::info!("Creating new tool registry for {:?}", key);
    let mut builder = caliche::agent::service::ToolRegistryBuilder::new(key.provider);

    if tool_names.is_none() {
        builder = builder.use_all_default_tools();
    } else {
        builder = builder.with_tool_names(tool_names);
    }

    #[cfg(feature = "tools-rag")]
    {
        builder = with_shared_rag_tool(builder, state.shared_rag_tool.as_ref());
    }

    let registry_arc: Arc<ToolRegistry> = builder.build().await?;

    state
        .tool_registry_cache
        .insert(key, registry_arc.clone())
        .await;

    state.cache_metrics.check_hit_rate(70.0);

    Ok(registry_arc)
}

/// Give server agents the process-wide RAG tool, when the server has one.
#[cfg(feature = "tools-rag")]
pub(crate) fn with_shared_rag_tool(
    builder: caliche::agent::service::ToolRegistryBuilder,
    rag_tool: Option<&Arc<caliche::agent::tools::RagTool>>,
) -> caliche::agent::service::ToolRegistryBuilder {
    match rag_tool {
        Some(tool) => builder.with_custom_tools(vec![
            Arc::clone(tool) as Arc<dyn caliche::agent::tools::Tool>
        ]),
        None => builder,
    }
}

#[cfg(all(test, feature = "tools-rag"))]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use caliche::agent::service::ToolRegistryBuilder;
    use caliche::agent::tools::RagTool;
    use caliche::llm::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};

    #[derive(Debug)]
    struct FixedEmbedding;

    #[async_trait]
    impl EmbeddingProvider for FixedEmbedding {
        async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
            Ok(EmbeddingResponse {
                embeddings: request.texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect(),
                usage: None,
                model: "fixed".to_string(),
            })
        }

        fn name(&self) -> &str {
            "fixed"
        }
    }

    #[tokio::test]
    async fn default_tool_registry_contains_shared_rag_tool() -> anyhow::Result<()> {
        let rag_tool = Arc::new(RagTool::new(Arc::new(FixedEmbedding), None::<String>));
        let builder = ToolRegistryBuilder::new(ProviderType::OpenAI).use_all_default_tools();
        let registry = with_shared_rag_tool(builder, Some(&rag_tool))
            .build()
            .await?;

        assert!(registry.get("rag").is_some());
        Ok(())
    }

    #[tokio::test]
    async fn default_tool_registry_has_no_rag_tool_without_shared_tool() -> anyhow::Result<()> {
        let builder = ToolRegistryBuilder::new(ProviderType::OpenAI).use_all_default_tools();
        let registry = with_shared_rag_tool(builder, None).build().await?;

        assert!(registry.get("rag").is_none());
        Ok(())
    }
}
