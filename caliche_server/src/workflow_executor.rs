use crate::{ProviderCacheKey, ToolRegistryCacheKey};
use anyhow::Result;
use async_trait::async_trait;
use caliche::agent::service::AgentService;
use caliche::agent::service::ProviderFactory;
use caliche::agent::service::ToolRegistryBuilder;
use caliche::agent::tools::ToolRegistry;
use caliche::llm::{EmbeddingProvider, Provider, ProviderType};
use caliche::observability::{CacheMetrics, ToolMetrics};
use caliche::orchestration::{
    ExecutionContext, WorkflowNodeExecutor, builtin_workflow_merge_custom,
    builtin_workflow_transform,
};
use caliche::traits::credentials::{CredentialProvider, EnvironmentCredentialProvider};
use moka::future::Cache;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct ApiWorkflowNodeExecutor {
    pub default_provider: ProviderType,
    pub default_model: String,
    pub provider_cache: Arc<Cache<ProviderCacheKey, Arc<dyn Provider>>>,
    pub tool_registry_cache: Arc<Cache<ToolRegistryCacheKey, Arc<ToolRegistry>>>,
    pub tool_metrics: Arc<ToolMetrics>,
    pub cache_metrics: Arc<CacheMetrics>,
    #[cfg(feature = "tools-rag")]
    pub shared_rag_tool: Option<Arc<caliche::agent::tools::RagTool>>,
}

impl ApiWorkflowNodeExecutor {
    async fn get_or_create_provider(
        &self,
        provider_type: ProviderType,
        model: String,
    ) -> Result<Arc<dyn Provider>> {
        let key = ProviderCacheKey {
            provider: provider_type,
            model: model.clone(),
        };

        if let Some(provider) = self.provider_cache.get(&key).await {
            self.cache_metrics.provider_cache.record_hit();
            return Ok(provider);
        }

        self.cache_metrics.provider_cache.record_miss();
        let cred_provider: Arc<dyn CredentialProvider> =
            Arc::new(EnvironmentCredentialProvider::new());

        let provider_arc = ProviderFactory::create(
            key.provider,
            model,
            cred_provider,
            None::<&str>,
            None,
            None,
            None,
        )
        .await?;

        self.provider_cache
            .insert(key, Arc::clone(&provider_arc))
            .await;
        self.cache_metrics.check_hit_rate(70.0);
        Ok(provider_arc)
    }

    async fn get_or_create_tool_registry(
        &self,
        provider_type: ProviderType,
        tool_names: Option<Vec<String>>,
    ) -> Result<Arc<ToolRegistry>> {
        let key = ToolRegistryCacheKey {
            provider: provider_type,
            tools: tool_names.clone(),
        };

        if let Some(registry) = self.tool_registry_cache.get(&key).await {
            self.cache_metrics.tool_registry_cache.record_hit();
            return Ok(registry);
        }

        self.cache_metrics.tool_registry_cache.record_miss();
        let mut builder = ToolRegistryBuilder::new(key.provider);

        if tool_names.is_none() {
            builder = builder.use_all_default_tools();
        } else {
            builder = builder.with_tool_names(tool_names);
        }

        #[cfg(feature = "tools-rag")]
        {
            builder =
                crate::provider_cache::with_shared_rag_tool(builder, self.shared_rag_tool.as_ref());
        }

        let registry_arc: Arc<ToolRegistry> = builder.build().await?;
        self.tool_registry_cache
            .insert(key, registry_arc.clone())
            .await;
        self.cache_metrics.check_hit_rate(70.0);
        Ok(registry_arc)
    }

    fn tools_from_context(&self, context: &ExecutionContext) -> Option<Vec<String>> {
        context
            .get_variable("__tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
    }
}

#[async_trait]
impl WorkflowNodeExecutor for ApiWorkflowNodeExecutor {
    async fn execute_agent(
        &self,
        prompt: &str,
        _agent_id: Option<&str>,
        _capability: Option<&str>,
        context: &ExecutionContext,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let provider_type = self.default_provider;
        let model = self.default_model.clone();
        let tools = self.tools_from_context(context);

        let provider = self
            .get_or_create_provider(provider_type, model.clone())
            .await?;
        let tool_registry = self
            .get_or_create_tool_registry(provider_type, tools)
            .await?;

        let embedding_provider: Option<Arc<dyn EmbeddingProvider>> = None;
        let mut service = AgentService::from_components(
            provider,
            tool_registry,
            provider_type,
            model.clone(),
            embedding_provider,
        )?;

        service.set_metrics(Arc::clone(&self.tool_metrics));

        let response = service.chat(prompt).await?;
        let text = response.text().to_string();

        let mut meta = HashMap::new();
        meta.insert(
            "provider".to_string(),
            Value::String(format!("{:?}", provider_type)),
        );
        meta.insert("model".to_string(), Value::String(model));

        Ok((Value::String(text), meta))
    }

    async fn transform(
        &self,
        function: &str,
        context: &ExecutionContext,
        last_output: Option<&Value>,
    ) -> Result<Value> {
        builtin_workflow_transform(function, context, last_output)
    }

    async fn merge_custom(
        &self,
        name: &str,
        inputs: &Value,
        context: &ExecutionContext,
    ) -> Result<Value> {
        builtin_workflow_merge_custom(name, inputs, context)
    }
}
