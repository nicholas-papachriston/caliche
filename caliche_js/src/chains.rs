//! JavaScript bindings for Caliche Chains
#![expect(
    clippy::trailing_empty_array,
    reason = "NAPI callback code generation emits the zero-sized ABI container"
)]

use caliche::chains::{
    Chain, ChainBuilder as RustChainBuilder, CompositionSequentialChain, LLMChain as RustLLMChain,
    MapReduceChain as RustMapReduceChain, ParallelChain as RustParallelChain,
};
use caliche::prompts::PromptTemplate;
use napi_derive::napi;
use serde_json::Value;
use std::sync::{Arc, RwLock};

async fn build_default_agent_service(
    provider: caliche::ProviderType,
    model: String,
) -> napi::Result<caliche::AgentService> {
    // Chains are prompt+LLM compositions; keep them tool-free so prompts like
    // "Process: {input}" cannot enter long tool loops during tests/examples.
    caliche::AgentService::from_config(
        caliche::AgentServiceConfig::new().with_provider(Some(provider), Some(model)),
    )
    .await
    .map_err(|e| napi::Error::from_reason(format!("Failed to create agent service: {e}")))
}

/// Helper function to handle `RwLock` read operations with proper error handling
fn rwlock_read<T>(lock: &RwLock<T>) -> napi::Result<std::sync::RwLockReadGuard<'_, T>> {
    lock.read()
        .map_err(|e| napi::Error::from_reason(format!("Failed to acquire read lock: {e}")))
}

/// Helper function to handle `RwLock` write operations with proper error handling
fn rwlock_write<T>(lock: &RwLock<T>) -> napi::Result<std::sync::RwLockWriteGuard<'_, T>> {
    lock.write()
        .map_err(|e| napi::Error::from_reason(format!("Failed to acquire write lock: {e}")))
}

/// JavaScript wrapper for `LLMChain`
#[napi]
pub struct LLMChain {
    inner: Arc<caliche::chains::LLMChain>,
}

#[napi]
impl LLMChain {
    /// Create a new `LLMChain` from provider/model
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(
        provider: String,
        model: String,
        prompt_template: String,
    ) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let prompt = PromptTemplate::new(&prompt_template);
        let chain = RustLLMChain::new(Arc::new(tokio::sync::Mutex::new(agent_service)), prompt);
        Ok(Self {
            inner: Arc::new(chain),
        })
    }

    /// Run an `LLMChain`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn run(&self, input: Value) -> napi::Result<ChainResult> {
        let result = self.inner.run(&input).await;
        match result {
            Ok(chain_result) => Ok(ChainResult {
                output: chain_result.output,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }

    /// Create an `LLMChain` from an Agent instance (for internal use)
    pub(crate) fn from_agent_internal(
        agent_service: Arc<tokio::sync::Mutex<caliche::AgentService>>,
        prompt_template: &str,
    ) -> Self {
        let prompt = PromptTemplate::new(prompt_template);
        let chain = RustLLMChain::new(agent_service, prompt);
        Self {
            inner: Arc::new(chain),
        }
    }
}

#[napi(object)]
pub struct ChainResult {
    pub output: Value,
}

/// JavaScript wrapper for `SequentialChain`
#[napi]
pub struct SequentialChain {
    chains: RwLock<Vec<(String, Arc<dyn Chain>)>>,
    #[expect(dead_code, reason = "stored but not read yet")]
    return_intermediate: RwLock<bool>,
}

impl SequentialChain {
    fn build_chain(&self) -> napi::Result<Arc<dyn Chain>> {
        let chains = {
            let guard = rwlock_read(&self.chains)?;
            guard.clone()
        };
        if chains.is_empty() {
            return Err(napi::Error::from_reason(
                "No chains added to sequential chain",
            ));
        }

        let mut chain: Arc<dyn Chain> = Arc::clone(&chains[0].1);

        for (_, chain_ref) in chains.iter().skip(1) {
            let sequential = CompositionSequentialChain::new(chain, Arc::clone(chain_ref));
            chain = Arc::new(sequential);
        }

        Ok(chain)
    }
}

#[napi]
impl SequentialChain {
    /// Create a new `SequentialChain`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            chains: RwLock::new(Vec::new()),
            return_intermediate: RwLock::new(false),
        }
    }

    /// Add a chain to `SequentialChain` from provider/model
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn add_chain(
        &self,
        name: String,
        provider: String,
        model: String,
        prompt_template: String,
    ) -> napi::Result<()> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let prompt = PromptTemplate::new(&prompt_template);
        let chain = RustLLMChain::new(Arc::new(tokio::sync::Mutex::new(agent_service)), prompt);
        {
            let mut chains = rwlock_write(&self.chains)?;
            chains.push((name, Arc::new(chain) as Arc<dyn Chain>));
        }
        Ok(())
    }

    /// Add a chain to `SequentialChain` from an `LLMChain` instance
    #[napi(js_name = "addChainInstance", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_chain_instance(&self, name: String, chain: &LLMChain) -> napi::Result<()> {
        {
            let mut chains = rwlock_write(&self.chains)?;
            chains.push((name, Arc::clone(&chain.inner) as Arc<dyn Chain>));
        }
        Ok(())
    }

    /// Run a `SequentialChain`
    #[napi]
    pub async fn run(&self, input: Value) -> napi::Result<ChainResult> {
        let chain_inner = self.build_chain()?;
        let result = chain_inner.run(&input).await;
        match result {
            Ok(chain_result) => Ok(ChainResult {
                output: chain_result.output,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `MapReduceChain`
#[napi]
pub struct MapReduceChain {
    inner: Arc<caliche::chains::MapReduceChain>,
}

#[napi]
impl MapReduceChain {
    /// Create a new `MapReduceChain` from provider/model
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(
        map_provider: String,
        map_model: String,
        map_prompt: String,
        reduce_provider: String,
        reduce_model: String,
        reduce_prompt: String,
    ) -> napi::Result<Self> {
        let map_provider_type = crate::parse_provider_type(&map_provider)?;
        let map_agent_service = build_default_agent_service(map_provider_type, map_model)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to create map agent service: {e}"))
            })?;

        let reduce_provider_type = crate::parse_provider_type(&reduce_provider)?;
        let reduce_agent_service = build_default_agent_service(reduce_provider_type, reduce_model)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to create reduce agent service: {e}"))
            })?;

        let map_prompt_template = PromptTemplate::new(&map_prompt);
        let reduce_prompt_template = PromptTemplate::new(&reduce_prompt);

        let map_chain = Arc::new(RustLLMChain::new(
            Arc::new(tokio::sync::Mutex::new(map_agent_service)),
            map_prompt_template,
        )) as Arc<dyn Chain>;
        let reduce_chain = Arc::new(RustLLMChain::new(
            Arc::new(tokio::sync::Mutex::new(reduce_agent_service)),
            reduce_prompt_template,
        )) as Arc<dyn Chain>;

        let chain = RustMapReduceChain::new(map_chain, reduce_chain);
        Ok(Self {
            inner: Arc::new(chain),
        })
    }

    /// Run a `MapReduceChain`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn run(&self, items: Vec<Value>) -> napi::Result<ChainResult> {
        let input = Value::Array(items);
        let result = self.inner.run(&input).await;
        match result {
            Ok(chain_result) => Ok(ChainResult {
                output: chain_result.output,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }

    /// Create a `MapReduceChain` from Agent services (for internal use)
    pub(crate) fn from_agents_internal(
        map_chain: Arc<dyn Chain>,
        reduce_chain: Arc<dyn Chain>,
    ) -> Self {
        let chain = RustMapReduceChain::new(map_chain, reduce_chain);
        Self {
            inner: Arc::new(chain),
        }
    }
}

/// JavaScript wrapper for `ParallelChain`
#[napi]
pub struct ParallelChain {
    inner: Arc<caliche::chains::ParallelChain>,
}

#[napi]
impl ParallelChain {
    /// Create a new `ParallelChain` from chain configurations
    /// Each chain config should be an object with: { provider: string, model: string, `prompt_template`: string }
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn from_chains(chains: Vec<serde_json::Value>) -> napi::Result<Self> {
        if chains.len() < 2 {
            return Err(napi::Error::from_reason(
                "ParallelChain requires at least 2 chains",
            ));
        }

        let mut rust_chains: Vec<Arc<dyn Chain>> = Vec::new();
        for chain_config in chains {
            let provider = chain_config
                .get("provider")
                .and_then(|v| v.as_str())
                .ok_or_else(|| napi::Error::from_reason("Chain config missing 'provider'"))?
                .to_string();
            let model = chain_config
                .get("model")
                .and_then(|v| v.as_str())
                .ok_or_else(|| napi::Error::from_reason("Chain config missing 'model'"))?
                .to_string();
            let prompt_template = chain_config
                .get("prompt_template")
                .and_then(|v| v.as_str())
                .ok_or_else(|| napi::Error::from_reason("Chain config missing 'prompt_template'"))?
                .to_string();

            let provider_type = crate::parse_provider_type(&provider)?;
            let agent_service = build_default_agent_service(provider_type, model).await?;

            let prompt = PromptTemplate::new(&prompt_template);
            let chain = RustLLMChain::new(Arc::new(tokio::sync::Mutex::new(agent_service)), prompt);
            rust_chains.push(Arc::new(chain) as Arc<dyn Chain>);
        }

        let parallel_chain = RustParallelChain::from_chains(rust_chains);
        Ok(Self {
            inner: Arc::new(parallel_chain),
        })
    }

    /// Run a `ParallelChain`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn run(&self, input: Value) -> napi::Result<ChainResult> {
        let result = self.inner.run(&input).await;
        match result {
            Ok(chain_result) => Ok(ChainResult {
                output: chain_result.output,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Parallel chain execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `ChainBuilder`
#[napi]
pub struct ChainBuilder {
    agent: RwLock<Option<Arc<tokio::sync::Mutex<caliche::AgentService>>>>,
    template: RwLock<Option<String>>,
    output_key: RwLock<Option<String>>,
}

#[napi]
impl ChainBuilder {
    /// Create a new `ChainBuilder`
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(provider: String, model: String) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        Ok(Self {
            agent: RwLock::new(Some(Arc::new(tokio::sync::Mutex::new(agent_service)))),
            template: RwLock::new(None),
            output_key: RwLock::new(None),
        })
    }

    /// Set template for `ChainBuilder`
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn with_template(&self, template: String) -> napi::Result<Self> {
        Ok(Self {
            agent: RwLock::new(rwlock_read(&self.agent)?.clone()),
            template: RwLock::new(Some(template)),
            output_key: RwLock::new(rwlock_read(&self.output_key)?.clone()),
        })
    }

    /// Set output key for `ChainBuilder`
    #[napi(catch_unwind)]
    pub fn with_output_key(&self, output_key: String) -> napi::Result<Self> {
        Ok(Self {
            agent: RwLock::new(rwlock_read(&self.agent)?.clone()),
            template: RwLock::new(rwlock_read(&self.template)?.clone()),
            output_key: RwLock::new(Some(output_key)),
        })
    }

    /// Build the chain from `ChainBuilder`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn build(&self) -> napi::Result<LLMChain> {
        // Clone values before await to avoid holding RwLockReadGuard across await
        let agent_service_arc = {
            let agent_arc = rwlock_read(&self.agent)?;
            agent_arc
                .as_ref()
                .ok_or_else(|| napi::Error::from_reason("Agent already consumed"))?
                .clone()
        };

        let template_str = {
            let template_guard = rwlock_read(&self.template)?;
            template_guard
                .as_ref()
                .ok_or_else(|| napi::Error::from_reason("Template is required"))?
                .clone()
        };

        let output_key = {
            let output_key_guard = rwlock_read(&self.output_key)?;
            output_key_guard.clone()
        };

        let agent_guard = agent_service_arc.lock().await;
        let provider = agent_guard.provider();
        let model = agent_guard.model().to_string();
        drop(agent_guard);

        let agent_service = build_default_agent_service(provider, model)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to recreate agent service: {e}"))
            })?;

        let prompt = PromptTemplate::new(template_str);
        let mut chain = caliche::chains::LLMChain::new(
            Arc::new(tokio::sync::Mutex::new(agent_service)),
            prompt,
        );

        if let Some(ref key) = output_key {
            chain = chain.with_output_key(key.clone());
        }

        Ok(LLMChain {
            inner: Arc::new(chain),
        })
    }

    /// Build and run the chain with a single string input
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn run_with(&self, input: String) -> napi::Result<ChainResult> {
        // Clone values before await to avoid holding RwLockReadGuard across await
        let agent_service_arc = {
            let agent_arc = rwlock_read(&self.agent)?;
            agent_arc
                .as_ref()
                .ok_or_else(|| napi::Error::from_reason("Agent already consumed"))?
                .clone()
        };

        let template_str = {
            let template_guard = rwlock_read(&self.template)?;
            template_guard
                .as_ref()
                .ok_or_else(|| napi::Error::from_reason("Template is required"))?
                .clone()
        };

        let output_key = {
            let output_key_guard = rwlock_read(&self.output_key)?;
            output_key_guard.clone()
        };

        let agent_guard = agent_service_arc.lock().await;
        let provider = agent_guard.provider();
        let model = agent_guard.model().to_string();
        drop(agent_guard);

        let agent_service = build_default_agent_service(provider, model)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to recreate agent service: {e}"))
            })?;

        let mut builder = RustChainBuilder::new(agent_service).with_template(template_str);

        if let Some(ref key) = output_key {
            builder = builder.with_output_key(key.clone());
        }

        let result = builder.run_with(input).await;
        match result {
            Ok(chain_result) => Ok(ChainResult {
                output: chain_result.output,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }

    /// Build and run the chain with JSON input
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn run(&self, input: Value) -> napi::Result<ChainResult> {
        // Clone values before await to avoid holding RwLockReadGuard across await
        let agent_service_arc = {
            let agent_arc = rwlock_read(&self.agent)?;
            agent_arc
                .as_ref()
                .ok_or_else(|| napi::Error::from_reason("Agent already consumed"))?
                .clone()
        };

        let template_str = {
            let template_guard = rwlock_read(&self.template)?;
            template_guard
                .as_ref()
                .ok_or_else(|| napi::Error::from_reason("Template is required"))?
                .clone()
        };

        let output_key = {
            let output_key_guard = rwlock_read(&self.output_key)?;
            output_key_guard.clone()
        };

        let agent_guard = agent_service_arc.lock().await;
        let provider = agent_guard.provider();
        let model = agent_guard.model().to_string();
        drop(agent_guard);

        let agent_service = build_default_agent_service(provider, model)
            .await
            .map_err(|e| {
                napi::Error::from_reason(format!("Failed to recreate agent service: {e}"))
            })?;

        let mut builder = RustChainBuilder::new(agent_service).with_template(template_str);

        if let Some(ref key) = output_key {
            builder = builder.with_output_key(key.clone());
        }

        let result = builder.run(&input).await;
        match result {
            Ok(chain_result) => Ok(ChainResult {
                output: chain_result.output,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Chain execution failed: {e}"
            ))),
        }
    }
}
