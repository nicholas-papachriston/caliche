//! JavaScript bindings for Caliche `AgentCoordinator` and `MessageBus`

use caliche::orchestration::coordinator::{
    AgentCapability, AgentCoordinator as RustAgentCoordinator, AgentMetadata, LoadBalancingStrategy,
};
use caliche::orchestration::message_passing::MessageBus as RustMessageBus;
use napi_derive::napi;
use std::sync::Arc;

async fn build_default_agent_service(
    provider: caliche::ProviderType,
    model: String,
) -> napi::Result<caliche::AgentService> {
    caliche::AgentService::from_config(
        caliche::AgentServiceConfig::new()
            .with_provider(Some(provider), Some(model))
            .with_all_default_tools(),
    )
    .await
    .map_err(|e| napi::Error::from_reason(format!("Failed to create agent service: {e}")))
}

// Helper to convert JS string to AgentCapability
fn js_string_to_capability(cap_str: &str) -> AgentCapability {
    match cap_str {
        "research" => AgentCapability::Research,
        "writing" => AgentCapability::Writing,
        "coding" => AgentCapability::Coding,
        "data_analysis" => AgentCapability::DataAnalysis,
        "planning" => AgentCapability::Planning,
        "coordination" => AgentCapability::Coordination,
        s => AgentCapability::Custom(s.to_string()),
    }
}

fn usize_to_u32(value: usize, field: &str) -> napi::Result<u32> {
    u32::try_from(value)
        .map_err(|_| napi::Error::from_reason(format!("{field} exceeds the JavaScript API limit")))
}

/// JavaScript wrapper for `AgentCoordinator`
#[napi]
pub struct AgentCoordinator {
    inner: Arc<caliche::orchestration::coordinator::AgentCoordinator>,
}

#[napi]
#[expect(
    clippy::needless_pass_by_value,
    reason = "NAPI converts and transfers ownership of JavaScript string arguments"
)]
impl AgentCoordinator {
    /// Create a new `AgentCoordinator`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RustAgentCoordinator::new()),
        }
    }

    /// Create `AgentCoordinator` with load balancing strategy
    #[napi(factory, js_name = "withStrategy", catch_unwind)]
    pub fn with_strategy(strategy: String) -> Self {
        let strategy_enum = match strategy.as_str() {
            "round_robin" => LoadBalancingStrategy::RoundRobin,
            "weighted" => LoadBalancingStrategy::Weighted,
            "least_connections" | _ => LoadBalancingStrategy::LeastConnections,
        };
        Self {
            inner: Arc::new(RustAgentCoordinator::with_strategy(strategy_enum)),
        }
    }

    /// Register an agent with the coordinator
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn register_agent(
        &self,
        name: String,
        role: String,
        capabilities: Vec<String>,
        provider: String,
        model: String,
    ) -> napi::Result<String> {
        let capabilities_enum: Vec<AgentCapability> = capabilities
            .iter()
            .map(|c| js_string_to_capability(c))
            .collect();

        let metadata = AgentMetadata::new(name, role, capabilities_enum);

        // Create provider and agent service
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;
        let tool_registry = agent_service.tool_registry().clone();
        let llm_provider = agent_service.llm_provider();

        // Create Agent from components
        let rust_agent = caliche::agent::Agent::new(llm_provider, tool_registry);

        self.inner
            .register_agent(metadata, rust_agent)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to register agent: {e}")))
    }

    /// Unregister an agent
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn unregister_agent(&self, agent_id: String) -> napi::Result<()> {
        self.inner
            .unregister_agent(&agent_id)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to unregister agent: {e}")))
    }

    /// Delegate a task to the best available agent
    #[napi]
    pub async fn delegate_task(
        &self,
        message: String,
        capability: Option<String>,
    ) -> napi::Result<DelegateTaskResult> {
        let capability_enum = capability.map(|c| js_string_to_capability(&c));
        let result = self
            .inner
            .delegate_task(&message, capability_enum.as_ref())
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to delegate task: {e}")))?;

        Ok(DelegateTaskResult {
            agent_id: result.0,
            response: result.1,
            usage: crate::TokenUsageJs::from(result.2),
        })
    }

    /// Execute a task on a specific agent
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute_on_agent(
        &self,
        agent_id: String,
        message: String,
    ) -> napi::Result<ExecuteTaskResult> {
        let result = self
            .inner
            .execute_on_agent(&agent_id, &message)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to execute on agent: {e}")))?;

        Ok(ExecuteTaskResult {
            response: result.0,
            usage: crate::TokenUsageJs::from(result.1),
        })
    }

    /// Send message from one agent to another
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn send_message(
        &self,
        from_agent_id: String,
        to_agent_id: String,
        content: String,
    ) -> napi::Result<()> {
        self.inner
            .send_message(&from_agent_id, &to_agent_id, content)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to send message: {e}")))
    }

    /// Broadcast message to all agents
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn broadcast(&self, from_agent_id: String, content: String) -> napi::Result<()> {
        self.inner
            .broadcast(&from_agent_id, content)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to broadcast: {e}")))
    }

    /// Get coordinator statistics
    #[napi]
    pub async fn get_stats(&self) -> napi::Result<CoordinatorStats> {
        let stats = self.inner.get_stats().await;
        Ok(CoordinatorStats {
            total_agents: usize_to_u32(stats.total_agents, "total agent count")?,
            total_load: usize_to_u32(stats.total_load, "total load")?,
            total_capacity: usize_to_u32(stats.total_capacity, "total capacity")?,
            agent_stats: stats
                .agent_stats
                .into_iter()
                .map(|s| {
                    Ok(AgentStat {
                        agent_id: s.agent_id,
                        name: s.name,
                        current_load: usize_to_u32(s.current_load, "agent load")?,
                        max_concurrent_tasks: usize_to_u32(
                            s.max_concurrent_tasks,
                            "agent task capacity",
                        )?,
                    })
                })
                .collect::<napi::Result<Vec<_>>>()?,
        })
    }

    /// Get list of agent IDs
    #[napi(catch_unwind)]
    pub fn list_agent_ids(&self) -> Vec<String> {
        self.inner.list_agent_ids()
    }

    /// Get agent count
    #[napi(catch_unwind)]
    pub fn agent_count(&self) -> napi::Result<u32> {
        usize_to_u32(self.inner.agent_count(), "agent count")
    }

    /// Get message bus
    #[napi(catch_unwind)]
    pub fn get_message_bus(&self) -> MessageBus {
        MessageBus {
            inner: self.inner.message_bus(),
        }
    }
}

/// JavaScript wrapper for `MessageBus`
#[napi]
pub struct MessageBus {
    inner: Arc<RustMessageBus>,
}

#[napi]
impl MessageBus {
    /// Get message bus statistics
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn get_stats(&self) -> napi::Result<MessageBusStats> {
        let stats = self.inner.get_stats().await;
        Ok(MessageBusStats {
            queue_size: usize_to_u32(stats.queue_size, "queue size")?,
            max_queue_size: usize_to_u32(stats.max_queue_size, "maximum queue size")?,
            low_priority: usize_to_u32(stats.low_priority, "low-priority message count")?,
            normal_priority: usize_to_u32(stats.normal_priority, "normal-priority message count")?,
            high_priority: usize_to_u32(stats.high_priority, "high-priority message count")?,
            critical_priority: usize_to_u32(
                stats.critical_priority,
                "critical-priority message count",
            )?,
            subscriber_count: usize_to_u32(stats.subscriber_count, "subscriber count")?,
        })
    }

    /// Get queue size
    #[napi]
    pub async fn queue_size(&self) -> napi::Result<u32> {
        usize_to_u32(self.inner.queue_size().await, "queue size")
    }
}

#[napi(object)]
pub struct DelegateTaskResult {
    pub agent_id: String,
    pub response: String,
    pub usage: crate::TokenUsageJs,
}

#[napi(object)]
pub struct ExecuteTaskResult {
    pub response: String,
    pub usage: crate::TokenUsageJs,
}

#[napi(object)]
pub struct CoordinatorStats {
    pub total_agents: u32,
    pub total_load: u32,
    pub total_capacity: u32,
    pub agent_stats: Vec<AgentStat>,
}

#[napi(object)]
pub struct AgentStat {
    pub agent_id: String,
    pub name: String,
    pub current_load: u32,
    pub max_concurrent_tasks: u32,
}

#[napi(object)]
pub struct MessageBusStats {
    pub queue_size: u32,
    pub max_queue_size: u32,
    pub low_priority: u32,
    pub normal_priority: u32,
    pub high_priority: u32,
    pub critical_priority: u32,
    pub subscriber_count: u32,
}
