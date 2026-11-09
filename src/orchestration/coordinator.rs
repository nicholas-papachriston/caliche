use crate::agent::Agent;
use crate::agent::kernel::turn::Handoff;
use crate::config::{OrchestrationConfig, SupervisionConfig};
use crate::orchestration::acl::TaskInvocationPolicy;
use crate::orchestration::local::LocalAgentRuntime;
use crate::orchestration::message_passing::{AgentMessage, MessageBus};
use crate::orchestration::privilege::{
    AgentPrivileges, effective_tools, rejected_tools, tool_ceiling,
};
use crate::orchestration::quarantine::quarantine_once;
use crate::orchestration::supervision::{AgentFactory, Supervisor, SupervisorMetrics};
use crate::orchestration::{ExternalCliRuntime, HarnessRequest, HarnessRuntime, HarnessUsage};
use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Label used in ACL errors and quarantine headers when no caller is attributed.
const ANONYMOUS_CALLER: &str = "orchestrator";
/// Hard cap on per-agent instance pool size (registration-time DoS).
const MAX_AGENT_POOL_SIZE: usize = 64;

/// Capability enum for agent specialization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentCapability {
    Research,
    Writing,
    Coding,
    DataAnalysis,
    Planning,
    Coordination,
    Custom(String),
}

/// Metadata describing an agent's properties and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub agent_id: String,
    pub name: String,
    pub role: String,
    pub capabilities: Vec<AgentCapability>,
    pub max_concurrent_tasks: usize,
    pub current_load: usize,
    pub metadata: serde_json::Value,
}

impl AgentMetadata {
    pub fn new(
        name: impl Into<String>,
        role: impl Into<String>,
        capabilities: Vec<AgentCapability>,
    ) -> Self {
        Self {
            agent_id: Uuid::now_v7().to_string(),
            name: name.into(),
            role: role.into(),
            capabilities,
            max_concurrent_tasks: 5,
            current_load: 0,
            metadata: serde_json::json!({}),
        }
    }

    pub fn has_capability(&self, capability: &AgentCapability) -> bool {
        self.capabilities.contains(capability)
    }

    pub fn can_accept_task(&self) -> bool {
        self.current_load < self.max_concurrent_tasks
    }
}

/// Agent pool entry combining metadata with a dispatch runtime.
///
/// The runtime may be a local [`LocalAgentRuntime`] (a pool of `Agent` instances) or an
/// external harness such as Pi or Hermes. Coordinator safety controls (ACL, privilege
/// intersection, quarantine, timeout) apply to every runtime the same way.
struct AgentPoolEntry {
    metadata: Arc<RwLock<AgentMetadata>>,
    runtime: Arc<dyn HarnessRuntime>,
}

/// Options the orchestrator can set when delegating to a subagent: persona (system prompt),
/// allowed tools, and detailed instructions that are combined with the task message.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DelegationOptions {
    /// Override the subagent's system prompt (persona) for this task only. Restored after the call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    /// Restrict the subagent to these tool names for this task. None = use agent's current allowed set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Detailed instructions for the subagent, prepended to the task message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl DelegationOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Load balancing strategy for agent selection
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    Weighted,
}

/// Coordinates multiple agents, manages routing, and handles inter-agent communication.
///
/// When supervision is enabled via `with_supervision`, agents can be registered with
/// `register_supervised_agent` and will run in isolated tasks with automatic restart on failure.
pub struct AgentCoordinator {
    agents: DashMap<String, AgentPoolEntry>,
    message_bus: Arc<MessageBus>,
    load_balancing: LoadBalancingStrategy,
    round_robin_counter: Arc<AtomicUsize>,
    /// Optional OTP-style supervisor for isolated, supervised agent tasks.
    supervisor: Option<Arc<Supervisor>>,
    /// Metadata for supervised agents (used for select_agent; execute uses message bus).
    supervised_metadata: DashMap<String, Arc<RwLock<AgentMetadata>>>,
    /// Privilege ceilings and task ACLs keyed by `agent_id`.
    privileges: DashMap<String, AgentPrivileges>,
    /// Agent name to `agent_id`, so `transfer_to` can name a target by its readable name.
    name_index: DashMap<String, String>,
    /// ACL applied when the caller has no policy of its own.
    default_task_policy: TaskInvocationPolicy,
    /// Refuse targets that are not registered with this coordinator.
    require_registered_targets: bool,
    /// Wrap attributed dispatch results in an untrusted block before a parent reads them.
    quarantine_output: bool,
    /// Timeout applied to one dispatch, so a hung provider call cannot pin a pool slot.
    dispatch_timeout: Option<Duration>,
}

/// A dispatch that passed authorization: the resolved target plus its narrowed options.
#[derive(Debug, Clone)]
pub struct AuthorizedDispatch {
    /// Resolved `agent_id` of the target.
    pub agent_id: String,
    /// Delegation options after privilege intersection.
    pub options: DelegationOptions,
    /// Tools the request asked for that the ceiling refused.
    pub rejected_tools: Vec<String>,
}

impl AgentCoordinator {
    /// Discover subagent descriptor files from supported ecosystems without registering them.
    pub fn discover_subagent_descriptors() -> Vec<crate::discovery::auto::DiscoveredSubagent> {
        let snapshot =
            crate::discovery::auto::discover_all(&crate::discovery::auto::DiscoveryOptions {
                rules: false,
                skills: false,
                commands: false,
                subagents: true,
                mcp: false,
            });
        snapshot.subagents
    }

    pub fn new() -> Self {
        Self::with_strategy(LoadBalancingStrategy::LeastConnections)
    }

    pub fn with_strategy(strategy: LoadBalancingStrategy) -> Self {
        Self {
            agents: DashMap::new(),
            message_bus: Arc::new(MessageBus::new()),
            load_balancing: strategy,
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
            supervisor: None,
            supervised_metadata: DashMap::new(),
            privileges: DashMap::new(),
            name_index: DashMap::new(),
            default_task_policy: TaskInvocationPolicy::allow_all(),
            require_registered_targets: true,
            quarantine_output: true,
            dispatch_timeout: None,
        }
    }

    /// Apply the orchestration section of the agent config: task ACL, target policy,
    /// quarantine, and dispatch timeout.
    pub fn with_orchestration_config(mut self, config: &OrchestrationConfig) -> Result<Self> {
        if let Some(acl) = config.task_acl.as_ref() {
            self.default_task_policy = TaskInvocationPolicy::from_config_map(acl)
                .context("Parse orchestration.task_acl")?;
        }
        self.require_registered_targets = config.require_registered_targets_or_default();
        self.quarantine_output = config.quarantine_subagent_output_or_default();
        self.dispatch_timeout = config.dispatch_timeout();
        Ok(self)
    }

    /// Set the ACL used for callers that have no policy of their own.
    #[must_use]
    pub fn with_default_task_policy(mut self, policy: TaskInvocationPolicy) -> Self {
        self.default_task_policy = policy;
        self
    }

    /// Choose whether unknown invocation targets are refused. Defaults to `true`.
    #[must_use]
    pub fn with_require_registered_targets(mut self, required: bool) -> Self {
        self.require_registered_targets = required;
        self
    }

    /// Choose whether attributed dispatch results are quarantined. Defaults to `true`.
    #[must_use]
    pub fn with_quarantine_output(mut self, quarantine: bool) -> Self {
        self.quarantine_output = quarantine;
        self
    }

    /// Bound how long one dispatch may run.
    #[must_use]
    pub fn with_dispatch_timeout(mut self, timeout: Duration) -> Self {
        self.dispatch_timeout = Some(timeout);
        self
    }

    /// Enable OTP-style supervision. Agents registered via `register_supervised_agent` will
    /// run in isolated tasks with automatic restart on panic or error.
    pub fn with_supervision(mut self, config: SupervisionConfig) -> Self {
        let supervisor = Arc::new(Supervisor::new(config));
        let sup = Arc::clone(&supervisor);
        tokio::spawn(async move {
            sup.monitor_loop().await;
        });
        self.supervisor = Some(supervisor);
        self
    }

    /// Restart counters for supervised agents, or `None` when supervision is disabled.
    pub async fn supervisor_metrics(&self) -> Option<SupervisorMetrics> {
        let supervisor = self.supervisor.as_ref()?;
        Some(supervisor.metrics().await)
    }

    /// Register a supervised agent (requires supervision via `with_supervision`).
    /// The agent runs in an isolated task; use the message bus for communication.
    pub async fn register_supervised_agent(
        &self,
        metadata: AgentMetadata,
        factory: Arc<dyn AgentFactory>,
    ) -> Result<String> {
        let supervisor = self
            .supervisor
            .as_ref()
            .context("Supervision not enabled; call with_supervision first")?;

        let agent_id = metadata.agent_id.clone();

        tracing::info!(
            agent_id = %agent_id,
            name = %metadata.name,
            role = %metadata.role,
            "Registering supervised agent"
        );

        self.name_index
            .insert(metadata.name.clone(), agent_id.clone());
        self.supervised_metadata
            .insert(agent_id.clone(), Arc::new(RwLock::new(metadata)));
        supervisor.register(agent_id.clone(), factory).await?;

        Ok(agent_id)
    }

    /// Register an agent with the coordinator.
    ///
    /// Spins up `metadata.max_concurrent_tasks` independent instances (via
    /// `Agent::clone_for_request`, sharing the expensive provider/tools/config but each with
    /// its own fresh session) so up to that many tasks can genuinely run concurrently against
    /// this `agent_id` instead of serializing on a single instance's lock.
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async across caliche_js/py bindings; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn register_agent(&self, metadata: AgentMetadata, agent: Agent) -> Result<String> {
        self.register_agent_with_privileges(metadata, agent, AgentPrivileges::unrestricted())
    }

    /// Register an agent together with its privilege ceiling and task ACL.
    ///
    /// Use this for subagents that must not hold every tool the parent holds, or that may
    /// only invoke a named subset of the other registered agents.
    pub fn register_agent_with_privileges(
        &self,
        metadata: AgentMetadata,
        agent: Agent,
        privileges: AgentPrivileges,
    ) -> Result<String> {
        let agent_id = self.register_agent_inner(metadata, agent)?;
        self.privileges.insert(agent_id.clone(), privileges);
        Ok(agent_id)
    }

    /// Register any [`HarnessRuntime`] (local agent, Pi, Hermes, or a custom adapter).
    ///
    /// ACL, privilege intersection, quarantine, and dispatch timeout apply the same way
    /// they do for `register_agent`.
    pub fn register_harness(
        &self,
        metadata: AgentMetadata,
        runtime: Arc<dyn HarnessRuntime>,
    ) -> Result<String> {
        self.register_harness_with_privileges(metadata, runtime, AgentPrivileges::unrestricted())
    }

    /// Register a harness together with its privilege ceiling and task ACL.
    pub fn register_harness_with_privileges(
        &self,
        metadata: AgentMetadata,
        runtime: Arc<dyn HarnessRuntime>,
        privileges: AgentPrivileges,
    ) -> Result<String> {
        let agent_id = self.register_runtime_inner(metadata, runtime)?;
        self.privileges.insert(agent_id.clone(), privileges);
        Ok(agent_id)
    }

    /// Register the Pi coding-agent CLI (`pi -p <prompt>`) as a dispatch target.
    ///
    /// Production requires `CALICHE_HARNESS_COMMAND_ALLOWLIST` to include `pi`.
    pub fn register_pi(
        &self,
        metadata: AgentMetadata,
        privileges: AgentPrivileges,
    ) -> Result<String> {
        let runtime = Arc::new(ExternalCliRuntime::pi()?);
        self.register_harness_with_privileges(metadata, runtime, privileges)
    }

    /// Register the Hermes CLI (`hermes chat -Q -q <prompt>`) as a dispatch target.
    ///
    /// Production requires `CALICHE_HARNESS_COMMAND_ALLOWLIST` to include `hermes`.
    pub fn register_hermes(
        &self,
        metadata: AgentMetadata,
        privileges: AgentPrivileges,
    ) -> Result<String> {
        let runtime = Arc::new(ExternalCliRuntime::hermes()?);
        self.register_harness_with_privileges(metadata, runtime, privileges)
    }

    fn register_agent_inner(&self, metadata: AgentMetadata, agent: Agent) -> Result<String> {
        let pool_size = metadata.max_concurrent_tasks.clamp(1, MAX_AGENT_POOL_SIZE);
        if metadata.max_concurrent_tasks > MAX_AGENT_POOL_SIZE {
            tracing::warn!(
                agent_id = %metadata.agent_id,
                requested = metadata.max_concurrent_tasks,
                capped = MAX_AGENT_POOL_SIZE,
                "Clamping agent pool size"
            );
        }
        let runtime = Arc::new(LocalAgentRuntime::new(agent, pool_size));
        self.register_runtime_inner(metadata, runtime)
    }

    fn register_runtime_inner(
        &self,
        metadata: AgentMetadata,
        runtime: Arc<dyn HarnessRuntime>,
    ) -> Result<String> {
        let agent_id = metadata.agent_id.clone();

        tracing::info!(
            agent_id = %agent_id,
            name = %metadata.name,
            role = %metadata.role,
            kind = runtime.kind(),
            max_concurrent_tasks = metadata.max_concurrent_tasks,
            "Registering harness with coordinator"
        );

        if self
            .name_index
            .get(&metadata.name)
            .is_some_and(|existing| existing.value() != &agent_id)
        {
            anyhow::bail!("agent name '{}' is already registered", metadata.name);
        }

        self.name_index
            .insert(metadata.name.clone(), agent_id.clone());

        let entry = AgentPoolEntry {
            metadata: Arc::new(RwLock::new(metadata)),
            runtime,
        };

        self.agents.insert(agent_id.clone(), entry);

        Ok(agent_id)
    }

    /// Register a discovered subagent descriptor as a live coordinator agent.
    ///
    /// The descriptor's declared tools become the agent's privilege ceiling and its
    /// `task_permissions` frontmatter becomes its task ACL, so a discovered file cannot grant
    /// itself more than it declares.
    pub fn register_discovered_subagent(
        &self,
        descriptor: &crate::discovery::auto::DiscoveredSubagent,
        agent: Agent,
    ) -> Result<String> {
        let metadata = descriptor.to_agent_metadata();
        let privileges = descriptor.to_privileges()?;
        tracing::info!(
            name = %descriptor.name,
            source = %descriptor.source_path.display(),
            tools = descriptor.tools.len(),
            "Registering discovered subagent with coordinator"
        );
        self.register_agent_with_privileges(metadata, agent, privileges)
    }

    /// Unregister an agent from the coordinator (unsupervised or supervised).
    pub async fn unregister_agent(&self, agent_id: &str) -> Result<()> {
        let removed_name = if let Some(entry) = self.agents.remove(agent_id) {
            let name = entry.1.metadata.read().await.name.clone();
            tracing::info!(agent_id = %agent_id, "Unregistered agent from coordinator");
            Some(name)
        } else if let Some(entry) = self.supervised_metadata.remove(agent_id) {
            let name = entry.1.read().await.name.clone();
            tracing::info!(agent_id = %agent_id, "Unregistered supervised agent");
            Some(name)
        } else {
            None
        };

        let Some(name) = removed_name else {
            anyhow::bail!("Agent not found: {}", agent_id);
        };

        self.privileges.remove(agent_id);
        // Only drop the name mapping when it still points at this agent; a later
        // registration reusing the same name owns the entry.
        if self
            .name_index
            .get(&name)
            .is_some_and(|mapped| mapped.value() == agent_id)
        {
            self.name_index.remove(&name);
        }
        Ok(())
    }

    /// Replace the privileges of an already registered agent.
    pub fn set_agent_privileges(&self, agent_id: &str, privileges: AgentPrivileges) {
        self.privileges.insert(agent_id.to_string(), privileges);
    }

    /// Resolve an invocation target that names either an `agent_id` or an agent name.
    pub fn resolve_agent_id(&self, target: &str) -> Option<String> {
        if self.agents.contains_key(target) || self.supervised_metadata.contains_key(target) {
            return Some(target.to_string());
        }
        self.name_index
            .get(target)
            .map(|entry| entry.value().clone())
    }

    fn name_for_id(&self, agent_id: &str) -> Option<String> {
        self.name_index.iter().find_map(|entry| {
            if entry.value() == agent_id {
                Some(entry.key().clone())
            } else {
                None
            }
        })
    }

    /// Tool ceiling recorded for an agent, if any.
    fn max_tools_for(&self, agent_id: &str) -> Option<Vec<String>> {
        self.privileges
            .get(agent_id)
            .and_then(|entry| entry.value().max_tools.clone())
    }

    /// ACL that governs what `caller` may invoke.
    fn policy_for(&self, caller: Option<&str>) -> TaskInvocationPolicy {
        let Some(caller) = caller else {
            return self.default_task_policy.clone();
        };
        let caller_id = self
            .resolve_agent_id(caller)
            .unwrap_or_else(|| caller.to_string());
        match self.privileges.get(&caller_id) {
            Some(entry) if !entry.value().task_policy.is_empty() => {
                entry.value().task_policy.clone()
            }
            _ => self.default_task_policy.clone(),
        }
    }

    /// Authorize an invocation and resolve the target `agent_id`.
    ///
    /// Refuses when the caller's ACL denies the target, and (unless
    /// [`Self::with_require_registered_targets`] was turned off) when the target is not
    /// registered with this coordinator. ACL evaluation uses every alias of the target
    /// (requested string, resolved id, registered name) so a name-based deny cannot be
    /// skipped by invoking the UUID.
    pub fn authorize_invocation(&self, caller: Option<&str>, target: &str) -> Result<String> {
        let caller_label = caller.unwrap_or(ANONYMOUS_CALLER);
        let policy = self.policy_for(caller);

        match self.resolve_agent_id(target) {
            Some(agent_id) => {
                let name = self.name_for_id(&agent_id);
                let mut aliases = vec![target, agent_id.as_str()];
                if let Some(name) = name.as_deref() {
                    aliases.push(name);
                }
                policy.authorize_aliases(caller_label, target, &aliases)?;
                Ok(agent_id)
            }
            None if self.require_registered_targets => {
                policy.authorize(caller_label, target)?;
                anyhow::bail!(
                    "Agent '{}' may not invoke '{}': target is not registered with the coordinator",
                    caller_label,
                    target
                )
            }
            None => {
                policy.authorize(caller_label, target)?;
                Ok(target.to_string())
            }
        }
    }

    /// Authorize a `transfer_to` handoff and narrow its options to the allowed privileges.
    ///
    /// The requested tool list is intersected with the caller's ceiling and the target's
    /// ceiling, so a handoff can only ever narrow privilege.
    pub fn authorize_handoff(
        &self,
        caller: Option<&str>,
        handoff: &Handoff,
    ) -> Result<AuthorizedDispatch> {
        let agent_id = self.authorize_invocation(caller, &handoff.target_agent)?;
        let mut options = handoff.to_delegation_options();
        self.narrow_options(caller, &agent_id, &mut options)
            .map(|rejected| AuthorizedDispatch {
                agent_id,
                options,
                rejected_tools: rejected,
            })
    }

    /// Intersect `options` with the caller and target ceilings; returns refused tools.
    fn narrow_options(
        &self,
        caller: Option<&str>,
        target_id: &str,
        options: &mut DelegationOptions,
    ) -> Result<Vec<String>> {
        let caller_max = caller
            .and_then(|caller| self.resolve_agent_id(caller))
            .and_then(|caller_id| self.max_tools_for(&caller_id));
        let target_max = self.max_tools_for(target_id);
        let ceiling = tool_ceiling(caller_max.as_deref(), target_max.as_deref());

        let refused = options
            .tools
            .as_deref()
            .map(|requested| rejected_tools(ceiling.as_deref(), requested))
            .unwrap_or_default();

        options.tools = effective_tools(ceiling.as_deref(), options.tools.as_deref());

        if options.persona.is_none()
            && let Some(entry) = self.privileges.get(target_id)
            && let Some(persona) = entry.value().persona.clone()
        {
            options.persona = Some(persona);
        }

        if !refused.is_empty() {
            tracing::warn!(
                caller = %caller.unwrap_or(ANONYMOUS_CALLER),
                target = %target_id,
                refused = ?refused,
                "Delegation requested tools outside the privilege ceiling; they were dropped"
            );
        }
        Ok(refused)
    }

    /// Get agent metadata by ID (unsupervised or supervised).
    pub async fn get_agent_metadata(&self, agent_id: &str) -> Result<AgentMetadata> {
        if let Some(entry) = self.agents.get(agent_id) {
            return Ok(entry.metadata.read().await.clone());
        }
        if let Some(entry) = self.supervised_metadata.get(agent_id) {
            return Ok(entry.read().await.clone());
        }
        anyhow::bail!("Agent not found: {}", agent_id);
    }

    /// Find agents with specific capability (unsupervised and supervised).
    pub async fn find_agents_by_capability(
        &self,
        capability: &AgentCapability,
    ) -> Vec<AgentMetadata> {
        let estimated_matches = (self.agents.len() + self.supervised_metadata.len()).max(4);
        let mut matching_agents = Vec::with_capacity(estimated_matches);

        for entry in self.agents.iter() {
            let metadata = entry.value().metadata.read().await;
            if metadata.has_capability(capability) && metadata.can_accept_task() {
                matching_agents.push(metadata.clone());
            }
        }
        for entry in self.supervised_metadata.iter() {
            let metadata = entry.value().read().await;
            if metadata.has_capability(capability) && metadata.can_accept_task() {
                matching_agents.push(metadata.clone());
            }
        }

        matching_agents
    }

    /// Select the best agent for a task based on capability and load balancing
    pub async fn select_agent(
        &self,
        required_capability: Option<&AgentCapability>,
    ) -> Result<String> {
        let candidates = self.candidate_agents(required_capability).await;
        if candidates.is_empty() {
            anyhow::bail!("No suitable agents available");
        }
        self.pick_by_load(&candidates).map(str::to_string)
    }

    /// Agents that can accept a task and, when given, hold the required capability.
    async fn candidate_agents(
        &self,
        required_capability: Option<&AgentCapability>,
    ) -> Vec<AgentMetadata> {
        if let Some(cap) = required_capability {
            return self.find_agents_by_capability(cap).await;
        }
        let mut all_agents = Vec::new();
        for entry in self.agents.iter() {
            let metadata = entry.value().metadata.read().await;
            if metadata.can_accept_task() {
                all_agents.push(metadata.clone());
            }
        }
        for entry in self.supervised_metadata.iter() {
            let metadata = entry.value().read().await;
            if metadata.can_accept_task() {
                all_agents.push(metadata.clone());
            }
        }
        all_agents
    }

    /// Apply the load balancing strategy to a non-empty candidate list.
    fn pick_by_load<'a>(&self, candidates: &'a [AgentMetadata]) -> Result<&'a str> {
        let selected = match self.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                // Lock-free atomic increment
                let idx =
                    self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
                candidates
                    .get(idx)
                    .ok_or_else(|| anyhow::anyhow!("No candidate found for round robin"))?
            }
            LoadBalancingStrategy::LeastConnections => candidates
                .iter()
                .min_by_key(|m| m.current_load)
                .ok_or_else(|| anyhow::anyhow!("No candidate found for least connections"))?,
            LoadBalancingStrategy::Weighted => {
                // Weighted by available capacity
                candidates
                    .iter()
                    .max_by_key(|m| m.max_concurrent_tasks - m.current_load)
                    .ok_or_else(|| anyhow::anyhow!("No candidate found for weighted selection"))?
            }
        };

        Ok(selected.agent_id.as_str())
    }

    /// Execute a task on a specific agent
    pub async fn execute_on_agent(
        &self,
        agent_id: &str,
        message: &str,
    ) -> Result<(String, crate::messages::TokenUsage)> {
        self.execute_on_agent_with_options(agent_id, message, None)
            .await
    }

    /// Execute a task on a specific agent with optional persona, tools, and instructions.
    /// When `options` is set, the subagent's system prompt and/or allowed tools are overridden
    /// for this call only and restored afterward. Instructions are prepended to the task message.
    pub async fn execute_on_agent_with_options(
        &self,
        agent_id: &str,
        message: &str,
        options: Option<DelegationOptions>,
    ) -> Result<(String, crate::messages::TokenUsage)> {
        self.execute_on_agent_as(None, agent_id, message, options)
            .await
    }

    /// Execute a task on a specific agent on behalf of `caller`.
    ///
    /// This is the authorization-aware entry point: the caller's task ACL decides whether the
    /// target may be invoked, the requested tools are intersected with the caller and target
    /// privilege ceilings, the dispatch runs under the configured timeout, and the result is
    /// quarantined before it returns to an attributed parent.
    pub async fn execute_on_agent_as(
        &self,
        caller: Option<&str>,
        target: &str,
        message: &str,
        options: Option<DelegationOptions>,
    ) -> Result<(String, crate::messages::TokenUsage)> {
        let agent_id = self.authorize_invocation(caller, target)?;
        let mut options = options.unwrap_or_default();
        let _ = self.narrow_options(caller, &agent_id, &mut options)?;

        let (response, usage) = self
            .run_on_agent(&agent_id, message, Some(options))
            .await
            .with_context(|| format!("Dispatch to agent '{agent_id}' failed"))?;

        // Only an attributed dispatch feeds a parent context, so that is the only place a
        // quarantine wrapper is useful; an unattributed call returns the raw text.
        let response = match (self.quarantine_output, caller) {
            (true, Some(_)) => {
                let label = self.display_name(&agent_id).await;
                quarantine_once(&label, &response)
            }
            _ => response,
        };
        Ok((response, usage))
    }

    /// Readable label for an agent, falling back to its id.
    async fn display_name(&self, agent_id: &str) -> String {
        if let Some(entry) = self.agents.get(agent_id) {
            let metadata = Arc::clone(&entry.metadata);
            drop(entry);
            return metadata.read().await.name.clone();
        }
        agent_id.to_string()
    }

    async fn run_on_agent(
        &self,
        agent_id: &str,
        message: &str,
        options: Option<DelegationOptions>,
    ) -> Result<(String, crate::messages::TokenUsage)> {
        if self.supervised_metadata.contains_key(agent_id) {
            anyhow::bail!(
                "Supervised agents use the message bus for communication; \
                 execute_on_agent is not supported for agent_id: {}",
                agent_id
            );
        }
        // Pull out just the Arcs we need and drop the `DashMap` shard guard immediately.
        // `entry` holds a read lock on the whole shard for as long as it's alive; keeping it
        // across an LLM or subprocess turn would block concurrent register/unregister on the
        // same shard.
        let (metadata_arc, runtime) = {
            let entry = self.agents.get(agent_id).context("Agent not found")?;
            (Arc::clone(&entry.metadata), Arc::clone(&entry.runtime))
        };

        {
            let mut metadata = metadata_arc.write().await;
            metadata.current_load += 1;
        }

        let options = options.unwrap_or_default();
        let request = HarnessRequest {
            message: message.to_string(),
            persona: options.persona,
            tools: options.tools,
            instructions: options.instructions,
            timeout: self.dispatch_timeout,
        };

        // Backstop timeout: a runtime that ignores `request.timeout` still releases the
        // pool slot. Process adapters must also kill the child (`kill_on_drop`).
        let execute = runtime.execute(request);
        let result = match self.dispatch_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, execute).await {
                Ok(outcome) => outcome,
                Err(_) => Err(anyhow::anyhow!(
                    "Dispatch to agent '{}' timed out after {}ms",
                    agent_id,
                    timeout.as_millis()
                )),
            },
            None => execute.await,
        }
        .map(|response| (response.text, engine_usage(response.usage)));

        {
            let mut metadata = metadata_arc.write().await;
            metadata.current_load = metadata.current_load.saturating_sub(1);
        }

        result
    }

    /// Delegate a task to the best available agent
    pub async fn delegate_task(
        &self,
        message: &str,
        required_capability: Option<&AgentCapability>,
    ) -> Result<(String, String, crate::messages::TokenUsage)> {
        self.delegate_task_with_options(message, required_capability, None)
            .await
    }

    /// Delegate a task to the best available agent with persona, tools, and instructions.
    pub async fn delegate_task_with_options(
        &self,
        message: &str,
        required_capability: Option<&AgentCapability>,
        options: Option<DelegationOptions>,
    ) -> Result<(String, String, crate::messages::TokenUsage)> {
        self.delegate_task_as(None, message, required_capability, options)
            .await
    }

    /// Delegate a task on behalf of `caller`.
    ///
    /// Capability selection runs first, then the caller's task ACL decides whether the chosen
    /// agent may be invoked. A denied selection is an error rather than a silent fallback, so
    /// a caller cannot reach a forbidden agent through capability routing.
    pub async fn delegate_task_as(
        &self,
        caller: Option<&str>,
        message: &str,
        required_capability: Option<&AgentCapability>,
        options: Option<DelegationOptions>,
    ) -> Result<(String, String, crate::messages::TokenUsage)> {
        let agent_id = self
            .select_agent_for(caller, required_capability)
            .await
            .context("No agent both matches the capability and passes the caller's task ACL")?;
        let (response, usage) = self
            .execute_on_agent_as(caller, &agent_id, message, options)
            .await?;

        tracing::info!(
            agent_id = %agent_id,
            caller = %caller.unwrap_or(ANONYMOUS_CALLER),
            "Task delegated and completed"
        );

        Ok((agent_id, response, usage))
    }

    /// Select an agent that both matches the capability and passes the caller's task ACL.
    pub async fn select_agent_for(
        &self,
        caller: Option<&str>,
        required_capability: Option<&AgentCapability>,
    ) -> Result<String> {
        let candidates = self.candidate_agents(required_capability).await;
        if candidates.is_empty() {
            anyhow::bail!("No suitable agents available");
        }

        let policy = self.policy_for(caller);
        let permitted: Vec<AgentMetadata> = candidates
            .into_iter()
            .filter(|metadata| {
                policy
                    .evaluate_aliases(&[metadata.name.as_str(), metadata.agent_id.as_str()])
                    .is_allowed()
            })
            .collect();

        if permitted.is_empty() {
            anyhow::bail!(
                "Agent '{}' may not invoke any available agent: all candidates are denied by its task ACL",
                caller.unwrap_or(ANONYMOUS_CALLER)
            );
        }

        self.pick_by_load(&permitted).map(str::to_string)
    }

    /// Get message bus for inter-agent communication
    pub fn message_bus(&self) -> Arc<MessageBus> {
        Arc::clone(&self.message_bus)
    }

    /// Send message from one agent to another
    pub async fn send_message(
        &self,
        from_agent_id: &str,
        to_agent_id: &str,
        content: String,
    ) -> Result<()> {
        let message = AgentMessage::new(
            from_agent_id.to_string(),
            Some(to_agent_id.to_string()),
            content,
        );

        self.message_bus.publish(message).await;
        Ok(())
    }

    /// Broadcast message to all agents
    pub async fn broadcast(&self, from_agent_id: &str, content: impl Into<String>) -> Result<()> {
        let message = AgentMessage::new(from_agent_id, None, content);

        self.message_bus.publish(message).await;
        Ok(())
    }

    /// Get number of registered agents (unsupervised and supervised).
    pub fn agent_count(&self) -> usize {
        self.agents.len() + self.supervised_metadata.len()
    }

    /// Get list of all agent IDs (unsupervised and supervised).
    pub fn list_agent_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.agents.iter().map(|e| e.key().clone()).collect();
        for entry in self.supervised_metadata.iter() {
            ids.push(entry.key().clone());
        }
        ids
    }

    /// Get coordinator statistics
    pub async fn get_stats(&self) -> CoordinatorStats {
        let mut total_load = 0;
        let mut total_capacity = 0;
        let mut agent_stats = Vec::new();

        for entry in self.agents.iter() {
            let metadata = entry.value().metadata.read().await;
            total_load += metadata.current_load;
            total_capacity += metadata.max_concurrent_tasks;
            agent_stats.push(AgentStat {
                agent_id: metadata.agent_id.clone(),
                name: metadata.name.clone(),
                current_load: metadata.current_load,
                max_concurrent_tasks: metadata.max_concurrent_tasks,
            });
        }
        for entry in self.supervised_metadata.iter() {
            let metadata = entry.value().read().await;
            total_load += metadata.current_load;
            total_capacity += metadata.max_concurrent_tasks;
            agent_stats.push(AgentStat {
                agent_id: metadata.agent_id.clone(),
                name: metadata.name.clone(),
                current_load: metadata.current_load,
                max_concurrent_tasks: metadata.max_concurrent_tasks,
            });
        }

        CoordinatorStats {
            total_agents: self.agents.len() + self.supervised_metadata.len(),
            total_load,
            total_capacity,
            agent_stats,
        }
    }
}

fn engine_usage(usage: HarnessUsage) -> crate::messages::TokenUsage {
    crate::messages::TokenUsage::from_counts(
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )
}

impl Default for AgentCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CoordinatorStats {
    pub total_agents: usize,
    pub total_load: usize,
    pub total_capacity: usize,
    pub agent_stats: Vec<AgentStat>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentStat {
    pub agent_id: String,
    pub name: String,
    pub current_load: usize,
    pub max_concurrent_tasks: usize,
}
