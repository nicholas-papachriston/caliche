//! Core Agent Harness implementation.
//!
//! The AgentHarness wraps AgentService with enhanced capabilities for
//! event streaming, context management, and state tracking.

use super::types::AgentHarness;
use super::types::{apply_permission_mode, register_harness_lifecycle_hooks};
use crate::agent::ToolExecutionDriver;
use crate::agent::service::AgentService;
use crate::harness::command_ledger::CommandLedger;
use crate::harness::config::{HarnessConfig, SessionConfig};
use crate::harness::context::ContextManager;
use crate::harness::diff_tracker::DiffTracker;
use crate::harness::events::EventStream;
use crate::harness::state::StateManager;
use crate::messages::TokenUsage;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

impl AgentHarness {
    /// Create a new AgentHarness with the given configuration.
    pub async fn new(config: HarnessConfig) -> Result<Self> {
        // Determine workspace root
        let workspace_root = config
            .workspace_root
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));

        // Create the underlying service
        let mut service = AgentService::from_config(
            crate::AgentServiceConfig::new()
                .with_provider(config.provider, config.model.clone())
                .with_tool_names_or_defaults(config.tools.clone()),
        )
        .await
        .context("Failed to create AgentService")?;

        // Apply system prompt if configured
        if let Some(ref prompt) = config.system_prompt {
            service.agent_mut().system_prompt = prompt.clone();
        }

        // Apply permission mode
        apply_permission_mode(&mut service, config.permission_mode);

        // Apply tool execution driver (config override, else environment fallback)
        let driver = config.tool_execution_driver.or_else(|| {
            std::env::var("CALICHE_TOOL_EXECUTION_DRIVER")
                .ok()
                .and_then(|v| ToolExecutionDriver::parse(&v))
        });
        if let Some(driver) = driver {
            service.agent_mut().set_tool_execution_driver(driver);
        }

        // Generate session ID (needed for durable stream key)
        let session_id = uuid::Uuid::now_v7().to_string();

        // Create event stream, optionally with durable backend
        let event_stream = {
            let mut stream = EventStream::new();
            if let Some(ref backend) = config.durable_stream_backend {
                let stream_key = format!("{}:{}", config.durable_stream_key_prefix, session_id);
                let writer =
                    crate::streams::DurableStreamWriter::new(Arc::clone(backend), stream_key);
                stream = stream.with_durable_writer(writer);
            }
            stream
        };

        // Create context manager
        let context_manager = Arc::new(RwLock::new(ContextManager::new(workspace_root.clone())));

        // Create state manager
        let state_manager = Arc::new(RwLock::new(StateManager::new()));

        // Create per-message diff tracker and register session diff hook
        let diff_tracker = Arc::new(RwLock::new(DiffTracker::from_state(
            service
                .session()
                .diff_state_snapshot()
                .context("hydrate session diff state")?,
        )));
        let notify_command = config.notify_command.clone();
        let command_ledger = Arc::new(RwLock::new(CommandLedger::default()));
        register_harness_lifecycle_hooks(
            &mut service,
            &event_stream,
            &state_manager,
            &diff_tracker,
            workspace_root,
            &config.formatter_rules,
            Arc::clone(&command_ledger),
        );
        ensure_turn_memory(&mut service);

        Ok(Self {
            service,
            config,
            event_stream,
            context_manager,
            state_manager,
            diff_tracker,
            session_id,
            turn_count: 0,
            total_usage: TokenUsage::default(),
            total_tool_calls: 0,
            total_compactions: 0,
            total_readonly_filter_hits: 0,
            history: Vec::new(),
            phase: crate::harness::suspend::SessionPhase::Idle,
            last_checkpoint_id: None,
            notify_command,
            command_ledger,
            side: std::sync::Arc::new(parking_lot::Mutex::new(
                super::types::SessionSideState::default(),
            )),
            compaction_reports: Vec::new(),
        })
    }

    /// Create a harness from an existing AgentService (e.g. from shared_service.clone_for_request()).
    /// Uses `config` for harness-level options (workspace_root, max_turns, event stream, etc.);
    /// provider/model/tools come from the service.
    pub fn from_service(mut service: AgentService, config: HarnessConfig) -> Result<Self> {
        let workspace_root = config
            .workspace_root
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));

        let session_id = uuid::Uuid::now_v7().to_string();

        let event_stream = {
            let mut stream = EventStream::new();
            if let Some(ref backend) = config.durable_stream_backend {
                let stream_key = format!("{}:{}", config.durable_stream_key_prefix, session_id);
                let writer =
                    crate::streams::DurableStreamWriter::new(Arc::clone(backend), stream_key);
                stream = stream.with_durable_writer(writer);
            }
            stream
        };

        let context_manager = Arc::new(RwLock::new(ContextManager::new(workspace_root.clone())));
        let state_manager = Arc::new(RwLock::new(StateManager::new()));
        let diff_tracker = Arc::new(RwLock::new(DiffTracker::from_state(
            service
                .session()
                .diff_state_snapshot()
                .context("hydrate session diff state")?,
        )));
        let notify_command = config.notify_command.clone();
        let command_ledger = Arc::new(RwLock::new(CommandLedger::default()));
        register_harness_lifecycle_hooks(
            &mut service,
            &event_stream,
            &state_manager,
            &diff_tracker,
            workspace_root,
            &config.formatter_rules,
            Arc::clone(&command_ledger),
        );

        if let Some(ref prompt) = config.system_prompt {
            service.agent_mut().system_prompt = prompt.clone();
        }

        apply_permission_mode(&mut service, config.permission_mode);
        ensure_turn_memory(&mut service);

        Ok(Self {
            service,
            config,
            event_stream,
            context_manager,
            state_manager,
            diff_tracker,
            session_id,
            turn_count: 0,
            total_usage: TokenUsage::default(),
            total_tool_calls: 0,
            total_compactions: 0,
            total_readonly_filter_hits: 0,
            history: Vec::new(),
            phase: crate::harness::suspend::SessionPhase::Idle,
            last_checkpoint_id: None,
            notify_command,
            command_ledger,
            side: std::sync::Arc::new(parking_lot::Mutex::new(
                super::types::SessionSideState::default(),
            )),
            compaction_reports: Vec::new(),
        })
    }

    /// Create a harness with a specific session configuration.
    pub async fn with_session(
        config: HarnessConfig,
        session_config: SessionConfig,
    ) -> Result<Self> {
        let mut harness = Self::new(config).await?;

        // Override session ID if provided
        if let Some(id) = session_config.session_id {
            harness.session_id = id;
        }

        // Set workspace if provided
        if let Some(workspace) = session_config.workspace {
            harness
                .context_manager
                .write()
                .await
                .set_workspace(workspace);
        }

        // Load context files
        for file in session_config.context_files {
            if let Err(e) = harness
                .context_manager
                .write()
                .await
                .track_file(&file)
                .await
            {
                tracing::warn!("Failed to load context file {:?}: {}", file, e);
            }
        }

        // Override max turns if provided
        if let Some(max_turns) = session_config.max_turns {
            harness.config.max_turns = max_turns;
        }

        // Set initial history
        harness.history = session_config.history;

        // Add system prompt addition
        if let Some(addition) = session_config.system_prompt_addition {
            let agent = harness.service.agent_mut();
            let current = std::mem::take(&mut agent.system_prompt);
            agent.system_prompt = format!("{current}\n\n{addition}");
        }

        Ok(harness)
    }
}

/// Give a session a buffer the next streamed turn can read.
///
/// `clone_for_request` and `AgentService::from_config` start with no memory.
/// Streamed turns then send only the current user message, and
/// `persist_stream_memory` has nowhere to write the reply.
fn ensure_turn_memory(service: &mut AgentService) {
    if service.memory().is_some() {
        return;
    }
    let counter = std::sync::Arc::new(crate::agent::memory::ApproximateTokenCounter::default());
    let budget =
        crate::agent::memory::session_memory_token_budget(service.agent().cached_context_window);
    let memory = crate::agent::memory::BufferMemory::new(
        crate::agent::memory::BufferMemoryConfig::default().with_max_tokens(budget),
        counter,
    );
    service.set_memory_strategy(Some(Box::new(memory)));
}
