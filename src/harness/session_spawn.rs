//! Default [`SessionSpawner`] backed by [`AgentHarness`].

use crate::agent::subagent::{SharedSubagentSelection, SubagentModelSelection};
use crate::agent::tools::session_tools::SessionSpawner;
use crate::data::persistence::task_ledger::{self, TaskKind, TaskState};
use crate::harness::complex_task::{
    ComplexTaskConfig, apply_complex_task_prompt, run_complex_task,
};
use crate::harness::config::HarnessConfig;
use crate::harness::core::AgentHarness;
use crate::llm::ProviderType;
use crate::messages::TokenUsage;
use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// Spawns a child harness session with optional subagent model overrides.
#[derive(Debug, Clone)]
pub struct HarnessSessionSpawner {
    /// Parent provider used when the selection and spawn args omit one.
    pub provider: ProviderType,
    /// Parent model used when the selection and spawn args omit one.
    pub model: String,
    /// Persistent `/subagent-provider` and `/subagent-model` choice.
    pub selection: SharedSubagentSelection,
    /// Default workspace for the child.
    pub workspace: PathBuf,
    /// Default child tools. Spawn args override this list.
    pub tools: Option<Vec<String>>,
    /// When true, the child runs [`run_complex_task`].
    pub complex: bool,
    /// Harness turn cap for non-complex children.
    pub max_turns: usize,
}

impl HarnessSessionSpawner {
    /// Build a spawner from the parent runtime identity.
    #[must_use]
    pub fn new(provider: ProviderType, model: impl Into<String>, workspace: PathBuf) -> Self {
        Self {
            provider,
            model: model.into(),
            selection: SharedSubagentSelection::default(),
            workspace,
            tools: None,
            complex: false,
            max_turns: 50,
        }
    }

    /// Replace the live subagent model selection.
    pub fn set_selection(&self, selection: SubagentModelSelection) {
        self.selection.set(selection);
    }

    fn resolve_identity(&self, model_arg: Option<&str>) -> (ProviderType, String) {
        let current = self.selection.get();
        if let Some(model) = model_arg {
            let (spec_provider, spec_model) = crate::agent::subagent::parse_model_spec(model);
            if let Some(provider) = spec_provider {
                return (provider, spec_model);
            }
            let mut selection = current;
            selection.model = Some(spec_model);
            return selection.resolve(None, self.provider, &self.model);
        }
        current.resolve(None, self.provider, &self.model)
    }
}

#[async_trait]
impl SessionSpawner for HarnessSessionSpawner {
    fn set_selection(&self, selection: SubagentModelSelection) {
        self.selection.set(selection);
    }

    async fn spawn(
        &self,
        model: Option<&str>,
        tools: Option<&[String]>,
        message: &str,
        _tags: Option<&[String]>,
        workspace: Option<&Path>,
    ) -> Result<(String, String, TokenUsage)> {
        let (provider, model_id) = self.resolve_identity(model);
        let workspace_root = workspace
            .map(Path::to_path_buf)
            .unwrap_or_else(|| self.workspace.clone());
        let tool_list = tools.map(ToOwned::to_owned).or_else(|| self.tools.clone());
        let mut config = HarnessConfig::default()
            .with_provider(provider)
            .with_model(model_id)
            .with_workspace(workspace_root)
            .with_max_turns(self.max_turns)
            .with_auto_checkpoint(false);
        if let Some(names) = tool_list {
            config = config.with_tools(names);
        }
        let mut harness = AgentHarness::new(config).await?;
        let session_id = harness.session_id().to_owned();
        let (text, usage) = run_recorded_child(&mut harness, self.complex, message).await?;
        Ok((session_id, text, usage))
    }
}

/// Run a spawned child on `harness` as a ledger subagent task named by the
/// child session id. Shell tasks begun inside the child record this task as
/// their parent. A cancelled parent drops this future, and the task guard
/// records `cancelled`.
pub async fn run_recorded_child(
    harness: &mut AgentHarness,
    complex: bool,
    message: &str,
) -> Result<(String, TokenUsage)> {
    let mut task = task_ledger::begin(TaskKind::Subagent, harness.session_id().to_owned());
    if let Err(error) = task.recorded().await {
        tracing::warn!(error = %format!("{error:#}"), "task ledger: subagent start not recorded yet");
    }
    let task_id = task.id().map(str::to_owned);
    let outcome = task_ledger::scope(task_id, run_child(harness, complex, message)).await;
    match &outcome {
        Ok(_) => task.finish(TaskState::Succeeded, None),
        Err(error) => task.finish(TaskState::Failed, Some(&format!("{error:#}"))),
    }
    outcome
}

async fn run_child(
    harness: &mut AgentHarness,
    complex: bool,
    message: &str,
) -> Result<(String, TokenUsage)> {
    if complex {
        apply_complex_task_prompt(harness.service_mut());
        let outcome = run_complex_task(
            harness.service_mut(),
            message,
            &ComplexTaskConfig::default(),
        )
        .await?;
        return Ok((outcome.text, TokenUsage::default()));
    }
    let result = harness.run_task(message).await?;
    Ok((result.response, result.total_usage))
}

#[cfg(test)]
mod tests {
    use super::HarnessSessionSpawner;
    use crate::agent::subagent::SubagentModelSelection;
    use crate::llm::ProviderType;
    use std::path::PathBuf;

    #[test]
    fn resolve_identity_uses_spawn_model_arg() {
        let spawner =
            HarnessSessionSpawner::new(ProviderType::OpenAI, "gpt-4o", PathBuf::from("."));
        spawner.selection.set(SubagentModelSelection::from_parts(
            Some("anthropic"),
            Some("haiku"),
        ));
        let (provider, model) = spawner.resolve_identity(Some("openai/gpt-4o-mini"));
        assert_eq!(provider, ProviderType::OpenAI);
        assert_eq!(model, "gpt-4o-mini");
    }

    /// The subagent ledger wiring, against a real temp-dir ledger.
    #[cfg(all(feature = "sqlite-store", feature = "testing"))]
    mod ledger {
        use super::super::run_recorded_child;
        use crate::agent::service::AgentService;
        use crate::agent::tools::{BashTool, ToolRegistry};
        use crate::data::persistence::task_ledger::test_support::TestLedger;
        use crate::harness::config::{HarnessConfig, PermissionModeConfig};
        use crate::harness::core::AgentHarness;
        use crate::llm::ProviderType;
        use crate::testing::MockProvider;
        use std::sync::Arc;

        fn harness(provider: MockProvider) -> AgentHarness {
            let tools = Arc::new(ToolRegistry::new());
            tools.register(Arc::new(BashTool));
            let agent = crate::agent::Agent::new(Arc::new(provider), tools);
            let service =
                AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock service");
            AgentHarness::from_service(
                service,
                HarnessConfig {
                    max_turns: 1,
                    auto_checkpoint: false,
                    permission_mode: PermissionModeConfig::AcceptAll,
                    ..HarnessConfig::default()
                },
            )
            .expect("mock harness")
        }

        #[tokio::test]
        async fn child_run_records_subagent_and_parents_its_shell() {
            let ledger = TestLedger::open().await;
            let mut child = harness(MockProvider::with_tool_call(
                "bash",
                serde_json::json!({ "command": "echo unit-child" }),
            ));
            let session_id = child.session_id().to_owned();
            ledger
                .run(run_recorded_child(&mut child, false, "run it"))
                .await
                .expect("child run");

            let subagent = ledger.task_named(&session_id).await;
            assert_eq!(
                (subagent.kind.as_str(), subagent.state.as_str()),
                ("subagent", "succeeded")
            );
            let shell = ledger.task_named("echo unit-child").await;
            assert_eq!(shell.parent_id, Some(subagent.id));
        }

        #[tokio::test]
        async fn failed_child_run_records_failed_subagent() {
            let ledger = TestLedger::open().await;
            let mut child = harness(MockProvider::with_error());
            let session_id = child.session_id().to_owned();
            let outcome = ledger
                .run(run_recorded_child(&mut child, false, "run it"))
                .await;

            assert!(outcome.is_err(), "the mock provider fails the run");
            let subagent = ledger.task_named(&session_id).await;
            assert_eq!(subagent.state, "failed", "{subagent:?}");
            assert!(subagent.terminal_reason.is_some());
        }
    }
}
