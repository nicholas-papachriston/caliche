//! Local Caliche runtime adapter for the coordinator.

use crate::agent::Agent;
use anyhow::Result;
use async_trait::async_trait;
use caliche_orchestration::{HarnessRequest, HarnessResponse, HarnessRuntime, HarnessUsage};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;

/// Pool of local [`Agent`] instances behind [`HarnessRuntime`].
pub struct LocalAgentRuntime {
    instances: Vec<Arc<RwLock<Agent>>>,
    next_instance: AtomicUsize,
}

impl LocalAgentRuntime {
    /// Build a pool of `pool_size` request clones from `agent`.
    #[must_use]
    pub fn new(agent: Agent, pool_size: usize) -> Self {
        let size = pool_size.max(1);
        let instances: Vec<Arc<RwLock<Agent>>> = (0..size)
            .map(|_| Arc::new(RwLock::new(agent.clone_for_request())))
            .collect();
        Self {
            instances,
            next_instance: AtomicUsize::new(0),
        }
    }

    fn pick_instance(&self) -> Arc<RwLock<Agent>> {
        let n = self.instances.len();
        let start = self.next_instance.fetch_add(1, Ordering::Relaxed);
        for offset in 0..n {
            let idx = (start + offset) % n;
            if self.instances[idx].try_write().is_ok() {
                return Arc::clone(&self.instances[idx]);
            }
        }
        Arc::clone(&self.instances[start % n])
    }
}

#[async_trait]
impl HarnessRuntime for LocalAgentRuntime {
    fn kind(&self) -> &'static str {
        "caliche"
    }

    async fn execute(&self, request: HarnessRequest) -> Result<HarnessResponse> {
        let instance = self.pick_instance();
        let mut agent = instance.write().await;

        let saved_persona = request
            .persona
            .as_ref()
            .map(|_| agent.system_prompt().to_string());
        let saved_tools = request.tools.as_ref().map(|_| agent.allowed_tools());

        if let Some(ref persona) = request.persona {
            agent.set_system_prompt(persona.as_str());
        }
        if let Some(ref tools) = request.tools {
            agent.set_allowed_tools(Some(tools.clone()));
        }

        let user_message = match request.instructions.as_ref() {
            Some(instructions) => format!(
                "Instructions:\n{}\n\nTask:\n{}",
                instructions.trim(),
                request.message
            ),
            None => request.message.clone(),
        };

        let turn = agent.chat(&user_message);
        let chat_outcome = match request.timeout {
            Some(timeout) => match tokio::time::timeout(timeout, turn).await {
                Ok(outcome) => outcome.map_err(|error| anyhow::anyhow!("{error}")),
                Err(_) => Err(anyhow::anyhow!(
                    "Dispatch to local agent timed out after {}ms",
                    timeout.as_millis()
                )),
            },
            None => turn.await.map_err(|error| anyhow::anyhow!("{error}")),
        };

        if request.persona.is_some()
            && let Some(persona) = saved_persona
        {
            agent.set_system_prompt(persona);
        }
        if request.tools.is_some()
            && let Some(tools) = saved_tools
        {
            agent.set_allowed_tools(tools);
        }

        let response = chat_outcome?;
        let usage = response.usage();
        Ok(HarnessResponse {
            text: response.text().to_string(),
            usage: HarnessUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            },
        })
    }
}
