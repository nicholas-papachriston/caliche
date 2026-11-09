//! `orchestration` API client methods.

use anyhow::Result;

use super::ApiClient;
use crate::api::types::{
    CoordinatorRequest, CoordinatorResponse, InterruptsListResponse, MemorySnapshot,
    MultiAgentRequest, MultiAgentResponse, ReasoningPatternInfo, ReasoningPatternRequest,
    ReasoningPatternResponse, ReasoningPatternsResponse, ResumeInterruptRequest,
    ResumeInterruptResponse, WorkflowRequest, WorkflowResponse, WorkflowStatusResponse,
};

impl ApiClient {
    /// Get available reasoning patterns
    pub async fn get_reasoning_patterns(&self) -> Result<Vec<ReasoningPatternInfo>> {
        let resp: ReasoningPatternsResponse = match self
            .rpc_call("reasoning.listPatterns", &serde_json::json!({}))
            .await
        {
            Ok(response) => response,
            Err(err) => {
                if Self::is_method_not_found_for(&err, "reasoning.listPatterns") {
                    anyhow::bail!("Reasoning features are not enabled on this server")
                }
                return Err(err);
            }
        };
        Ok(resp.patterns)
    }
    // ========================================================================
    // Reasoning Pattern Methods
    // ========================================================================

    /// Execute ReAct reasoning pattern
    pub async fn reasoning_react(
        &self,
        request: &ReasoningPatternRequest,
    ) -> Result<ReasoningPatternResponse> {
        match self.rpc_call("reasoning.react", request).await {
            Ok(response) => Ok(response),
            Err(err) => {
                if Self::is_method_not_found_for(&err, "reasoning.react") {
                    anyhow::bail!("Reasoning features are not enabled on this server")
                }
                Err(err)
            }
        }
    }

    /// Execute Plan-and-Execute reasoning pattern
    pub async fn reasoning_plan_execute(
        &self,
        request: &ReasoningPatternRequest,
    ) -> Result<ReasoningPatternResponse> {
        match self.rpc_call("reasoning.planExecute", request).await {
            Ok(response) => Ok(response),
            Err(err) => {
                if Self::is_method_not_found_for(&err, "reasoning.planExecute") {
                    anyhow::bail!("Reasoning features are not enabled on this server")
                }
                Err(err)
            }
        }
    }

    /// Execute Tree-of-Thoughts reasoning pattern
    pub async fn reasoning_tree_of_thoughts(
        &self,
        request: &ReasoningPatternRequest,
    ) -> Result<ReasoningPatternResponse> {
        match self.rpc_call("reasoning.treeOfThoughts", request).await {
            Ok(response) => Ok(response),
            Err(err) => {
                if Self::is_method_not_found_for(&err, "reasoning.treeOfThoughts") {
                    anyhow::bail!("Reasoning features are not enabled on this server")
                }
                Err(err)
            }
        }
    }
    // ========================================================================
    // Orchestration Methods
    // ========================================================================

    pub async fn multi_agent(&self, request: &MultiAgentRequest) -> Result<MultiAgentResponse> {
        self.rpc_call("orchestration.multiAgent", request).await
    }

    // ========================================================================
    // Memory Export / Import
    // ========================================================================

    pub async fn export_memory(&self, session_id: &str) -> Result<MemorySnapshot> {
        self.rpc_call(
            "memory.export",
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    pub async fn import_memory(&self, session_id: &str, snapshot: &MemorySnapshot) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                "memory.import",
                &serde_json::json!({ "session_id": session_id, "snapshot": snapshot }),
            )
            .await?;
        Ok(())
    }
    // -------------------------------------------------------------------------
    // Workflow / Orchestration
    // -------------------------------------------------------------------------

    pub async fn execute_workflow(&self, req: &WorkflowRequest) -> Result<WorkflowResponse> {
        self.rpc_call("orchestration.executeWorkflow", req).await
    }

    pub async fn get_workflow_status(&self, workflow_id: &str) -> Result<WorkflowStatusResponse> {
        self.rpc_call(
            "orchestration.getWorkflowStatus",
            &serde_json::json!({ "id": workflow_id }),
        )
        .await
    }

    pub async fn coordinator_route(&self, req: &CoordinatorRequest) -> Result<CoordinatorResponse> {
        self.rpc_call("orchestration.coordinator", req).await
    }

    pub async fn list_interrupts(&self) -> Result<InterruptsListResponse> {
        self.rpc_call("orchestration.listInterrupts", &serde_json::json!({}))
            .await
    }

    pub async fn resume_interrupt(
        &self,
        interrupt_id: &str,
        req: &ResumeInterruptRequest,
    ) -> Result<ResumeInterruptResponse> {
        self.rpc_call(
            "orchestration.resumeInterrupt",
            &serde_json::json!({ "id": interrupt_id, "feedback": req.feedback }),
        )
        .await
    }

    // -------------------------------------------------------------------------
    // Additional Reasoning Patterns
    // -------------------------------------------------------------------------

    pub async fn reasoning_self_ask(
        &self,
        req: &ReasoningPatternRequest,
    ) -> Result<ReasoningPatternResponse> {
        match self.rpc_call("reasoning.selfAsk", req).await {
            Ok(response) => Ok(response),
            Err(err) => {
                if Self::is_method_not_found_for(&err, "reasoning.selfAsk") {
                    anyhow::bail!("Reasoning features are not enabled on this server")
                }
                Err(err)
            }
        }
    }

    pub async fn reasoning_reflexion(
        &self,
        req: &ReasoningPatternRequest,
    ) -> Result<ReasoningPatternResponse> {
        match self.rpc_call("reasoning.reflexion", req).await {
            Ok(response) => Ok(response),
            Err(err) => {
                if Self::is_method_not_found_for(&err, "reasoning.reflexion") {
                    anyhow::bail!("Reasoning features are not enabled on this server")
                }
                Err(err)
            }
        }
    }
}
