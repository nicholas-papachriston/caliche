//! `session` API client methods.

use anyhow::{Context, Result};
use caliche_contracts::rpc::methods as rpc_methods;

use super::ApiClient;
use crate::api::types::{
    CheckpointResponse, CodingRunResponse, ContextItemsResponse, CreateSessionRequest,
    CreateSessionResponse, DiffResponse, InteractionMode, LiveSessionInfo, MemoryEntry,
    MemorySearchResponse, MemoryStatsResponse, ModeResponse, PlanResponse, RedoResponse,
    RollbackResponse, TrackedFileInfo, UndoResponse, WorkspaceContextInfo,
};

impl ApiClient {
    // ========================================================================
    // Session Management Methods
    // ========================================================================

    /// Create a new agent session
    pub async fn create_session(
        &self,
        request: &CreateSessionRequest,
    ) -> Result<CreateSessionResponse> {
        self.rpc_call(rpc_methods::SESSIONS_CREATE, request).await
    }

    /// List every live session in this process (`GET /api/agent/sessions`).
    ///
    /// The array is the process-wide cache, the same list the CLI sees. It is
    /// not filtered by caller. `title` is absent when create omitted it.
    pub async fn list_live_sessions(&self) -> Result<Vec<LiveSessionInfo>> {
        let response = self
            .client
            .get(format!("{}/api/agent/sessions", self.base_url))
            .send()
            .await
            .context("request failed")?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{}", super::http_error_text(status, &body));
        }
        response.json().await.context("request failed")
    }

    /// Read one live session, including its title when create set one.
    pub async fn get_live_session(&self, session_id: &str) -> Result<LiveSessionInfo> {
        self.rpc_call(
            rpc_methods::SESSIONS_GET,
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Delete a session
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                rpc_methods::SESSIONS_DELETE,
                &serde_json::json!({ "session_id": session_id }),
            )
            .await?;
        Ok(())
    }

    /// Start a backend coding run and return its session ID plus the final response.
    ///
    /// This dispatches the `coding.run` RPC method. It is a coding-run request, not a
    /// multi-agent subagent delegation: the run has its own session and no parent agent
    /// context, ACL, or privilege ceiling applies to it.
    pub async fn spawn_coding_run(
        &self,
        task: &str,
        provider: Option<String>,
        model: Option<String>,
    ) -> Result<CodingRunResponse> {
        let result: serde_json::Value = self
            .rpc_call(
                rpc_methods::CODING_RUN,
                &serde_json::json!({
                    "task": task,
                    "workspace_root": null,
                    "provider": provider,
                    "model": model,
                    "mode": "edit",
                    "permission_mode": "default",
                    "max_turns": 20,
                    "tools": null,
                    "verify": "auto",
                    "context": {}
                }),
            )
            .await
            .context("coding task dispatch failed")?;
        let session_id = result
            .get("session_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let response = result
            .get("final_response")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();

        Ok(CodingRunResponse {
            session_id,
            response,
        })
    }

    /// `POST /api/agent/sessions/{id}/cancel`.
    ///
    /// Signals the in-flight turn and removes the session. This is not
    /// `sessions.delete`: that RPC only deletes and does not cancel the turn.
    pub async fn cancel_session(&self, session_id: &str) -> Result<()> {
        self.session_action(session_id, "cancel").await
    }

    /// `DELETE /api/agent/sessions/{id}`.
    pub async fn delete_live_session(&self, session_id: &str) -> Result<()> {
        let response = self
            .client
            .delete(format!("{}/api/agent/sessions/{session_id}", self.base_url))
            .send()
            .await
            .context("request failed")?;
        if response.status().is_success() {
            return Ok(());
        }
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("{}", super::http_error_text(status, &body));
    }

    async fn session_action(&self, session_id: &str, action: &str) -> Result<()> {
        let response = self
            .client
            .post(format!(
                "{}/api/agent/sessions/{session_id}/{action}",
                self.base_url
            ))
            .send()
            .await
            .context("request failed")?;
        if response.status().is_success() {
            return Ok(());
        }
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("{}", super::http_error_text(status, &body));
    }

    /// Undo last file change in a session
    pub async fn undo(&self, session_id: &str) -> Result<UndoResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_UNDO,
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Redo the most recently undone change set in a session.
    pub async fn redo(&self, session_id: &str) -> Result<RedoResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_REDO,
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Clear conversation history in a session
    pub async fn clear_history(&self, session_id: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                rpc_methods::SESSIONS_CLEAR_HISTORY,
                &serde_json::json!({ "session_id": session_id }),
            )
            .await?;
        Ok(())
    }

    /// Create a checkpoint in a session
    pub async fn create_checkpoint(
        &self,
        session_id: &str,
        label: &str,
    ) -> Result<CheckpointResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_CHECKPOINT,
            &serde_json::json!({ "session_id": session_id, "label": label }),
        )
        .await
    }

    /// Rollback to a checkpoint
    pub async fn rollback(
        &self,
        session_id: &str,
        checkpoint_id: &str,
    ) -> Result<RollbackResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_ROLLBACK,
            &serde_json::json!({ "session_id": session_id, "checkpoint_id": checkpoint_id }),
        )
        .await
    }

    /// Get current session mode
    pub async fn get_mode(&self, session_id: &str) -> Result<ModeResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_GET_MODE,
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Set session mode
    pub async fn set_mode(&self, session_id: &str, mode: InteractionMode) -> Result<ModeResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_SET_MODE,
            &serde_json::json!({ "session_id": session_id, "mode": mode }),
        )
        .await
    }

    /// Get session context info
    pub async fn get_context(&self, session_id: &str) -> Result<WorkspaceContextInfo> {
        self.rpc_call(
            rpc_methods::SESSIONS_GET_CONTEXT,
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    // ========================================================================
    // Memory Methods
    // ========================================================================

    /// Get memory statistics for a session
    pub async fn get_memory_stats(&self, session_id: &str) -> Result<MemoryStatsResponse> {
        self.rpc_call(
            "memory.stats",
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Search session memory
    pub async fn search_memory(
        &self,
        session_id: &str,
        query: &str,
    ) -> Result<MemorySearchResponse> {
        let raw: serde_json::Value = self
            .rpc_call(
                "memory.search",
                &serde_json::json!({ "session_id": session_id, "q": query }),
            )
            .await?;
        let mut out = MemorySearchResponse {
            results: Vec::new(),
            total: 0,
        };
        if let Some(arr) = raw.as_array() {
            out.total = arr.len();
            for item in arr {
                if let Some(tuple) = item.as_array()
                    && let Some(message) = tuple.first()
                {
                    let content = message
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    out.results.push(MemoryEntry {
                        content,
                        relevance: tuple
                            .get(1)
                            .and_then(serde_json::Value::as_f64)
                            .map(|v| v as f32),
                        timestamp: None,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Clear session memory
    pub async fn clear_memory(&self, session_id: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                "memory.clear",
                &serde_json::json!({ "session_id": session_id }),
            )
            .await?;
        Ok(())
    }

    // ========================================================================
    // Diff / Plan / Revert Methods
    // ========================================================================

    /// Get the session diff (what files were changed)
    pub async fn get_diff(&self, session_id: &str) -> Result<DiffResponse> {
        self.rpc_call(
            rpc_methods::SESSIONS_DIFF,
            &serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Get the session plan document (Plan mode)
    pub async fn get_plan(&self, session_id: &str) -> Result<PlanResponse> {
        let raw: serde_json::Value = self
            .rpc_call(
                rpc_methods::SESSIONS_GET_PLAN,
                &serde_json::json!({ "session_id": session_id }),
            )
            .await?;
        Ok(PlanResponse {
            content: raw
                .get("markdown")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string),
        })
    }

    // ========================================================================
    // Context Methods
    // ========================================================================

    /// Auto-detect project type in session
    pub async fn detect_project(&self, session_id: &str) -> Result<WorkspaceContextInfo> {
        let _: serde_json::Value = self
            .rpc_call(
                "sessions.detectProject",
                &serde_json::json!({ "session_id": session_id }),
            )
            .await?;
        self.get_context(session_id).await
    }

    /// Track a file in the session context
    pub async fn track_file(&self, session_id: &str, path: &str) -> Result<TrackedFileInfo> {
        self.rpc_call(
            rpc_methods::SESSIONS_TRACK_FILE,
            &serde_json::json!({ "session_id": session_id, "path": path }),
        )
        .await
    }
    // -------------------------------------------------------------------------
    // Advanced Context
    // -------------------------------------------------------------------------

    pub async fn get_relevant_context(
        &self,
        session_id: &str,
        query: &str,
    ) -> Result<ContextItemsResponse> {
        self.rpc_call(
            "sessions.getRelevantContext",
            &serde_json::json!({ "session_id": session_id, "query": query }),
        )
        .await
    }

    // -------------------------------------------------------------------------
    // Session diff / revert / clear / plan
    // -------------------------------------------------------------------------

    pub async fn delete_plan(&self, session_id: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call(
                "sessions.clearPlan",
                &serde_json::json!({ "session_id": session_id }),
            )
            .await?;
        Ok(())
    }
}
