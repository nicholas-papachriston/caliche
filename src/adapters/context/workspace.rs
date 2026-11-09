use async_trait::async_trait;
#[cfg(feature = "tools-rag")]
use std::sync::Arc;

#[cfg(feature = "tools-rag")]
use crate::agent::hooks::is_steering_only_query;
#[cfg(feature = "tools-rag")]
use crate::agent::workspace_indexing::{WorkspaceIndexingReady, WorkspaceIndexingState};
#[cfg(feature = "tools-rag")]
use crate::rag::{RagService, WorkspaceContextQuery};

#[cfg(feature = "tools-rag")]
const INDEXING_WAIT_PARTIAL: std::time::Duration = std::time::Duration::from_millis(250);

#[async_trait]
pub trait WorkspaceContextProvider: Send + Sync {
    async fn fetch_workspace_context(&self, query: &str) -> Option<String>;
}

#[derive(Default)]
pub struct DefaultWorkspaceContextProvider {
    #[cfg(feature = "tools-rag")]
    rag: Option<Arc<RagService>>,
    #[cfg(feature = "tools-rag")]
    indexing_ready: Option<Arc<WorkspaceIndexingReady>>,
    #[cfg(feature = "tools-rag")]
    query: WorkspaceContextQuery,
}

impl DefaultWorkspaceContextProvider {
    #[cfg(feature = "tools-rag")]
    pub fn with_rag(
        rag: Option<Arc<RagService>>,
        indexing_ready: Option<Arc<WorkspaceIndexingReady>>,
        query: WorkspaceContextQuery,
    ) -> Self {
        Self {
            rag,
            indexing_ready,
            query,
        }
    }
}

#[async_trait]
impl WorkspaceContextProvider for DefaultWorkspaceContextProvider {
    async fn fetch_workspace_context(&self, query: &str) -> Option<String> {
        #[cfg(feature = "tools-rag")]
        {
            fetch_workspace_context_from_rag(
                self.rag.as_deref(),
                query,
                self.indexing_ready.as_ref(),
                &self.query,
            )
            .await
        }
        #[cfg(not(feature = "tools-rag"))]
        {
            let _ = query;
            None
        }
    }
}

#[cfg(feature = "tools-rag")]
#[must_use]
pub(crate) fn should_skip_workspace_context_query(query: &str) -> bool {
    is_steering_only_query(query)
}

#[cfg(feature = "tools-rag")]
pub async fn fetch_workspace_context_from_rag(
    rag: Option<&RagService>,
    query: &str,
    indexing_ready: Option<&Arc<WorkspaceIndexingReady>>,
    query_opts: &WorkspaceContextQuery,
) -> Option<String> {
    let rag = rag?;
    if should_skip_workspace_context_query(query) {
        return None;
    }
    if let Some(ready) = indexing_ready {
        if ready.state() == WorkspaceIndexingState::Partial {
            tracing::debug!(
                "Workspace indexing in progress, waiting up to {:?} then using available context",
                INDEXING_WAIT_PARTIAL
            );
            ready.wait_once_with_timeout(INDEXING_WAIT_PARTIAL).await;
        } else {
            tracing::debug!("Using workspace context (indexing complete)");
        }
    }

    let Ok((results, _, _)) = rag.query_context(query, query_opts).await else {
        tracing::debug!("Workspace RAG query failed");
        return None;
    };
    if results.is_empty() {
        tracing::debug!("Workspace RAG: no relevant chunks for query");
        return None;
    }
    tracing::info!(
        "Workspace RAG: using {} relevant chunk(s) for context",
        results.len()
    );
    Some(results.join("\n\n"))
}

#[cfg(all(test, feature = "tools-rag"))]
mod tests {
    use super::should_skip_workspace_context_query;

    #[test]
    fn skips_empty_and_harness_continuation_prompts() {
        use crate::agent::hooks::{CONTINUE_TASK_PROMPT, STREAM_TIMEOUT_CONTINUATION_PREFIX};

        assert!(should_skip_workspace_context_query(""));
        assert!(should_skip_workspace_context_query(CONTINUE_TASK_PROMPT));
        assert!(should_skip_workspace_context_query(&format!(
            "{STREAM_TIMEOUT_CONTINUATION_PREFIX} extra"
        )));
        assert!(!should_skip_workspace_context_query(
            "fix the failing indexer test in workspace_rag_wiring"
        ));
    }
}
