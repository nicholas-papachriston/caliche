//! Tool inspector trait and manager (Goose-inspired).

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;

/// Action recommended by an inspector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectionAction {
    Allow,
    Deny { reason: String },
    RequireApproval { reason: String },
}

impl InspectionAction {
    #[must_use]
    pub const fn precedence(&self) -> u8 {
        match self {
            Self::Allow => 0,
            Self::RequireApproval { .. } => 1,
            Self::Deny { .. } => 2,
        }
    }
}

/// Result produced by a single inspector for one tool call.
#[derive(Debug, Clone)]
pub struct InspectionResult {
    pub call_id: String,
    pub action: InspectionAction,
    pub reason: String,
    pub inspector_name: &'static str,
    pub confidence: f32,
}

/// Input for inspecting a single tool call.
#[derive(Debug, Clone, Copy)]
pub struct ToolInspectorRequest<'a> {
    pub tool_name: &'a str,
    pub arguments: &'a serde_json::Value,
    pub call_id: &'a str,
}

/// Trait for tool inspectors.
#[async_trait]
pub trait ToolInspector: Send + Sync {
    fn name(&self) -> &'static str;

    fn is_enabled(&self) -> bool {
        true
    }

    async fn inspect(&self, request: &ToolInspectorRequest<'_>) -> Result<Vec<InspectionResult>>;
}

/// Coordinates inspectors and merges their recommendations.
#[derive(Default)]
pub struct ToolInspectionManager {
    inspectors: Vec<Box<dyn ToolInspector>>,
}

impl ToolInspectionManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inspectors: Vec::new(),
        }
    }

    pub fn add_inspector(&mut self, inspector: Box<dyn ToolInspector>) {
        self.inspectors.push(inspector);
    }

    #[must_use]
    pub fn inspector_names(&self) -> Vec<&'static str> {
        self.inspectors.iter().map(|i| i.name()).collect()
    }

    /// Run all inspectors and return the highest-precedence action for `request`.
    pub async fn inspect_one(
        &self,
        request: &ToolInspectorRequest<'_>,
    ) -> Result<InspectionAction> {
        let results = self.inspect(std::slice::from_ref(request)).await?;
        Ok(merge_results_for_call(request.call_id, &results))
    }

    /// Inspect a batch of tool calls.
    pub async fn inspect(
        &self,
        requests: &[ToolInspectorRequest<'_>],
    ) -> Result<Vec<InspectionResult>> {
        let mut all = Vec::new();
        for inspector in &self.inspectors {
            if !inspector.is_enabled() {
                continue;
            }
            for request in requests {
                match inspector.inspect(request).await {
                    Ok(mut results) => all.append(&mut results),
                    Err(err) => {
                        tracing::error!(
                            inspector = inspector.name(),
                            error = %err,
                            "inspector failed"
                        );
                    }
                }
            }
        }
        Ok(all)
    }

    /// Merge per-call results into a map of `call_id` -> action.
    #[must_use]
    pub fn merge_by_call(results: &[InspectionResult]) -> HashMap<String, InspectionAction> {
        let mut map = HashMap::new();
        for result in results {
            let entry = map
                .entry(result.call_id.clone())
                .or_insert_with(|| InspectionAction::Allow);
            if result.action.precedence() > entry.precedence() {
                *entry = result.action.clone();
            }
        }
        map
    }
}

fn merge_results_for_call(call_id: &str, results: &[InspectionResult]) -> InspectionAction {
    let mut best = InspectionAction::Allow;
    for result in results {
        if result.call_id == call_id && result.action.precedence() > best.precedence() {
            best = result.action.clone();
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct DenyBash;
    struct AllowAll;

    #[async_trait]
    impl ToolInspector for DenyBash {
        fn name(&self) -> &'static str {
            "deny_bash"
        }

        async fn inspect(
            &self,
            request: &ToolInspectorRequest<'_>,
        ) -> Result<Vec<InspectionResult>> {
            if request.tool_name == "bash" {
                Ok(vec![InspectionResult {
                    call_id: request.call_id.to_string(),
                    action: InspectionAction::Deny {
                        reason: "bash blocked".to_string(),
                    },
                    reason: "bash blocked".to_string(),
                    inspector_name: self.name(),
                    confidence: 1.0,
                }])
            } else {
                Ok(vec![])
            }
        }
    }

    #[async_trait]
    impl ToolInspector for AllowAll {
        fn name(&self) -> &'static str {
            "allow_all"
        }

        async fn inspect(
            &self,
            request: &ToolInspectorRequest<'_>,
        ) -> Result<Vec<InspectionResult>> {
            Ok(vec![InspectionResult {
                call_id: request.call_id.to_string(),
                action: InspectionAction::Allow,
                reason: "ok".to_string(),
                inspector_name: self.name(),
                confidence: 1.0,
            }])
        }
    }

    #[tokio::test]
    async fn deny_outranks_allow() {
        let mut manager = ToolInspectionManager::new();
        manager.add_inspector(Box::new(AllowAll));
        manager.add_inspector(Box::new(DenyBash));
        let args = json!({"command":"ls"});
        let request = ToolInspectorRequest {
            tool_name: "bash",
            arguments: &args,
            call_id: "c1",
        };
        let action = manager.inspect_one(&request).await.expect("inspect");
        assert!(matches!(action, InspectionAction::Deny { .. }));
    }
}
