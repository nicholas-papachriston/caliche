use crate::agent::kernel::turn::Handoff;
use crate::agent::tools::tool::{Tool, ToolMeta, ToolResult};
use crate::orchestration::{TaskInvocationPolicy, intersect_tools, rejected_tools};
use async_trait::async_trait;
use serde_json::json;

/// Special marker constant that indicates a handoff tool result
pub const HANDOFF_TOOL_NAME: &str = "transfer_to";

const DESCRIPTION: &str = "Transfer control to another agent. You can set the target's persona (system prompt), allowed tools, and detailed instructions for this task. Use when you need to delegate to a specialized subagent with specific role and constraints.";

/// Tool that allows an agent to explicitly hand off control to another agent.
///
/// The tool carries the caller's task invocation policy and tool ceiling, so a refused
/// target or an attempt to widen privileges fails here with a message the model can act on,
/// before the orchestration layer runs the same checks.
#[derive(Debug, Default)]
pub struct HandoffTool {
    policy: TaskInvocationPolicy,
    max_tools: Option<Vec<String>>,
    caller: Option<String>,
}

impl HandoffTool {
    /// Tool with no local restriction; the orchestration layer still authorizes the handoff.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict which targets the calling agent may transfer to.
    #[must_use]
    pub fn with_policy(mut self, policy: TaskInvocationPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the tools the calling agent may pass on. Requested tools are intersected with
    /// this ceiling, so a handoff can narrow privileges but never widen them.
    #[must_use]
    pub fn with_max_tools(mut self, max_tools: Vec<String>) -> Self {
        self.max_tools = Some(max_tools);
        self
    }

    /// Name of the calling agent, used in refusal messages.
    #[must_use]
    pub fn with_caller(mut self, caller: impl Into<String>) -> Self {
        self.caller = Some(caller.into());
        self
    }

    fn caller_name(&self) -> &str {
        self.caller.as_deref().unwrap_or("agent")
    }
}

impl ToolMeta for HandoffTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(HANDOFF_TOOL_NAME)
    }

    fn name(&self) -> &str {
        HANDOFF_TOOL_NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "target_agent": {
                    "type": "string",
                    "description": "The identifier of the agent to transfer control to"
                },
                "reason": {
                    "type": "string",
                    "description": "A brief explanation of why you are transferring control"
                },
                "persona": {
                    "type": "string",
                    "description": "Optional system prompt (persona) for the target agent for this task only. E.g. 'You are a research assistant. Use only RAG and fs with op=read.'"
                },
                "tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of tool names the target agent may use for this task. E.g. ['semantic_search', 'fs']"
                },
                "instructions": {
                    "type": "string",
                    "description": "Optional detailed instructions for the target agent, prepended to the task message"
                }
            },
            "required": ["target_agent", "reason"]
        })
    }
}

#[async_trait]
impl Tool for HandoffTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let target_agent = match arguments.get("target_agent").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => {
                return ToolResult::error("Missing required parameter: target_agent");
            }
        };

        let reason = match arguments.get("reason").and_then(|v| v.as_str()) {
            Some(r) => r.to_string(),
            None => {
                return ToolResult::error("Missing required parameter: reason");
            }
        };

        if let Err(e) = self.policy.authorize(self.caller_name(), &target_agent) {
            return ToolResult::error(format!("{e:#}"));
        }

        let persona = arguments
            .get("persona")
            .and_then(|v| v.as_str())
            .map(String::from);
        let requested_tools = arguments
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<String>>()
            });
        let tools = match (requested_tools, self.max_tools.as_deref()) {
            (Some(requested), ceiling) => {
                let rejected = rejected_tools(ceiling, &requested);
                if !rejected.is_empty() {
                    tracing::warn!(
                        caller = self.caller_name(),
                        target = %target_agent,
                        rejected = ?rejected,
                        "Dropped handoff tools the calling agent does not hold"
                    );
                }
                Some(intersect_tools(ceiling, &requested))
            }
            (None, _) => None,
        };
        let instructions = arguments
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(String::from);

        let handoff = Handoff {
            target_agent,
            reason,
            persona,
            tools,
            instructions,
        };

        // Return a special JSON payload that the Agent runtime can detect
        // The content is a JSON string containing the handoff information
        let handoff_json = serde_json::to_string(&handoff).unwrap_or_else(|_| {
            json!({
                "error": "Failed to serialize handoff"
            })
            .to_string()
        });

        ToolResult::success(handoff_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::PolicyEffect;

    fn handoff_args(target: &str, tools: Option<Vec<&str>>) -> serde_json::Value {
        let mut args = json!({ "target_agent": target, "reason": "specialist needed" });
        if let Some(tools) = tools
            && let Some(map) = args.as_object_mut()
        {
            map.insert("tools".to_string(), json!(tools));
        }
        args
    }

    #[tokio::test]
    async fn open_policy_passes_the_handoff_through() {
        let result = HandoffTool::new()
            .execute(handoff_args("reviewer", None))
            .await;
        assert!(result.success);
        assert!(result.content.contains("reviewer"));
    }

    #[tokio::test]
    async fn denied_target_is_refused_at_the_tool() {
        let tool = HandoffTool::new()
            .with_policy(TaskInvocationPolicy::deny_all())
            .with_caller("planner");

        let result = tool.execute(handoff_args("shell-runner", None)).await;
        assert!(!result.success);
        let error = result.error.unwrap_or_default();
        assert!(error.contains("planner"));
        assert!(error.contains("shell-runner"));
    }

    #[tokio::test]
    async fn allowed_target_survives_a_broad_deny() {
        let policy = TaskInvocationPolicy::deny_all().with_rule("reviewer", PolicyEffect::Allow);
        let tool = HandoffTool::new().with_policy(policy);

        assert!(tool.execute(handoff_args("reviewer", None)).await.success);
        assert!(!tool.execute(handoff_args("other", None)).await.success);
    }

    #[tokio::test]
    async fn requested_tools_are_capped_by_the_caller_ceiling() {
        let tool = HandoffTool::new().with_max_tools(vec!["fs".to_string()]);

        let result = tool
            .execute(handoff_args("reviewer", Some(vec!["fs", "bash"])))
            .await;
        assert!(result.success);

        let handoff: Handoff =
            serde_json::from_str(&result.content).expect("handoff payload is valid JSON");
        assert_eq!(
            handoff.tools.as_deref(),
            Some(["fs".to_string()].as_slice())
        );
    }
}
