//! Agent node execution with retries and idempotency keys.
//!
//! A workflow agent node calls a model, so it is the node type most likely to fail for
//! transient reasons and the most expensive one to repeat. Retries make a transient failure
//! recoverable; an idempotency key makes a repeat cheap by returning the recorded output
//! instead of calling the model again.

use super::constants::VAR_LAST_OUTPUT;
use super::{
    ExecutionContext, NodeErrorPolicy, NodeExecutionResult, NodeType, WorkflowEdge, WorkflowEngine,
    WorkflowNode,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// Node metadata key holding an explicit idempotency key.
const META_IDEMPOTENCY_KEY: &str = "idempotency_key";
/// Node metadata key that opts into a derived idempotency key.
const META_IDEMPOTENT: &str = "idempotent";
/// Node metadata key overriding the engine retry attempts.
const META_RETRY_ATTEMPTS: &str = "retry_attempts";
/// Node metadata key overriding the engine retry backoff.
const META_RETRY_BACKOFF_MS: &str = "retry_backoff_ms";
/// Metadata recorded on a result served from the idempotency cache.
const META_IDEMPOTENT_HIT: &str = "idempotent_cache_hit";
/// Metadata recording how many retries a node needed.
const META_RETRIES_USED: &str = "retries_used";

impl WorkflowEngine {
    pub(super) async fn run_agent_node(
        &self,
        node: &WorkflowNode,
        _nodes: &[WorkflowNode],
        _edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
        metadata: &mut HashMap<String, serde_json::Value>,
        start_time: DateTime<Utc>,
    ) -> Result<NodeExecutionResult> {
        let NodeType::Agent {
            agent_id,
            capability,
            prompt_template,
        } = &node.node_type
        else {
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                format!("expected Agent node, got {:?}", node.node_type),
                start_time,
                std::mem::take(metadata),
            ));
        };

        let prompt = format_agent_prompt(prompt_template, context)?;

        if let Err(reason) = self.reserve_agent_node(context) {
            tracing::warn!(
                node_id = %node.id,
                execution_id = %context.execution_id,
                reason = %reason,
                "Refusing a workflow agent node because the run budget is spent"
            );
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                reason,
                start_time,
                std::mem::take(metadata),
            ));
        }

        let cache_key = idempotency_key(node, context, prompt.as_str());
        if let Some(key) = cache_key.as_deref()
            && let Some(cached) = self.idempotency_cache.get(key)
        {
            let value = cached.value().clone();
            drop(cached);
            tracing::debug!(
                node_id = %node.id,
                idempotency_key = %key,
                "Serving agent node output from the idempotency cache"
            );
            metadata.insert(META_IDEMPOTENT_HIT.to_owned(), serde_json::json!(true));
            context.set_variable(format!("node_{}_output", node.id), value.clone());
            return Ok(NodeExecutionResult::ok(
                node.id.clone(),
                value,
                start_time,
                std::mem::take(metadata),
            ));
        }

        let attempts = self.retry_attempts_for(node);
        let backoff = retry_backoff_for(node, self.node_retry_backoff);
        let mut last_error = String::new();

        for attempt in 0..=attempts {
            if attempt > 0 {
                let delay = backoff.saturating_mul(1u32 << (attempt - 1).min(10));
                tracing::warn!(
                    node_id = %node.id,
                    attempt,
                    attempts,
                    delay_ms = delay.as_millis(),
                    error = %last_error,
                    "Retrying failed workflow agent node"
                );
                tokio::time::sleep(delay).await;
                if context.is_cancelled() {
                    break;
                }
            }

            match self
                .executor
                .execute_agent(
                    prompt.as_str(),
                    agent_id.as_deref(),
                    capability.as_deref(),
                    context,
                )
                .await
            {
                Ok((value, extra_meta)) => {
                    for (k, v) in extra_meta {
                        metadata.insert(k, v);
                    }
                    self.charge_tokens(context, metadata);
                    if attempt > 0 {
                        metadata.insert(META_RETRIES_USED.to_owned(), serde_json::json!(attempt));
                    }
                    if let Some(key) = cache_key {
                        self.idempotency_cache.insert(key, value.clone());
                    }
                    context.set_variable(format!("node_{}_output", node.id), value.clone());
                    return Ok(NodeExecutionResult::ok(
                        node.id.clone(),
                        value,
                        start_time,
                        std::mem::take(metadata),
                    ));
                }
                Err(e) => last_error = e.to_string(),
            }
        }

        if attempts > 0 {
            metadata.insert(META_RETRIES_USED.to_owned(), serde_json::json!(attempts));
        }
        Ok(NodeExecutionResult::err(
            node.id.clone(),
            last_error,
            start_time,
            std::mem::take(metadata),
        ))
    }

    /// Retry attempts for one node: node metadata wins, then the engine default when the
    /// error policy is `Retry`, otherwise none.
    fn retry_attempts_for(&self, node: &WorkflowNode) -> u32 {
        if let Some(attempts) = node
            .metadata
            .get(META_RETRY_ATTEMPTS)
            .and_then(serde_json::Value::as_u64)
        {
            return u32::try_from(attempts).unwrap_or(u32::MAX).min(10);
        }
        if matches!(self.node_error_policy, NodeErrorPolicy::Retry) {
            return self.node_retry_attempts.min(10);
        }
        0
    }

    /// Number of cached idempotent node outputs.
    pub fn idempotency_cache_len(&self) -> usize {
        self.idempotency_cache.len()
    }

    /// Drop every cached idempotent node output.
    pub fn clear_idempotency_cache(&self) {
        self.idempotency_cache.clear();
    }
}

/// Render an agent-node prompt, accepting both `{name}` and `{{name}}`.
///
/// Placeholders resolve against execution variables, including aliases published
/// after earlier nodes (`research`, `last_output`, `__last_output`).
fn format_agent_prompt(template: &str, context: &ExecutionContext) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut index = 0;
    let bytes = template.as_bytes();
    while index < bytes.len() {
        if bytes[index] == b'{' {
            if let Some((name, end)) = parse_mustache_placeholder(template, index) {
                out.push_str(&resolve_placeholder(name, context)?);
                index = end;
                continue;
            }
            if let Some((name, end)) = parse_brace_placeholder(template, index) {
                out.push_str(&resolve_placeholder(name, context)?);
                index = end;
                continue;
            }
        }
        let Some(character) = template[index..].chars().next() else {
            anyhow::bail!("invalid prompt template encoding");
        };
        out.push(character);
        index += character.len_utf8();
    }
    Ok(out)
}

fn parse_ident_at(template: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = template.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let mut end = start;
    while end < bytes.len() {
        let byte = bytes[end];
        if byte.is_ascii_alphanumeric() || byte == b'_' {
            end += 1;
        } else {
            break;
        }
    }
    if end == start {
        None
    } else {
        Some((start, end))
    }
}

fn parse_mustache_placeholder(template: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = template.as_bytes();
    if start + 1 >= bytes.len() || bytes[start] != b'{' || bytes[start + 1] != b'{' {
        return None;
    }
    let (ident_start, ident_end) = parse_ident_at(template, start + 2)?;
    if ident_end + 1 >= bytes.len() || bytes[ident_end] != b'}' || bytes[ident_end + 1] != b'}' {
        return None;
    }
    Some((&template[ident_start..ident_end], ident_end + 2))
}

fn parse_brace_placeholder(template: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = template.as_bytes();
    if start >= bytes.len() || bytes[start] != b'{' {
        return None;
    }
    let (ident_start, ident_end) = parse_ident_at(template, start + 1)?;
    if ident_end >= bytes.len() || bytes[ident_end] != b'}' {
        return None;
    }
    Some((&template[ident_start..ident_end], ident_end + 1))
}

fn json_value_to_prompt(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(flag) => {
            if *flag {
                "true".to_owned()
            } else {
                "false".to_owned()
            }
        }
        serde_json::Value::Number(number) => number.to_string(),
        other => other.to_string(),
    }
}

fn resolve_placeholder(name: &str, context: &ExecutionContext) -> Result<String> {
    let value = context.get_variable(name).or_else(|| match name {
        "last_output" => context.get_variable(VAR_LAST_OUTPUT),
        "__last_output" => context.get_variable("last_output"),
        _ => None,
    });
    match value {
        Some(resolved) => Ok(json_value_to_prompt(resolved)),
        None => anyhow::bail!("Missing required variable: {name}"),
    }
}

/// Per-node retry backoff, from node metadata or the engine default.
fn retry_backoff_for(node: &WorkflowNode, default: Duration) -> Duration {
    node.metadata
        .get(META_RETRY_BACKOFF_MS)
        .and_then(serde_json::Value::as_u64)
        .map(Duration::from_millis)
        .unwrap_or(default)
}

/// Resolve the idempotency key for a node, or `None` when the node opted out.
///
/// An explicit `idempotency_key` is used verbatim, so a caller can make two nodes — or two
/// runs — share one recorded result. `idempotent: true` derives a key from the workflow, the
/// node, and the rendered prompt, which makes a retry free while keeping different prompts
/// apart.
fn idempotency_key(
    node: &WorkflowNode,
    context: &ExecutionContext,
    prompt: &str,
) -> Option<String> {
    if let Some(key) = node
        .metadata
        .get(META_IDEMPOTENCY_KEY)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|key| !key.is_empty())
    {
        return Some(key.to_string());
    }
    let opted_in = node
        .metadata
        .get(META_IDEMPOTENT)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if !opted_in {
        return None;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    prompt.hash(&mut hasher);
    Some(format!(
        "{}:{}:{:016x}",
        context.workflow_id,
        node.id,
        hasher.finish()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_with_metadata(metadata: serde_json::Value) -> WorkflowNode {
        WorkflowNode::new(
            "agent",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: "task".to_string(),
            },
        )
        .with_metadata(metadata)
    }

    #[test]
    fn no_key_without_opt_in() {
        let node = node_with_metadata(serde_json::json!({}));
        let context = ExecutionContext::new("wf");
        assert!(idempotency_key(&node, &context, "prompt").is_none());
    }

    #[test]
    fn explicit_key_is_used_verbatim() {
        let node = node_with_metadata(serde_json::json!({ "idempotency_key": " order-123 " }));
        let context = ExecutionContext::new("wf");
        assert_eq!(
            idempotency_key(&node, &context, "prompt").as_deref(),
            Some("order-123")
        );
    }

    #[test]
    fn derived_key_changes_with_the_prompt() {
        let node = node_with_metadata(serde_json::json!({ "idempotent": true }));
        let context = ExecutionContext::new("wf");
        let first = idempotency_key(&node, &context, "prompt one").expect("derived key");
        let second = idempotency_key(&node, &context, "prompt two").expect("derived key");
        let repeat = idempotency_key(&node, &context, "prompt one").expect("derived key");

        assert_ne!(first, second);
        assert_eq!(first, repeat);
        assert!(first.starts_with("wf:"));
    }

    #[test]
    fn retry_backoff_reads_node_metadata() {
        let node = node_with_metadata(serde_json::json!({ "retry_backoff_ms": 1500 }));
        assert_eq!(
            retry_backoff_for(&node, Duration::from_millis(250)),
            Duration::from_millis(1500)
        );
        let plain = node_with_metadata(serde_json::json!({}));
        assert_eq!(
            retry_backoff_for(&plain, Duration::from_millis(250)),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn retry_attempts_follow_metadata_then_policy() {
        let engine = WorkflowEngine::new();
        let plain = node_with_metadata(serde_json::json!({}));
        assert_eq!(engine.retry_attempts_for(&plain), 0);

        let explicit = node_with_metadata(serde_json::json!({ "retry_attempts": 3 }));
        assert_eq!(engine.retry_attempts_for(&explicit), 3);

        let capped = node_with_metadata(serde_json::json!({ "retry_attempts": 999 }));
        assert_eq!(engine.retry_attempts_for(&capped), 10);

        let retrying = WorkflowEngine::new().with_node_error_policy(NodeErrorPolicy::Retry);
        assert_eq!(retrying.retry_attempts_for(&plain), 2);
    }

    #[test]
    fn mustache_placeholder_uses_node_name_without_wrapping_braces() {
        let mut context = ExecutionContext::new("wf");
        let node = WorkflowNode::new(
            "research",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: "find facts".to_string(),
            },
        );
        super::super::publish_template_outputs(
            &mut context,
            &node,
            &serde_json::json!("rust is fast"),
        );

        let prompt = format_agent_prompt(
            "summarize: {{research}}\nlegacy: {{__last_output}}\nalias: {{last_output}}",
            &context,
        )
        .expect("template should resolve named node output");
        assert_eq!(
            prompt,
            "summarize: rust is fast\nlegacy: rust is fast\nalias: rust is fast"
        );
        assert!(!prompt.contains("{rust is fast}"));
    }

    #[test]
    fn single_brace_placeholder_resolves_node_name() {
        let mut context = ExecutionContext::new("wf");
        let node = WorkflowNode::new(
            "research",
            NodeType::Agent {
                agent_id: None,
                capability: None,
                prompt_template: "find facts".to_string(),
            },
        );
        super::super::publish_template_outputs(&mut context, &node, &serde_json::json!("notes"));

        let prompt = format_agent_prompt("use {research}", &context).expect("single brace");
        assert_eq!(prompt, "use notes");
    }

    #[test]
    fn missing_named_placeholder_errors() {
        let context = ExecutionContext::new("wf");
        let error = format_agent_prompt("summarize: {{research}}", &context)
            .expect_err("unknown node name must fail closed");
        assert!(
            error
                .to_string()
                .contains("Missing required variable: research")
        );
    }
}
