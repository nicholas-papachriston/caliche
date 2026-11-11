//! Schema and step-graph validation for plugin definitions.
//!
//! Validates plugin structure and step pipelines before registration: unique step
//! names, valid references, sandbox policy, domain allowlist, loop bounds, nesting depth.

use crate::plugins::schema::{PluginDefinition, PluginSpec};
use crate::plugins::step_dsl::StepType::{Conditional, Http, Loop, Shell};
use crate::plugins::step_dsl::{Step, StepType};
use anyhow::{Context, Result, anyhow};
use std::collections::HashSet;

/// Policy for shell step execution. Used at validation time to reject shell steps
/// when the agent is in read-only mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SandboxPolicy {
    /// Shell steps are rejected at validation and runtime.
    #[default]
    ReadOnly,
    /// Shell steps are allowed (subject to sandbox handle at runtime in release builds).
    Full,
}

/// Context passed to validation (agent capabilities / config at validation time).
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    pub sandbox_policy: SandboxPolicy,
    pub domain_allowlist: Option<Vec<String>>,
    pub known_connection_refs: Option<HashSet<String>>,
}

const MAX_LOOP_ITERATIONS: usize = 1000;
const MAX_NESTING_DEPTH: usize = 5;

/// Validate a full plugin definition (structure + type-specific rules).
pub fn validate_plugin(def: &PluginDefinition, ctx: &ValidationContext) -> Result<()> {
    if def.name.is_empty() {
        return Err(anyhow!("plugin name must be non-empty"));
    }
    if def.id.is_empty() {
        return Err(anyhow!("plugin id must be non-empty"));
    }
    validate_plugin_spec(&def.spec, ctx)
}

fn validate_plugin_spec(spec: &PluginSpec, ctx: &ValidationContext) -> Result<()> {
    if let Some(result) = validate_primary_spec(spec, ctx) {
        return result;
    }
    validate_extra_spec(spec, ctx)
}

fn validate_primary_spec(spec: &PluginSpec, ctx: &ValidationContext) -> Option<Result<()>> {
    Some(match spec {
        PluginSpec::Tool(spec) => validate_tool_spec(spec, ctx),
        PluginSpec::Provider(spec) => validate_provider_spec(spec, ctx),
        PluginSpec::Backend(spec) => validate_backend_spec(spec, ctx),
        PluginSpec::Skill(spec) => validate_skill_spec(spec),
        PluginSpec::Source(spec) => validate_source_spec(spec, ctx),
        PluginSpec::Hook(spec) => validate_hook_spec(spec, ctx),
        _ => return None,
    })
}

fn validate_extra_spec(spec: &PluginSpec, ctx: &ValidationContext) -> Result<()> {
    match spec {
        PluginSpec::Embedding(spec) => validate_embedding_spec(spec, ctx),
        PluginSpec::Evaluator(spec) => validate_evaluator_spec(spec, ctx),
        PluginSpec::DocumentLoader(spec) => validate_document_loader_spec(spec, ctx),
        PluginSpec::StreamBackend(spec) => validate_stream_backend_spec(spec, ctx),
        PluginSpec::OutputParser(spec) => validate_output_parser_spec(spec, ctx),
        PluginSpec::ContextEngine(spec) => validate_context_engine_spec(spec),
        PluginSpec::Tool(_)
        | PluginSpec::Provider(_)
        | PluginSpec::Backend(_)
        | PluginSpec::Skill(_)
        | PluginSpec::Source(_)
        | PluginSpec::Hook(_) => Ok(()),
    }
}

fn validate_tool_spec(
    spec: &crate::plugins::schema::ToolPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.steps.is_empty() {
        return Err(anyhow!("tool plugin must have at least one step"));
    }
    validate_step_graph(&spec.steps, ctx, 0)?;
    Ok(())
}

fn validate_egress_url(url: &str, ctx: &ValidationContext) -> Result<()> {
    crate::security::http_policy::HttpPolicy::new(ctx.domain_allowlist.clone())
        .validate_static_url(url)
        .with_context(|| format!("plugin egress url rejected: {url}"))
}

fn validate_connection_spec(
    conn: &crate::plugins::schema::ConnectionSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    match conn {
        crate::plugins::schema::ConnectionSpec::Inline { url } => validate_egress_url(url, ctx),
        crate::plugins::schema::ConnectionSpec::Named { .. } => Ok(()),
    }
}

fn validate_provider_spec(
    spec: &crate::plugins::schema::ProviderPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.base_url.is_empty() {
        return Err(anyhow!("provider base_url must be non-empty"));
    }
    validate_egress_url(&spec.base_url, ctx)?;
    if spec.completion_endpoint.path.is_empty() {
        return Err(anyhow!(
            "provider completion_endpoint.path must be non-empty"
        ));
    }
    Ok(())
}

fn validate_backend_spec(
    spec: &crate::plugins::schema::BackendPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.dimension == 0 {
        return Err(anyhow!("backend dimension must be positive"));
    }
    validate_connection_spec(&spec.connection, ctx)?;
    Ok(())
}

fn validate_skill_spec(spec: &crate::plugins::schema::SkillPluginSpec) -> Result<()> {
    if spec.system_prompt_template.is_empty() {
        return Err(anyhow!("skill system_prompt_template must be non-empty"));
    }
    Ok(())
}

fn validate_source_spec(
    spec: &crate::plugins::schema::SourcePluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.uri_scheme.is_empty() {
        return Err(anyhow!("source plugin uri_scheme must be non-empty"));
    }
    if spec.resolver_steps.is_empty() {
        return Err(anyhow!(
            "source plugin must have at least one resolver step"
        ));
    }
    validate_step_graph(&spec.resolver_steps, ctx, 0)?;
    if let Some(ref step) = spec.content_transform {
        validate_step_graph(std::slice::from_ref(step), ctx, 0)?;
    }
    Ok(())
}

fn validate_hook_spec(
    spec: &crate::plugins::schema::HookPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.pre_tool_use_steps.is_empty() {
        return Err(anyhow!(
            "hook plugin must have at least one pre_tool_use step"
        ));
    }
    validate_step_graph(&spec.pre_tool_use_steps, ctx, 0)?;
    Ok(())
}

fn validate_embedding_spec(
    spec: &crate::plugins::schema::EmbeddingPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.base_url.is_empty() {
        return Err(anyhow!("embedding plugin base_url must be non-empty"));
    }
    validate_egress_url(&spec.base_url, ctx)?;
    if spec.embed_endpoint.path.is_empty() {
        return Err(anyhow!(
            "embedding plugin embed_endpoint.path must be non-empty"
        ));
    }
    if spec.request_mapping.texts_path.is_empty() {
        return Err(anyhow!(
            "embedding plugin request_mapping.texts_path must be non-empty"
        ));
    }
    if spec.response_mapping.embeddings_path.is_empty() {
        return Err(anyhow!(
            "embedding plugin response_mapping.embeddings_path must be non-empty"
        ));
    }
    Ok(())
}

fn validate_evaluator_spec(
    spec: &crate::plugins::schema::EvaluatorPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.steps.is_empty() {
        return Err(anyhow!("evaluator plugin must have at least one step"));
    }
    validate_step_graph(&spec.steps, ctx, 0)?;
    Ok(())
}

fn validate_document_loader_spec(
    spec: &crate::plugins::schema::DocumentLoaderPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.extension.is_empty() {
        return Err(anyhow!(
            "document loader plugin extension must be non-empty"
        ));
    }
    if spec.parsing_steps.is_empty() {
        return Err(anyhow!(
            "document loader plugin must have at least one parsing step"
        ));
    }
    validate_step_graph(&spec.parsing_steps, ctx, 0)?;
    Ok(())
}

fn validate_stream_backend_spec(
    spec: &crate::plugins::schema::StreamBackendPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.endpoints.append.is_none()
        && spec.endpoints.read_after.is_none()
        && spec.endpoints.last_id.is_none()
        && spec.endpoints.delete_stream.is_none()
    {
        return Err(anyhow!(
            "stream backend plugin must define at least one endpoint (append, read_after, last_id, delete_stream)"
        ));
    }
    validate_connection_spec(&spec.connection, ctx)?;
    Ok(())
}

fn validate_output_parser_spec(
    spec: &crate::plugins::schema::OutputParserPluginSpec,
    ctx: &ValidationContext,
) -> Result<()> {
    if spec.extraction_steps.is_empty() {
        return Err(anyhow!(
            "output parser plugin must have at least one extraction step"
        ));
    }
    validate_step_graph(&spec.extraction_steps, ctx, 0)?;
    Ok(())
}

fn validate_context_engine_spec(
    spec: &crate::plugins::schema::ContextEnginePluginSpec,
) -> Result<()> {
    if spec.assemble_steps.is_empty() {
        return Err(anyhow!(
            "context engine plugin must have at least one assemble step"
        ));
    }
    Ok(())
}

/// Validate a linear or nested step graph: unique names, valid refs, sandbox, allowlist, loop bounds, depth.
fn validate_step_graph(steps: &[Step], ctx: &ValidationContext, depth: usize) -> Result<()> {
    if depth > MAX_NESTING_DEPTH {
        return Err(anyhow!(
            "nesting depth exceeds maximum {}",
            MAX_NESTING_DEPTH
        ));
    }
    let mut seen_names = HashSet::new();
    let mut step_order: Vec<&str> = Vec::new();
    for step in steps {
        if !seen_names.insert(step.name.as_str()) {
            return Err(anyhow!("duplicate step name: {}", step.name));
        }
        step_order.push(step.name.as_str());
    }
    let has_return = steps
        .iter()
        .any(|s| matches!(&s.step_type, StepType::Return(..)));
    if steps.len() > 1 && !has_return {
        return Err(anyhow!("multi-step pipeline must end with a return step"));
    }
    for step in steps {
        validate_one_step(step, &step_order, ctx, depth)?;
    }
    Ok(())
}

fn validate_one_step(
    step: &Step,
    _step_order: &[&str],
    ctx: &ValidationContext,
    depth: usize,
) -> Result<()> {
    match &step.step_type {
        Shell(s) => {
            if ctx.sandbox_policy == SandboxPolicy::ReadOnly {
                return Err(anyhow!(
                    "shell step '{}' is not allowed when sandbox policy is read-only",
                    step.name
                ));
            }
            if s.command.contains("{{") {
                return Err(anyhow!(
                    "shell step '{}' command must be a static executable; interpolate only args",
                    step.name
                ));
            }
        }
        Http(h) => {
            for name in h.headers.keys() {
                if crate::security::http_policy::is_forbidden_egress_header(name) {
                    return Err(anyhow!(
                        "http step '{}' header '{name}' is forbidden",
                        step.name
                    ));
                }
            }
            let url = h.url.as_str();
            if !url.contains("{{") {
                validate_egress_url(url, ctx)
                    .with_context(|| format!("http step '{}' url rejected", step.name))?;
            }
        }
        StepType::DbQuery(d) => {
            if let Some(ref refs) = ctx.known_connection_refs
                && !refs.contains(&d.connection_ref)
            {
                return Err(anyhow!(
                    "db_query step '{}' references unknown connection: {}",
                    step.name,
                    d.connection_ref
                ));
            }
        }
        Loop(l) => {
            if l.max_iterations == 0 || l.max_iterations > MAX_LOOP_ITERATIONS {
                return Err(anyhow!(
                    "loop step '{}' max_iterations must be 1..{}",
                    step.name,
                    MAX_LOOP_ITERATIONS
                ));
            }
            validate_step_graph(&l.body_steps, ctx, depth + 1)
                .context("validating loop body steps")?;
        }
        Conditional(c) => {
            validate_step_graph(&c.then_steps, ctx, depth + 1)
                .context("validating conditional then_steps")?;
            validate_step_graph(&c.else_steps, ctx, depth + 1)
                .context("validating conditional else_steps")?;
        }
        // These step types have no structural constraints beyond the graph-level
        // checks (name uniqueness, step-ref ordering) applied by the caller.
        StepType::Transform(_)
        | StepType::VectorSearch(_)
        | StepType::LlmCall(_)
        | StepType::Return(_) => {}
    }
    Ok(())
}

/// Extract step names referenced in `{{steps.X...}}` from a template string.
/// Used to check that referenced steps appear earlier in the pipeline.
pub fn extract_step_refs_from_template(template: &str) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let bytes = template.as_bytes();
    let mut i = 0;
    while i + 2 <= bytes.len() {
        if !bytes[i..].starts_with(b"{{") {
            i += 1;
            continue;
        }
        i = skip_blanks(bytes, i + 2);
        if bytes[i..].starts_with(b"steps.") {
            let start = i + 6;
            i = step_name_end(bytes, start);
            let name = &template[start..i];
            if !name.is_empty() && !names.iter().any(|known| known == name) {
                names.push(name.to_string());
            }
        }
        i = skip_past_close(bytes, i);
    }
    names
}

/// The index of the first byte at or after `i` that is not a space or tab.
fn skip_blanks(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    i
}

/// The end of the step name that starts at `i`: ASCII letters, digits and `_`.
fn step_name_end(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()) {
        i += 1;
    }
    i
}

/// Skip to the next `}`, then past `}}` when it is there.
fn skip_past_close(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i] != b'}' {
        i += 1;
    }
    if bytes[i..].starts_with(b"}}") {
        i += 2;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::schema::ToolPluginSpec;
    use crate::plugins::step_dsl::{ReturnStep, StepType};
    use serde_json::json;

    #[test]
    fn step_refs_skip_multibyte_text_after_braces() {
        assert_eq!(
            extract_step_refs_from_template("{{aaaaa東}} {{steps.fetch.out}}"),
            vec!["fetch".to_string()]
        );
    }

    #[test]
    fn step_refs_follow_the_template_scan() {
        let cases: &[(&str, &[&str])] = &[
            ("no refs here", &[]),
            ("{{steps.a.out}} and {{ steps.b }}", &["a", "b"]),
            ("{{\tsteps.tab[0]}}", &["tab"]),
            (
                "{{steps.dup}} {{steps.dup.x}} {{steps.other}}",
                &["dup", "other"],
            ),
            ("{{steps.}} {{steps..x}}", &[]),
            ("{{steps.my-step}}", &["my"]),
            ("{{steps.open", &["open"]),
            ("{steps.single} {{inputs.x}}", &[]),
            ("{{x}}{{steps.after_close}}", &["after_close"]),
            ("{{steps.a}x}} {{steps.b}}", &["a", "b"]),
        ];
        for (template, expected) in cases {
            assert_eq!(
                extract_step_refs_from_template(template),
                expected.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "template: {template}"
            );
        }
    }

    #[test]
    fn test_duplicate_step_name_rejected() {
        let spec = ToolPluginSpec {
            parameters: json!({"type":"object","properties":{}}),
            steps: vec![
                Step {
                    name: "a".to_string(),
                    step_type: StepType::Return(ReturnStep {
                        value: "{{args}}".to_string(),
                    }),
                    condition: None,
                },
                Step {
                    name: "a".to_string(),
                    step_type: StepType::Return(ReturnStep {
                        value: "x".to_string(),
                    }),
                    condition: None,
                },
            ],
            is_readonly: true,
            capabilities: vec![],
            timeout_secs: None,
        };
        let def = PluginDefinition {
            id: "id".to_string(),
            name: "t".to_string(),
            version: "1.0.0".to_string(),
            description: "d".to_string(),
            author: None,
            tags: vec![],
            plugin_type: crate::plugins::schema::PluginType::Tool,
            spec: PluginSpec::Tool(spec),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        let ctx = ValidationContext::default();
        assert!(validate_plugin(&def, &ctx).is_err());
    }

    fn provider_plugin(base_url: &str) -> PluginDefinition {
        let spec = crate::plugins::schema::ProviderPluginSpec {
            base_url: base_url.to_string(),
            auth: crate::plugins::schema::ProviderAuth::None,
            models_endpoint: None,
            completion_endpoint: crate::plugins::schema::EndpointMapping {
                path: "/chat".to_string(),
                method: Some("POST".to_string()),
            },
            streaming: false,
            supports_tools: false,
            supports_vision: false,
            request_mapping: crate::plugins::schema::RequestMapping {
                messages_path: Some("messages".to_string()),
                max_tokens_path: None,
                temperature_path: None,
                tools_path: None,
                extra: Default::default(),
            },
            response_mapping: crate::plugins::schema::ResponseMapping {
                content_path: "content".to_string(),
                finish_reason_path: None,
                usage_path: None,
                tool_calls_path: None,
                extra: Default::default(),
            },
        };
        PluginDefinition {
            id: "id".to_string(),
            name: "evil".to_string(),
            version: "1.0.0".to_string(),
            description: "d".to_string(),
            author: None,
            tags: vec![],
            plugin_type: crate::plugins::schema::PluginType::Provider,
            spec: PluginSpec::Provider(spec),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn provider_loopback_base_url_is_rejected() {
        let err = validate_plugin(
            &provider_plugin("http://127.0.0.1/v1"),
            &ValidationContext::default(),
        )
        .expect_err("loopback");
        assert!(
            err.to_string().contains("blocked") || err.to_string().contains("rejected"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn provider_localhost_hostname_is_rejected() {
        let err = validate_plugin(
            &provider_plugin("http://localhost/v1"),
            &ValidationContext::default(),
        )
        .expect_err("localhost");
        assert!(
            err.to_string().contains("blocked") || err.to_string().contains("rejected"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn provider_public_base_url_does_not_require_dns() {
        validate_plugin(
            &provider_plugin("https://api.example.invalid/v1"),
            &ValidationContext::default(),
        )
        .expect("public hostname must load without DNS");
    }

    #[test]
    fn http_step_host_header_is_rejected() {
        let mut headers = std::collections::HashMap::new();
        headers.insert("Host".to_string(), "169.254.169.254".to_string());
        let spec = ToolPluginSpec {
            parameters: json!({"type":"object","properties":{}}),
            steps: vec![Step {
                name: "fetch".to_string(),
                step_type: StepType::Http(crate::plugins::step_dsl::HttpStep {
                    method: "GET".to_string(),
                    url: "https://api.example.invalid/v1".to_string(),
                    headers,
                    body: None,
                    auth_ref: None,
                    extract: None,
                }),
                condition: None,
            }],
            is_readonly: true,
            capabilities: vec![],
            timeout_secs: None,
        };
        let def = PluginDefinition {
            id: "id".to_string(),
            name: "t".to_string(),
            version: "1.0.0".to_string(),
            description: "d".to_string(),
            author: None,
            tags: vec![],
            plugin_type: crate::plugins::schema::PluginType::Tool,
            spec: PluginSpec::Tool(spec),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        let err = validate_plugin(&def, &ValidationContext::default()).expect_err("host header");
        assert!(
            err.to_string().to_lowercase().contains("header"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn shell_step_command_must_be_static() {
        let spec = ToolPluginSpec {
            parameters: json!({"type":"object","properties":{}}),
            steps: vec![Step {
                name: "run".to_string(),
                step_type: StepType::Shell(crate::plugins::step_dsl::ShellStep {
                    command: "{{args.cmd}}".to_string(),
                    args: vec!["{{args.file}}".to_string()],
                    parse_output: Default::default(),
                }),
                condition: None,
            }],
            is_readonly: false,
            capabilities: vec![],
            timeout_secs: None,
        };
        let def = PluginDefinition {
            id: "id".to_string(),
            name: "t".to_string(),
            version: "1.0.0".to_string(),
            description: "d".to_string(),
            author: None,
            tags: vec![],
            plugin_type: crate::plugins::schema::PluginType::Tool,
            spec: PluginSpec::Tool(spec),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        let ctx = ValidationContext {
            sandbox_policy: SandboxPolicy::Full,
            ..Default::default()
        };
        let err = validate_plugin(&def, &ctx).expect_err("interpolated command");
        assert!(
            err.to_string().contains("static"),
            "unexpected error: {err}"
        );
    }
}
