//! Step DSL runtime: template interpolation and async step execution.
//!
//! Executes step pipelines with `AgentCapabilities` (provider, http client,
//! credentials, optional vector backend) injected at construction time.

#[cfg(feature = "tools-sql")]
use crate::agent::tools::SqlExecutor;
#[cfg(feature = "tools-sql")]
use crate::agent::tools::sql_execution::SqlExecutionPolicy;
use crate::llm::EmbeddingProvider;
use crate::llm::embedding::EmbeddingRequest;
use crate::llm::provider::Provider;
use crate::messages::{CompletionContent, CompletionRequest, Message};
use crate::messages::{MessageContent, MessageRole};
use crate::plugins::http_policy::HttpPolicy;
use crate::plugins::schema::ToolPluginSpec;
use crate::plugins::step_dsl::{
    Condition, ConditionOp, Step, StepContext,
    StepType::{
        Conditional as ConditionalStep, DbQuery as DbQueryStep, Http as HttpStep,
        LlmCall as LlmCallStep, Loop as LoopStep, Return as ReturnStep, Shell as ShellStep,
        Transform as TransformStep, VectorSearch as VectorSearchStep,
    },
};
use crate::plugins::validation::SandboxPolicy;
use crate::traits::credentials::CredentialProvider;
use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
#[cfg(debug_assertions)]
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
#[cfg(debug_assertions)]
use tokio::process::Command as TokioCommand;
#[cfg(debug_assertions)]
use tokio::time::timeout;

mod transform;
pub(crate) use transform::run_transform_operations;

/// Capabilities injected at construction time so authored plugins never hold credentials.
#[derive(Clone)]
pub struct AgentCapabilities {
    pub llm_provider: Arc<dyn Provider>,
    pub http_client: Arc<reqwest::Client>,
    pub credential_provider: Arc<dyn CredentialProvider>,
    /// Optional SQL executor for db_query steps (set when agent has SQL tools configured).
    #[cfg(feature = "tools-sql")]
    pub sql_executor: Option<Arc<dyn SqlExecutor>>,
    #[cfg(feature = "vector-memory")]
    pub vector_backend: Option<Arc<dyn dripstone::Backend + Send + Sync>>,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub sandbox_policy: SandboxPolicy,
    pub domain_allowlist: Option<Vec<String>>,
    /// When `sandbox` feature is enabled and a handle is set, shell steps use sandbox `exec`.
    #[cfg(feature = "sandbox")]
    pub sandbox: Option<Arc<caliche_sandbox::SandboxHandle>>,
}

impl std::fmt::Debug for AgentCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("AgentCapabilities");
        d.field("llm_provider", &self.llm_provider.name())
            .field("http_client", &"Arc<reqwest::Client>")
            .field("credential_provider", &"Arc<dyn CredentialProvider>");
        #[cfg(feature = "tools-sql")]
        d.field("sql_executor", &self.sql_executor.is_some());
        #[cfg(feature = "vector-memory")]
        d.field("vector_backend", &self.vector_backend.is_some());
        d.field(
            "embedding_provider",
            &self.embedding_provider.as_ref().map(|e| e.name()),
        )
        .field("sandbox_policy", &self.sandbox_policy)
        .field("domain_allowlist", &self.domain_allowlist);
        #[cfg(feature = "sandbox")]
        d.field("sandbox", &self.sandbox.is_some());
        d.finish()
    }
}

/// Resolve a dot-path with optional array indexing on a JSON value.
/// Path format: `key.subkey[0].field` (no leading `$`).
pub fn resolve_path(root: &serde_json::Value, path: &str) -> Result<serde_json::Value> {
    crate::utils::json_path::evaluate(root, path).map_err(|err| anyhow!("{err}"))
}

/// Interpolate a template string with `{{namespace.path}}` placeholders.
/// If the entire template is a single `{{...}}` with no surrounding text,
/// returns the resolved Value as-is (for passing structured data between steps).
pub fn interpolate(template: &str, ctx: &StepContext) -> Result<serde_json::Value> {
    let template = template.trim();
    let single = extract_single_expression(template);
    if let Some(inner) = single {
        let value = resolve_expression(inner, ctx)?;
        return Ok(value);
    }
    let mut out = String::with_capacity(template.len());
    let mut i = 0;
    let bytes = template.as_bytes();
    while i < bytes.len() {
        if i + 2 <= bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'{' {
            let _start = i;
            i += 2;
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                i += 1;
            }
            let expr_start = i;
            while i < bytes.len() {
                if bytes[i] == b'}' && i + 1 < bytes.len() && bytes[i + 1] == b'}' {
                    let expr = std::str::from_utf8(&bytes[expr_start..i])
                        .map_err(|_| anyhow!("invalid utf-8 in template"))?
                        .trim();
                    let value = resolve_expression(expr, ctx)?;
                    out.push_str(&value_to_string(&value));
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    Ok(serde_json::Value::String(out))
}

fn extract_single_expression(template: &str) -> Option<&str> {
    let t = template.trim();
    if !t.starts_with("{{") || !t.ends_with("}}") {
        return None;
    }
    let inner = t[2..t.len() - 2].trim();
    if inner.is_empty() {
        return None;
    }
    if t.find("{{").map(|i| i + 2) != t.rfind("}}") {
        return None;
    }
    Some(inner)
}

pub(super) fn resolve_expression(expr: &str, ctx: &StepContext) -> Result<serde_json::Value> {
    let expr = expr.trim();
    let (namespace, path) = expr
        .find('.')
        .map(|i| (expr[..i].trim(), expr[i + 1..].trim()))
        .unwrap_or((expr, ""));
    let root: &serde_json::Value = match namespace {
        "args" => &ctx.args,
        "steps" => {
            let step_name = path.find('.').map(|i| &path[..i]).unwrap_or(path);
            ctx.step_outputs
                .get(step_name)
                .ok_or_else(|| anyhow!("step '{}' not found in context", step_name))?
        }
        "loop" => ctx
            .loop_vars
            .get(path)
            .or_else(|| ctx.loop_vars.get("item"))
            .ok_or_else(|| anyhow!("loop variable '{}' not found", path))?,
        "env" => {
            let var = path;
            let val = caliche_contracts::credential_env::var(var).unwrap_or_default();
            return Ok(serde_json::Value::String(val));
        }
        _ => return Err(anyhow!("unknown namespace '{}'", namespace)),
    };
    if path.is_empty() || path == namespace {
        return Ok(root.clone());
    }
    let subpath = if namespace == "steps" {
        path.find('.').map(|i| &path[i + 1..]).unwrap_or("")
    } else {
        path
    };
    if subpath.is_empty() {
        Ok(root.clone())
    } else {
        resolve_path(root, subpath)
    }
}

pub(super) fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => String::new(),
        _ => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
    }
}

/// Text inserted for `{{value}}` in `string_format` transform steps.
pub(super) fn transform_value_placeholder(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        _ => serde_json::to_string(v).unwrap_or_else(|_| String::new()),
    }
}

const SHELL_STEP_TIMEOUT_SECS: u64 = 60;
#[cfg(debug_assertions)]
const SHELL_MAX_OUTPUT_BYTES: usize = 10_048_576;

fn parse_shell_step_output(
    output: &str,
    mode: crate::plugins::step_dsl::ShellParseOutput,
) -> Result<serde_json::Value> {
    use crate::plugins::step_dsl::ShellParseOutput;
    match mode {
        ShellParseOutput::Raw => Ok(serde_json::Value::String(output.to_string())),
        ShellParseOutput::Lines => {
            let lines: Vec<serde_json::Value> = output
                .lines()
                .map(|line| serde_json::Value::String(line.to_string()))
                .collect();
            Ok(serde_json::Value::Array(lines))
        }
        ShellParseOutput::Json => serde_json::from_str(output.trim())
            .map_err(|e| anyhow!("shell step output is not valid JSON: {e}")),
        ShellParseOutput::Regex => Err(anyhow!(
            "shell step parse_output \"regex\" is not supported yet"
        )),
    }
}

fn require_sandbox_for_shell_in_release(capabilities: &AgentCapabilities) -> Result<()> {
    if cfg!(debug_assertions) {
        return Ok(());
    }
    #[cfg(feature = "sandbox")]
    if capabilities.sandbox.is_some() {
        return Ok(());
    }
    #[cfg(not(feature = "sandbox"))]
    let _ = capabilities;
    Err(anyhow!(
        "shell step requires a sandbox handle in release builds (enable the sandbox feature and configure a backend)"
    ))
}

/// Run shell step: optional sandbox `exec`, otherwise `tokio::process::Command`.
async fn run_shell_exec_capture(
    program: &str,
    args: &[String],
    cwd: Option<&std::path::Path>,
    capabilities: &AgentCapabilities,
    timeout_secs: u64,
) -> Result<String> {
    require_sandbox_for_shell_in_release(capabilities)?;

    #[cfg(feature = "sandbox")]
    if let Some(h) = capabilities.sandbox.as_ref() {
        let cmd_strings: Vec<String> = if args.is_empty() {
            vec!["sh".to_string(), "-c".to_string(), program.to_string()]
        } else {
            let mut v = vec![program.to_string()];
            v.extend_from_slice(args);
            v
        };
        let cmd_refs: Vec<&str> = cmd_strings.iter().map(String::as_str).collect();
        let wd_owned = cwd.map(|p| p.to_string_lossy().into_owned());
        let exec_result = h
            .exec(
                &cmd_refs,
                wd_owned.as_deref(),
                None::<&HashMap<String, String>>,
                Duration::from_secs(timeout_secs),
            )
            .await
            .context("sandbox exec for shell step")?;
        let note = crate::agent::tools::common::sandbox_exec::truncation_note(&exec_result);
        let mut out = exec_result.stdout;
        if !exec_result.stderr.is_empty() {
            if !out.is_empty() {
                out.push_str("\n\n[stderr]\n");
            } else {
                out.push_str("[stderr]\n");
            }
            out.push_str(&exec_result.stderr);
        }
        out.push_str(&note);
        if exec_result.exit_code != 0 {
            anyhow::bail!(
                "shell step exited with code {}: {}",
                exec_result.exit_code,
                out
            );
        }
        return Ok(out);
    }

    #[cfg(debug_assertions)]
    {
        let mut cmd = if args.is_empty() {
            #[cfg(windows)]
            {
                let comspec = std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string());
                let mut c = TokioCommand::new(comspec);
                c.arg("/C").arg(program);
                c
            }
            #[cfg(not(windows))]
            {
                let mut c = TokioCommand::new("sh");
                c.args(["-c", program]);
                c
            }
        } else {
            let mut c = TokioCommand::new(program);
            c.args(args);
            c
        };
        if let Some(wd) = cwd {
            cmd.current_dir(wd);
        }
        cmd.env_clear()
            .envs(caliche_contracts::child_env::host_child_env())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let output = timeout(Duration::from_secs(timeout_secs), cmd.output())
            .await
            .map_err(|_| anyhow!("shell step timed out after {timeout_secs}s"))?
            .map_err(|e| anyhow!("shell step spawn/run failed: {e}"))?;

        if output.stdout.len() + output.stderr.len() > SHELL_MAX_OUTPUT_BYTES {
            anyhow::bail!(
                "shell step output exceeds {} bytes (stdout+stderr combined)",
                SHELL_MAX_OUTPUT_BYTES
            );
        }

        let stdout_str = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr_str = String::from_utf8_lossy(&output.stderr).into_owned();
        let mut combined = stdout_str;
        if !stderr_str.is_empty() {
            if !combined.is_empty() {
                combined.push_str("\n\n[stderr]\n");
            } else {
                combined.push_str("[stderr]\n");
            }
            combined.push_str(&stderr_str);
        }

        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            anyhow::bail!("shell step exited with code {code}: {combined}");
        }

        Ok(combined)
    }

    #[cfg(not(debug_assertions))]
    {
        let _ = (program, args, cwd, timeout_secs);
        Err(anyhow!(
            "shell step requires a sandbox handle in release builds (no sandbox available for exec)"
        ))
    }
}

/// Async executor for step pipelines.
#[derive(Debug)]
pub struct StepRuntime {
    pub http_client: Arc<reqwest::Client>,
    pub sandbox_policy: SandboxPolicy,
    pub http_policy: HttpPolicy,
    #[cfg(feature = "tools-sql")]
    pub sql_execution_policy: Arc<SqlExecutionPolicy>,
}

impl StepRuntime {
    pub fn new(
        http_client: Arc<reqwest::Client>,
        sandbox_policy: SandboxPolicy,
        domain_allowlist: Option<Vec<String>>,
        #[cfg(feature = "tools-sql")] sql_execution_policy: Arc<SqlExecutionPolicy>,
    ) -> Self {
        Self {
            http_client,
            sandbox_policy,
            http_policy: HttpPolicy::new(domain_allowlist),
            #[cfg(feature = "tools-sql")]
            sql_execution_policy,
        }
    }

    pub async fn execute(
        &self,
        spec: &ToolPluginSpec,
        args: serde_json::Value,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        let timeout = spec
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(60));
        let deadline = tokio::time::Instant::now() + timeout;
        let mut ctx = StepContext {
            args: args.clone(),
            step_outputs: HashMap::new(),
            loop_vars: HashMap::new(),
        };
        for step in &spec.steps {
            if tokio::time::Instant::now() > deadline {
                return Err(anyhow!(
                    "plugin execution timeout after {}s",
                    spec.timeout_secs.unwrap_or(60)
                ));
            }
            if let Some(ref cond) = step.condition
                && !evaluate_condition(cond, &ctx)?
            {
                ctx.step_outputs
                    .insert(step.name.clone(), serde_json::Value::Null);
                continue;
            }
            let output = self
                .execute_one_step(step, &ctx, capabilities)
                .await
                .with_context(|| format!("step '{}'", step.name))?;
            ctx.step_outputs.insert(step.name.clone(), output);
        }
        ctx.step_outputs
            .get(spec.steps.last().map(|s| s.name.as_str()).unwrap_or(""))
            .cloned()
            .ok_or_else(|| anyhow!("no final step output"))
    }

    /// Run a list of steps with args (for hook/evaluator plugins). Returns the last step output.
    pub async fn execute_steps(
        &self,
        steps: &[Step],
        args: serde_json::Value,
        capabilities: &AgentCapabilities,
        timeout_secs: u64,
    ) -> Result<serde_json::Value> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
        let mut ctx = StepContext {
            args,
            step_outputs: HashMap::new(),
            loop_vars: HashMap::new(),
        };
        for step in steps {
            if tokio::time::Instant::now() > deadline {
                return Err(anyhow!("plugin execution timeout after {}s", timeout_secs));
            }
            if let Some(ref cond) = step.condition
                && !evaluate_condition(cond, &ctx)?
            {
                ctx.step_outputs
                    .insert(step.name.clone(), serde_json::Value::Null);
                continue;
            }
            let output = self
                .execute_one_step(step, &ctx, capabilities)
                .await
                .with_context(|| format!("step '{}'", step.name))?;
            ctx.step_outputs.insert(step.name.clone(), output);
        }
        steps
            .last()
            .and_then(|s| ctx.step_outputs.get(&s.name).cloned())
            .ok_or_else(|| anyhow!("no final step output"))
    }

    fn execute_one_step<'a>(
        &'a self,
        step: &'a Step,
        ctx: &'a StepContext,
        capabilities: &'a AgentCapabilities,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send + 'a>>
    {
        Box::pin(async move {
            match &step.step_type {
                HttpStep(h) => self.run_http(h, ctx, capabilities).await,
                TransformStep(t) => self.run_transform(t, ctx),
                ShellStep(s) => self.run_shell(s, ctx, capabilities).await,
                DbQueryStep(d) => self.run_db_query(d, ctx, capabilities).await,
                VectorSearchStep(v) => self.run_vector_search(v, ctx, capabilities).await,
                LlmCallStep(l) => self.run_llm_call(l, ctx, capabilities).await,
                ConditionalStep(c) => self.run_conditional(c, ctx, capabilities).await,
                LoopStep(l) => self.run_loop(l, ctx, capabilities).await,
                ReturnStep(r) => interpolate(&r.value, ctx),
            }
        })
    }

    async fn run_http(
        &self,
        h: &crate::plugins::step_dsl::HttpStep,
        ctx: &StepContext,
        _capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        let url = interpolate(&h.url, ctx)?
            .as_str()
            .ok_or_else(|| anyhow!("http step url must interpolate to a string"))?
            .to_string();
        let validated = self.http_policy.validate_resolved_url(&url)?;
        let client = validated.pinned_client()?;
        let method = h.method.to_uppercase();
        let request_url = validated.url.as_str();
        let mut req = match method.as_str() {
            "GET" => client.get(request_url),
            "POST" => client.post(request_url),
            "PUT" => client.put(request_url),
            "PATCH" => client.patch(request_url),
            "DELETE" => client.delete(request_url),
            _ => return Err(anyhow!("unsupported method: {}", h.method)),
        };
        for (k, v) in &h.headers {
            if crate::security::http_policy::is_forbidden_egress_header(k) {
                return Err(anyhow!("http step header '{k}' is forbidden"));
            }
            req = req.header(k, v);
        }
        if let Some(ref body) = h.body {
            req = req.json(body);
        }
        let resp = req.send().await.context("http request failed")?;
        let status = resp.status();
        let body = resp.text().await.context("read response body")?;
        if !status.is_success() {
            return Err(anyhow!("http {}: {}", status, body));
        }
        let value: serde_json::Value =
            serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body));
        Ok(if let Some(ref path) = h.extract {
            resolve_path(&value, path)?
        } else {
            value
        })
    }

    fn run_transform(
        &self,
        t: &crate::plugins::step_dsl::TransformStep,
        ctx: &StepContext,
    ) -> Result<serde_json::Value> {
        let input = resolve_expression(t.input.trim(), ctx)?;
        run_transform_operations(input, &t.operations, ctx)
    }

    async fn run_shell(
        &self,
        s: &crate::plugins::step_dsl::ShellStep,
        ctx: &StepContext,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        if self.sandbox_policy == SandboxPolicy::ReadOnly {
            return Err(anyhow!(
                "shell step not allowed in read-only sandbox policy"
            ));
        }
        require_sandbox_for_shell_in_release(capabilities)?;

        if s.command.contains("{{") {
            return Err(anyhow!(
                "shell step command must be a static executable; interpolate only args"
            ));
        }
        let program = s.command.trim();
        if program.is_empty() {
            return Err(anyhow!("shell step command is empty"));
        }

        let mut interp_args: Vec<String> = Vec::with_capacity(s.args.len());
        for arg_tmpl in &s.args {
            let v = interpolate(arg_tmpl, ctx)?;
            let s_arg = v
                .as_str()
                .ok_or_else(|| anyhow!("shell step argument must interpolate to a string"))?;
            interp_args.push(s_arg.to_string());
        }

        let cwd = std::env::current_dir().ok();

        let combined = run_shell_exec_capture(
            program,
            &interp_args,
            cwd.as_deref(),
            capabilities,
            SHELL_STEP_TIMEOUT_SECS,
        )
        .await?;

        parse_shell_step_output(&combined, s.parse_output)
    }

    #[cfg_attr(
        not(feature = "tools-sql"),
        expect(clippy::unused_async, reason = "the query path needs tools-sql")
    )]
    async fn run_db_query(
        &self,
        d: &crate::plugins::step_dsl::DbQueryStep,
        ctx: &StepContext,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        #[cfg(not(feature = "tools-sql"))]
        {
            let _ = (d, ctx, capabilities);
            return Err(anyhow!(
                "db_query step requires the 'tools-sql' feature and agent SQL configuration"
            ));
        }
        #[cfg(feature = "tools-sql")]
        {
            let executor = capabilities
                .sql_executor
                .as_ref()
                .ok_or_else(|| anyhow!("db_query step requires agent SQL configuration"))?;
            let query = interpolate(&d.query, ctx)?
                .as_str()
                .ok_or_else(|| anyhow!("db_query query must resolve to a string"))?
                .to_string();
            let result = self
                .sql_execution_policy
                .prepare_and_execute(executor.as_ref(), &query, 1000, false, None)
                .await
                .context("db_query execution failed")?;
            Ok(serde_json::json!({
                "rows": result.rows,
                "rows_affected": result.rows_affected,
                "execution_time_ms": result.execution_time_ms,
                "truncated": result.truncated,
            }))
        }
    }

    async fn run_vector_search(
        &self,
        v: &crate::plugins::step_dsl::VectorSearchStep,
        ctx: &StepContext,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        #[cfg(not(feature = "vector-memory"))]
        {
            let _ = (v, ctx, capabilities);
            return Err(anyhow!(
                "vector_search step requires the vector-memory feature and an embedding provider"
            ));
        }
        #[cfg(feature = "vector-memory")]
        {
            let embedding_provider = capabilities
                .embedding_provider
                .as_ref()
                .ok_or_else(|| anyhow!("vector_search step requires an embedding provider"))?;
            let backend = capabilities
                .vector_backend
                .as_ref()
                .ok_or_else(|| anyhow!("vector_search step requires a vector backend"))?;
            let query_text = interpolate(&v.query_text, ctx)?
                .as_str()
                .ok_or_else(|| anyhow!("vector_search query_text must resolve to a string"))?
                .to_string();
            let resp = embedding_provider
                .embed(EmbeddingRequest {
                    texts: vec![query_text],
                    model: None,
                })
                .await
                .context("vector_search embedding failed")?;
            let vector: Arc<[f32]> = resp
                .embeddings
                .first()
                .cloned()
                .map(Into::into)
                .ok_or_else(|| anyhow!("vector_search embedding returned no vector"))?;
            let mut query = dripstone::SearchQuery::new(vector).limit(v.limit);
            if let Some(ref ns) = v.namespace {
                query = query.namespace(ns.clone());
            }
            let results = backend
                .search(&query)
                .await
                .map_err(|e| anyhow!("vector_search failed: {}", e))?;
            let json_results: Vec<serde_json::Value> = results
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.chunk.id,
                        "score": r.score,
                        "rank": r.rank,
                        "content": r.chunk.content,
                        "metadata": r.chunk.metadata,
                    })
                })
                .collect();
            Ok(serde_json::Value::Array(json_results))
        }
    }

    async fn run_llm_call(
        &self,
        l: &crate::plugins::step_dsl::LlmCallStep,
        ctx: &StepContext,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        let prompt = interpolate(&l.prompt_template, ctx)?
            .as_str()
            .unwrap_or("")
            .to_string();
        let messages = vec![Message {
            role: MessageRole::User,
            content: MessageContent::Text(Arc::new(prompt)),
            metadata: None,
        }];
        let req = CompletionRequest {
            messages: Arc::new(messages),
            max_tokens: l.max_tokens.or(Some(1024)),
            temperature: l.temperature,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let resp = capabilities
            .llm_provider
            .complete(req)
            .await
            .context("llm call failed")?;
        let text = match resp.content {
            CompletionContent::Text { content } => content,
            CompletionContent::WithToolCalls { content, .. } => content,
            CompletionContent::Multimodal { content, .. } => content,
        };
        Ok(serde_json::Value::String(text))
    }

    async fn run_conditional(
        &self,
        c: &crate::plugins::step_dsl::ConditionalStep,
        ctx: &StepContext,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        let cond_result = evaluate_condition(&c.condition, ctx)?;
        let steps = if cond_result {
            &c.then_steps
        } else {
            &c.else_steps
        };
        let mut loop_ctx = ctx.clone();
        for step in steps {
            let out = self.execute_one_step(step, &loop_ctx, capabilities).await?;
            loop_ctx.step_outputs.insert(step.name.clone(), out);
        }
        Ok(steps
            .last()
            .and_then(|s| loop_ctx.step_outputs.get(&s.name))
            .cloned()
            .unwrap_or(serde_json::Value::Null))
    }

    async fn run_loop(
        &self,
        l: &crate::plugins::step_dsl::LoopStep,
        ctx: &StepContext,
        capabilities: &AgentCapabilities,
    ) -> Result<serde_json::Value> {
        let arr = resolve_expression(&l.over, ctx)?;
        let arr = arr
            .as_array()
            .ok_or_else(|| anyhow!("loop 'over' must be an array"))?;
        let mut results = Vec::new();
        for (idx, item) in arr.iter().enumerate().take(l.max_iterations) {
            let mut loop_ctx = ctx.clone();
            loop_ctx.loop_vars.insert(l.as_var.clone(), item.clone());
            loop_ctx
                .loop_vars
                .insert("index".to_string(), serde_json::json!(idx));
            loop_ctx.loop_vars.insert("item".to_string(), item.clone());
            for step in &l.body_steps {
                let out = self.execute_one_step(step, &loop_ctx, capabilities).await?;
                loop_ctx.step_outputs.insert(step.name.clone(), out);
            }
            if let Some(last) = l.body_steps.last()
                && let Some(v) = loop_ctx.step_outputs.get(&last.name)
            {
                results.push(v.clone());
            }
        }
        Ok(serde_json::Value::Array(results))
    }
}

fn evaluate_condition(cond: &Condition, ctx: &StepContext) -> Result<bool> {
    let value = resolve_expression(&cond.path, ctx)
        .ok()
        .unwrap_or(serde_json::Value::Null);
    match cond.op {
        ConditionOp::Exists => Ok(!value.is_null()),
        ConditionOp::NotExists => Ok(value.is_null()),
        ConditionOp::Truthy => Ok(value.as_bool().unwrap_or_else(|| {
            !value.is_null() && value != serde_json::Value::String(String::new())
        })),
        ConditionOp::Falsy => Ok(value.as_bool() == Some(false) || value.is_null()),
        ConditionOp::Eq => {
            let other = cond.value.as_ref().unwrap_or(&serde_json::Value::Null);
            Ok(value == *other)
        }
        ConditionOp::Ne => {
            let other = cond.value.as_ref().unwrap_or(&serde_json::Value::Null);
            Ok(value != *other)
        }
    }
}

#[cfg(all(test, feature = "plugins"))]
mod transform_op_tests {
    use super::*;
    use crate::plugins::step_dsl::TransformOperation;
    use serde_json::json;

    fn ctx_with_steps(outputs: HashMap<String, serde_json::Value>) -> StepContext {
        StepContext {
            args: json!({}),
            step_outputs: outputs,
            loop_vars: HashMap::new(),
        }
    }

    #[test]
    fn filter_array_keeps_matching() {
        let ctx = ctx_with_steps(HashMap::new());
        let input = json!([{"k": 1}, {"k": 2}, {"k": 1}]);
        let out = run_transform_operations(
            input,
            &[TransformOperation::Filter {
                path: "k".into(),
                value: json!(1),
            }],
            &ctx,
        )
        .expect("filter");
        assert_eq!(out, json!([{"k": 1}, {"k": 1}]));
    }

    #[test]
    fn flatten_one_level() {
        let ctx = ctx_with_steps(HashMap::new());
        let nested = json!([[1, 2], [3]]);
        let out = run_transform_operations(
            nested,
            &[TransformOperation::Flatten {
                path: String::new(),
            }],
            &ctx,
        )
        .expect("flatten");
        assert_eq!(out, json!([1, 2, 3]));
    }

    #[test]
    fn regex_extract_and_merge() {
        let ctx = ctx_with_steps(HashMap::new());
        let out = run_transform_operations(
            json!("id:42"),
            &[TransformOperation::RegexExtract {
                pattern: r"id:(\d+)".into(),
                group: Some(1),
            }],
            &ctx,
        )
        .expect("regex");
        assert_eq!(out, json!("42"));

        let mut steps = HashMap::new();
        steps.insert("extra".into(), json!({"b": 2}));
        let ctx2 = ctx_with_steps(steps);
        let merged = run_transform_operations(
            json!({"a": 1}),
            &[TransformOperation::Merge {
                with: "steps.extra".into(),
            }],
            &ctx2,
        )
        .expect("merge");
        assert_eq!(merged, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn string_format_inserts_value() {
        let ctx = StepContext {
            args: json!({"name": "World"}),
            step_outputs: HashMap::new(),
            loop_vars: HashMap::new(),
        };
        let out = run_transform_operations(
            json!("X"),
            &[TransformOperation::StringFormat {
                template: "prefix-{{value}}-{{args.name}}".into(),
            }],
            &ctx,
        )
        .expect("format");
        assert_eq!(out, json!("prefix-X-World"));
    }

    #[test]
    fn string_format_replaces_all_value_placeholders() {
        let ctx = StepContext {
            args: json!({}),
            step_outputs: HashMap::new(),
            loop_vars: HashMap::new(),
        };
        let out = run_transform_operations(
            json!(7),
            &[TransformOperation::StringFormat {
                template: "{{value}}={{value}}".into(),
            }],
            &ctx,
        )
        .expect("format");
        assert_eq!(out, json!("7=7"));
    }
}

#[cfg(all(test, feature = "plugins", debug_assertions, unix))]
mod shell_env_tests {
    use super::*;

    #[tokio::test]
    async fn shell_step_does_not_receive_host_credentials() {
        use crate::testing::planted_credential::{PRINT_PLANTED, rerun_with_planted_credential};
        if rerun_with_planted_credential(
            "plugins::runtime::shell_env_tests::shell_step_does_not_receive_host_credentials",
        ) {
            return;
        }
        let capabilities = AgentCapabilities {
            llm_provider: Arc::new(crate::testing::MockProvider::new(Vec::new())),
            http_client: Arc::new(reqwest::Client::new()),
            credential_provider: Arc::new(
                crate::traits::credentials::EnvironmentCredentialProvider::new(),
            ),
            #[cfg(feature = "tools-sql")]
            sql_executor: None,
            #[cfg(feature = "vector-memory")]
            vector_backend: None,
            embedding_provider: None,
            sandbox_policy: SandboxPolicy::ReadOnly,
            domain_allowlist: None,
            #[cfg(feature = "sandbox")]
            sandbox: None,
        };
        let out = run_shell_exec_capture(PRINT_PLANTED, &[], None, &capabilities, 10)
            .await
            .expect("shell step");
        assert_eq!(out, "unset");
    }
}
