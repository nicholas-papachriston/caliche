//! Code-mode JavaScript execution utilities.

use crate::{SandboxConfig, create_backend};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

const RESULT_MARKER: &str = "__CODE_MODE_RESULT__";
const REQUESTS_MARKER: &str = "__CODE_MODE_REQUESTS__";

#[derive(Debug, Clone)]
pub struct CodeModeSandboxConfig {
    pub sandbox: SandboxConfig,
    pub workspace: PathBuf,
    pub runtime_alias: String,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct CodeModeExecuteContext {
    pub api_base_url: String,
    pub api_key: String,
}

/// Run code-mode `search` JavaScript inside the configured sandbox.
pub async fn run_search(
    config: &CodeModeSandboxConfig,
    code: &str,
    schema: &Value,
) -> Result<Value> {
    let escaped_schema = serde_json::to_string(schema).context("serialize code-mode schema")?;
    let wrapper = format!(
        r#"
const __schema = JSON.parse({schema_json});
const __ops = (__schema && __schema.operations) ? __schema.operations : {{}};
const schema = {{}};
for (const __name of Object.keys(__ops)) {{
  schema[__name] = async (params = {{}}) => {{
    return {{
      operation: __name,
      params,
      description: __ops[__name]?.description ?? null,
      parameters: __ops[__name]?.parameters ?? null,
      returns: __ops[__name]?.returns ?? null
    }};
  }};
}}
const __result = await (async () => {{
{user_code}
}})();
console.log("{marker}" + JSON.stringify(__result ?? null));
"#,
        schema_json = escaped_schema,
        user_code = code,
        marker = RESULT_MARKER
    );

    run_script(config, &wrapper).await
}

/// Run code-mode `execute` JavaScript inside the configured sandbox.
pub async fn run_execute(
    config: &CodeModeSandboxConfig,
    code: &str,
    api_descriptor: &Value,
) -> Result<Value> {
    let auth_token = api_descriptor
        .get("auth")
        .and_then(|value| value.get("token"))
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let api_base_url = api_descriptor
        .get("base_url")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    if let (Some(api_base_url), Some(api_key)) = (api_base_url, auth_token) {
        return run_execute_with_api(
            config,
            code,
            api_descriptor,
            &CodeModeExecuteContext {
                api_base_url,
                api_key,
            },
        )
        .await;
    }

    let escaped_descriptor =
        serde_json::to_string(api_descriptor).context("serialize code-mode API descriptor")?;
    let wrapper = format!(
        r#"
const __api = JSON.parse({api_descriptor});
const __ops = (__api && __api.operations) ? __api.operations : {{}};
const api = {{}};
for (const __name of Object.keys(__ops)) {{
  api[__name] = async (params = {{}}) => {{
    return {{
      status: "simulated",
      operation: __name,
      params
    }};
  }};
}}
const __result = await (async () => {{
{user_code}
}})();
console.log("{marker}" + JSON.stringify(__result ?? null));
"#,
        api_descriptor = escaped_descriptor,
        user_code = code,
        marker = RESULT_MARKER
    );

    run_script(config, &wrapper).await
}

pub async fn run_execute_with_api(
    config: &CodeModeSandboxConfig,
    code: &str,
    api_descriptor: &Value,
    context: &CodeModeExecuteContext,
) -> Result<Value> {
    let mut responses = serde_json::json!({});
    for pass in 0..2 {
        let wrapper = build_execute_wrapper(code, api_descriptor, &responses)?;
        let stdout = run_script_stdout(config, &wrapper).await?;
        let pending = extract_requests(&stdout)?;
        if pending.is_empty() {
            return extract_result(&stdout);
        }
        if pass == 1 {
            return Err(anyhow!(
                "code-mode execution still has unresolved API calls after second pass"
            ));
        }

        let mut resolved = HashMap::<String, Value>::new();
        for request in pending {
            let key = request_key(&request.operation, &request.params);
            let value = call_forge_api(context, &request)
                .await
                .unwrap_or_else(|err| {
                    serde_json::json!({
                        "error": err.to_string(),
                        "operation": request.operation,
                    })
                });
            resolved.insert(key, value);
        }
        responses = serde_json::to_value(resolved).context("serialize resolved API responses")?;
    }

    Err(anyhow!("unreachable execute phase"))
}

async fn run_script(config: &CodeModeSandboxConfig, wrapper_source: &str) -> Result<Value> {
    let stdout = run_script_stdout(config, wrapper_source).await?;
    extract_result(&stdout)
}

async fn run_script_stdout(config: &CodeModeSandboxConfig, wrapper_source: &str) -> Result<String> {
    let sandbox_backend = create_backend(&config.sandbox)
        .await
        .context("create sandbox backend for code-mode execution")?;

    std::fs::create_dir_all(&config.workspace).with_context(|| {
        format!(
            "create code-mode workspace directory '{}'",
            config.workspace.display()
        )
    })?;

    let script_path = config.workspace.join("code_mode_entry.js");
    std::fs::write(&script_path, wrapper_source).with_context(|| {
        format!(
            "write code-mode wrapper script to '{}'",
            script_path.display()
        )
    })?;

    let sandbox_instance = sandbox_backend
        .create(&config.sandbox, &config.workspace)
        .await
        .context("create sandbox instance for code-mode execution")?;

    let script_path_string = script_path.to_string_lossy().into_owned();
    let command = vec![config.runtime_alias.as_str(), script_path_string.as_str()];
    let exec_result = sandbox_instance
        .exec(&command, None, None, config.timeout)
        .await
        .context("execute code-mode JavaScript wrapper")?;

    if exec_result.exit_code != 0 {
        bail!(
            "code-mode sandbox execution failed (exit {}): {}",
            exec_result.exit_code,
            exec_result.stderr
        );
    }

    Ok(exec_result.stdout)
}

fn extract_result(stdout: &str) -> Result<Value> {
    for line in stdout.lines() {
        if let Some(payload) = line.strip_prefix(RESULT_MARKER) {
            let parsed: Value =
                serde_json::from_str(payload).context("parse code-mode JSON result payload")?;
            return Ok(parsed);
        }
    }

    Err(anyhow!(
        "code-mode execution did not emit '{}' marker",
        RESULT_MARKER
    ))
}

#[derive(Debug, Clone)]
struct PendingRequest {
    operation: String,
    params: Value,
}

fn extract_requests(stdout: &str) -> Result<Vec<PendingRequest>> {
    for line in stdout.lines() {
        if let Some(payload) = line.strip_prefix(REQUESTS_MARKER) {
            let requests: Vec<PendingRequestWire> =
                serde_json::from_str(payload).context("parse pending request payload")?;
            return Ok(requests
                .into_iter()
                .map(|item| PendingRequest {
                    operation: item.operation,
                    params: item.params,
                })
                .collect());
        }
    }
    Ok(Vec::new())
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PendingRequestWire {
    operation: String,
    #[serde(default)]
    params: Value,
}

fn build_execute_wrapper(code: &str, api_descriptor: &Value, responses: &Value) -> Result<String> {
    let escaped_descriptor =
        serde_json::to_string(api_descriptor).context("serialize code-mode API descriptor")?;
    let escaped_responses =
        serde_json::to_string(responses).context("serialize code-mode execute responses")?;
    Ok(format!(
        r#"
const __api = JSON.parse({api_descriptor});
const __ops = (__api && __api.operations) ? __api.operations : {{}};
const __responses = JSON.parse({responses_json});
const __pending = [];
const api = {{}};
for (const __name of Object.keys(__ops)) {{
  api[__name] = async (params = {{}}) => {{
    const __key = __name + ":" + JSON.stringify(params ?? {{}});
    if (Object.prototype.hasOwnProperty.call(__responses, __key)) {{
      return __responses[__key];
    }}
    __pending.push({{ operation: __name, params }});
    return {{ status: "pending", operation: __name, params }};
  }};
}}
const __result = await (async () => {{
{user_code}
}})();
console.log("{requests_marker}" + JSON.stringify(__pending));
console.log("{result_marker}" + JSON.stringify(__result ?? null));
"#,
        api_descriptor = escaped_descriptor,
        responses_json = escaped_responses,
        user_code = code,
        requests_marker = REQUESTS_MARKER,
        result_marker = RESULT_MARKER
    ))
}

fn request_key(operation: &str, params: &Value) -> String {
    format!(
        "{operation}:{}",
        serde_json::to_string(params).unwrap_or_else(|_| "{}".to_string())
    )
}

async fn call_forge_api(
    context: &CodeModeExecuteContext,
    request: &PendingRequest,
) -> Result<Value> {
    let client = reqwest::Client::new();
    let base = context.api_base_url.trim_end_matches('/');
    let response = match request.operation.as_str() {
        "searchArtifacts" => {
            let query = request
                .params
                .get("query")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("searchArtifacts requires query"))?;
            let mut req = client
                .get(format!("{base}/search"))
                .header("x-forge-api-key", &context.api_key)
                .query(&[("query", query)]);
            if let Some(limit) = request.params.get("limit").and_then(Value::as_u64) {
                req = req.query(&[("limit", &limit.to_string())]);
            }
            req.send().await?
        }
        "getArtifact" => {
            let id = request
                .params
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("getArtifact requires id"))?;
            client
                .get(format!("{base}/artifacts/{id}"))
                .header("x-forge-api-key", &context.api_key)
                .send()
                .await?
        }
        "submitArtifact" => {
            client
                .post(format!("{base}/artifacts"))
                .header("x-forge-api-key", &context.api_key)
                .json(&request.params)
                .send()
                .await?
        }
        "vote" => {
            client
                .post(format!("{base}/vote"))
                .header("x-forge-api-key", &context.api_key)
                .json(&request.params)
                .send()
                .await?
        }
        "confirm" => {
            client
                .post(format!("{base}/confirm"))
                .header("x-forge-api-key", &context.api_key)
                .json(&request.params)
                .send()
                .await?
        }
        _ => {
            return Err(anyhow!(
                "unsupported code-mode execute operation: {}",
                request.operation
            ));
        }
    };
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(anyhow!(
            "forge api call failed for {}: status={} body={}",
            request.operation,
            status.as_u16(),
            body
        ));
    }
    if body.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str::<Value>(&body).context("parse Forge API response as JSON")
}
