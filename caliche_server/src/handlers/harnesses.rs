//! Registry of external CLI harnesses the UI can configure and dispatch.
//!
//! Operators register a command plus argv prefix. The prompt is never joined
//! into a shell string. Production still requires
//! `CALICHE_HARNESS_COMMAND_ALLOWLIST`.

use crate::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::orchestration::{
    CommandPolicy, ExternalCliRuntime, ExternalCliSpec, HarnessRequest, HarnessRuntime,
    PromptPlacement,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

type ApiResult<T> = Result<Json<T>, (StatusCode, String)>;

/// How the composed prompt is delivered to the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptPlacementDto {
    /// Prompt is the last argv element (value of a flag such as `-p`).
    #[default]
    LastArg,
    /// Insert `--` immediately before the prompt.
    LastArgEoo,
    /// Prompt is written to stdin.
    Stdin,
}

impl From<PromptPlacementDto> for PromptPlacement {
    fn from(value: PromptPlacementDto) -> Self {
        match value {
            PromptPlacementDto::LastArg => PromptPlacement::LastArg {
                end_of_options: false,
            },
            PromptPlacementDto::LastArgEoo => PromptPlacement::LastArg {
                end_of_options: true,
            },
            PromptPlacementDto::Stdin => PromptPlacement::Stdin,
        }
    }
}

/// Request to register an external CLI harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateHarnessRequest {
    pub name: String,
    #[serde(default)]
    pub preset_id: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub prompt_placement: PromptPlacementDto,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Registered harness returned to clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessInfo {
    pub id: String,
    pub name: String,
    pub preset_id: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub prompt_placement: PromptPlacementDto,
    pub capabilities: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_error: Option<String>,
    #[serde(default)]
    pub last_output: Option<String>,
    pub last_dispatched_at: Option<DateTime<Utc>>,
}

/// List envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessListResponse {
    pub harnesses: Vec<HarnessInfo>,
}

/// One-shot dispatch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchHarnessRequest {
    pub message: String,
}

/// One-shot dispatch result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchHarnessResponse {
    pub harness_id: String,
    pub name: String,
    pub ok: bool,
    pub text: String,
}

struct ManagerInner {
    harnesses: HashMap<String, HarnessInfo>,
}

/// In-memory CLI harness registry.
pub struct HarnessManager {
    inner: RwLock<ManagerInner>,
}

impl HarnessManager {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(ManagerInner {
                harnesses: HashMap::new(),
            }),
        })
    }

    /// Snapshot all registered harnesses, newest last.
    pub async fn list(&self) -> Vec<HarnessInfo> {
        let inner = self.inner.read().await;
        let mut items: Vec<HarnessInfo> = inner.harnesses.values().cloned().collect();
        items.sort_by(|left, right| left.created_at.cmp(&right.created_at));
        items
    }

    /// Validate and store a harness.
    pub async fn create(&self, req: CreateHarnessRequest) -> Result<HarnessInfo, String> {
        let info = validated_harness(req)?;
        let mut inner = self.inner.write().await;
        inner.harnesses.insert(info.id.clone(), info.clone());
        Ok(info)
    }

    /// Remove a harness. Errors when the id is unknown.
    pub async fn delete(&self, id: &str) -> Result<(), String> {
        let mut inner = self.inner.write().await;
        inner
            .harnesses
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| format!("unknown harness {id}"))
    }

    /// Run one task against a registered harness.
    pub async fn dispatch(
        &self,
        id: &str,
        message: &str,
    ) -> Result<DispatchHarnessResponse, String> {
        let info = {
            let inner = self.inner.read().await;
            inner
                .harnesses
                .get(id)
                .cloned()
                .ok_or_else(|| format!("unknown harness {id}"))?
        };
        let result = dispatch_registered(&info, message).await;
        let mut inner = self.inner.write().await;
        if let Some(stored) = inner.harnesses.get_mut(id) {
            stored.last_dispatched_at = Some(Utc::now());
            stored.last_output = Some(truncate_output(&result.text));
            stored.last_error = if result.ok {
                None
            } else {
                Some(truncate_output(&result.text))
            };
        }
        Ok(result)
    }
}

/// Validate a create request and build the stored record.
pub fn validated_harness(req: CreateHarnessRequest) -> Result<HarnessInfo, String> {
    let name = require_trimmed(&req.name, "name")?;
    let command = require_trimmed(&req.command, "command")?;
    if command.contains('\0') {
        return Err("command must not contain a NUL byte".to_string());
    }
    for arg in &req.args {
        if arg.contains('\0') {
            return Err("arguments must not contain a NUL byte".to_string());
        }
    }
    let cwd = req
        .cwd
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string);
    let spec = spec_from_fields(
        &command,
        req.args.clone(),
        cwd.clone(),
        req.prompt_placement,
    );
    let policy = CommandPolicy::from_env().map_err(|err| err.to_string())?;
    ExternalCliRuntime::new(spec, &policy).map_err(|err| err.to_string())?;
    Ok(HarnessInfo {
        id: Uuid::now_v7().to_string(),
        name,
        preset_id: req.preset_id.trim().to_string(),
        command,
        args: req.args,
        cwd,
        prompt_placement: req.prompt_placement,
        capabilities: req
            .capabilities
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        created_at: Utc::now(),
        last_error: None,
        last_output: None,
        last_dispatched_at: None,
    })
}

const LAST_OUTPUT_CHARS: usize = 4096;

fn truncate_output(text: &str) -> String {
    let count = text.chars().count();
    if count <= LAST_OUTPUT_CHARS {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(LAST_OUTPUT_CHARS).collect();
    clipped.push('…');
    clipped
}

fn spec_from_fields(
    command: &str,
    args: Vec<String>,
    cwd: Option<String>,
    placement: PromptPlacementDto,
) -> ExternalCliSpec {
    let mut spec = ExternalCliSpec::custom(command.to_string(), args);
    spec.cwd = cwd.map(PathBuf::from);
    spec.prompt_placement = PromptPlacement::from(placement);
    spec
}

async fn dispatch_registered(info: &HarnessInfo, message: &str) -> DispatchHarnessResponse {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return DispatchHarnessResponse {
            harness_id: info.id.clone(),
            name: info.name.clone(),
            ok: false,
            text: "message is required".to_string(),
        };
    }
    let spec = spec_from_fields(
        &info.command,
        info.args.clone(),
        info.cwd.clone(),
        info.prompt_placement,
    );
    let policy = match CommandPolicy::from_env() {
        Ok(policy) => policy,
        Err(err) => {
            return DispatchHarnessResponse {
                harness_id: info.id.clone(),
                name: info.name.clone(),
                ok: false,
                text: err.to_string(),
            };
        }
    };
    let runtime = match ExternalCliRuntime::new(spec, &policy) {
        Ok(runtime) => runtime,
        Err(err) => {
            return DispatchHarnessResponse {
                harness_id: info.id.clone(),
                name: info.name.clone(),
                ok: false,
                text: err.to_string(),
            };
        }
    };
    match runtime.execute(HarnessRequest::new(trimmed)).await {
        Ok(response) => DispatchHarnessResponse {
            harness_id: info.id.clone(),
            name: info.name.clone(),
            ok: true,
            text: response.text,
        },
        Err(err) => DispatchHarnessResponse {
            harness_id: info.id.clone(),
            name: info.name.clone(),
            ok: false,
            text: err.to_string(),
        },
    }
}

fn require_trimmed(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{field} is required"));
    }
    Ok(trimmed.to_string())
}

/// GET /api/orchestration/harnesses
pub async fn list_harnesses(State(state): State<AppState>) -> ApiResult<HarnessListResponse> {
    Ok(Json(HarnessListResponse {
        harnesses: state.harness_manager.list().await,
    }))
}

/// POST /api/orchestration/harnesses
pub async fn create_harness(
    State(state): State<AppState>,
    Json(req): Json<CreateHarnessRequest>,
) -> ApiResult<HarnessInfo> {
    state
        .harness_manager
        .create(req)
        .await
        .map(Json)
        .map_err(|err| (StatusCode::BAD_REQUEST, err))
}

/// DELETE /api/orchestration/harnesses/{id}
pub async fn delete_harness(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<serde_json::Value> {
    state
        .harness_manager
        .delete(&id)
        .await
        .map(|()| Json(serde_json::json!({ "ok": true })))
        .map_err(|err| (StatusCode::NOT_FOUND, err))
}

/// POST /api/orchestration/harnesses/{id}/dispatch
pub async fn dispatch_harness(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<DispatchHarnessRequest>,
) -> ApiResult<DispatchHarnessResponse> {
    match state.harness_manager.dispatch(&id, &req.message).await {
        Ok(response) => Ok(Json(response)),
        Err(err) => Err((StatusCode::NOT_FOUND, err)),
    }
}

#[cfg(test)]
mod tests {
    use super::{CreateHarnessRequest, HarnessManager, PromptPlacementDto, validated_harness};

    fn echo_request() -> CreateHarnessRequest {
        CreateHarnessRequest {
            name: "echo worker".to_string(),
            preset_id: "custom".to_string(),
            command: "echo".to_string(),
            args: Vec::new(),
            cwd: None,
            prompt_placement: PromptPlacementDto::LastArg,
            capabilities: vec!["coding".to_string()],
        }
    }

    #[test]
    fn rejects_empty_name_and_shell_command() {
        let mut req = echo_request();
        req.name = "  ".to_string();
        assert!(validated_harness(req).is_err());

        let mut req = echo_request();
        req.command = "bash".to_string();
        let err = validated_harness(req).expect_err("shell");
        assert!(err.to_lowercase().contains("shell"));
    }

    #[tokio::test]
    async fn create_list_delete_and_dispatch_echo() {
        let manager = HarnessManager::new();
        let created = manager.create(echo_request()).await.expect("create");
        assert_eq!(created.command, "echo");
        assert_eq!(manager.list().await.len(), 1);

        let dispatched = manager
            .dispatch(&created.id, "hello-harness")
            .await
            .expect("dispatch");
        assert!(dispatched.ok, "{}", dispatched.text);
        assert!(dispatched.text.contains("hello-harness"));
        let listed = manager.list().await;
        let stored = listed.first().expect("listed");
        assert!(
            stored
                .last_output
                .as_ref()
                .is_some_and(|text| text.contains("hello-harness"))
        );
        assert!(stored.last_error.is_none());

        manager.delete(&created.id).await.expect("delete");
        assert!(manager.list().await.is_empty());
        assert!(manager.dispatch(&created.id, "nope").await.is_err());
    }
}
