//! GET /api/policy/effective and PUT /api/policy.

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use caliche::policy::{
    EffectivePolicy, HostCapability, PolicyDocument, PolicyError, host_for_process,
    open_policy_file, parse_host, write_policy_file,
};
use serde::Deserialize;

use crate::AppState;

/// Optional host label for the capability panel.
#[derive(Debug, Deserialize)]
pub struct PolicyHostQuery {
    /// `windows_job_only`, `macos_seatbelt`, `linux_landlock_no_network`, and the other labels.
    pub host: Option<String>,
}

/// GET /api/policy/effective
///
/// A missing file returns the locked new-deploy view. It does not write the file.
pub async fn get_effective_policy(
    State(state): State<AppState>,
    Query(query): Query<PolicyHostQuery>,
) -> Result<Json<EffectivePolicy>, (StatusCode, String)> {
    let host = resolve_host(query.host.as_deref())?;
    open_policy_file(&state.policy_path, host)
        .map(Json)
        .map_err(policy_status)
}

/// PUT /api/policy
///
/// Compiles the document before writing. A compile error leaves the file unchanged.
pub async fn put_policy(
    State(state): State<AppState>,
    Query(query): Query<PolicyHostQuery>,
    Json(document): Json<PolicyDocument>,
) -> Result<Json<EffectivePolicy>, (StatusCode, String)> {
    let host = resolve_host(query.host.as_deref())?;
    write_policy_file(&state.policy_path, &document, host)
        .map(Json)
        .map_err(policy_status)
}

fn resolve_host(value: Option<&str>) -> Result<HostCapability, (StatusCode, String)> {
    match value.map(str::trim).filter(|token| !token.is_empty()) {
        None => Ok(host_for_process(caliche_sandbox::landlock_tcp_enforceable())),
        Some(token) => parse_host(token).map_err(policy_status),
    }
}

fn policy_status(err: PolicyError) -> (StatusCode, String) {
    let status = match err {
        PolicyError::PolicyWrite(_) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    };
    (status, err.to_string())
}
