use crate::handlers::sessions::SessionsState;
use crate::webhook_verify::{
    discord_signing_payload, inbound_public_url, signatures_equal, slack_signature_v0,
    slack_timestamp_is_fresh, twilio_expected_signature, verify_teams_signature,
};
use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use caliche::gateway::{
    BindingKey, ChannelKind, GatewayIngressEnvelope, GatewaySessionRef, IngressPayload,
};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteRow};
use sqlx::{Row, SqlitePool};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::OnceLock;
use url::form_urlencoded;

/// Generic body for webhook authentication / validation failures (details only in logs).
const WEBHOOK_REJECT_MESSAGE: &str = "invalid webhook request";
/// Generic body for webhook handler misconfiguration or internal errors.
const WEBHOOK_SERVER_ERROR_MESSAGE: &str = "webhook service error";

fn webhook_client_reject(
    status: StatusCode,
    detail: impl std::fmt::Display,
) -> (StatusCode, String) {
    tracing::warn!(
        target: "caliche_server::gateway::webhook",
        detail = %detail,
        "webhook request rejected"
    );
    (status, WEBHOOK_REJECT_MESSAGE.to_string())
}

fn webhook_server_error(detail: impl std::fmt::Display) -> (StatusCode, String) {
    tracing::error!(
        target: "caliche_server::gateway::webhook",
        detail = %detail,
        "webhook handler error"
    );
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        WEBHOOK_SERVER_ERROR_MESSAGE.to_string(),
    )
}

#[derive(Debug, Serialize)]
pub struct GatewayAdapterResponse {
    pub adapter_id: String,
    pub channel_kind: String,
    pub enabled: bool,
    pub priority: u16,
}

#[derive(Debug, Serialize)]
pub struct DeliveryDiagnosticsResponse {
    pub supported_targets: Vec<String>,
    pub retry_supported: bool,
    pub dead_letter_queue: String,
    pub dead_letter_count: usize,
    pub runtime_metrics: GatewayRuntimeMetricsResponse,
}

#[derive(Debug, Serialize)]
pub struct GatewayRuntimeMetricsResponse {
    pub accepted: u64,
    pub rejected_rate_limited: u64,
    pub rejected_backpressure: u64,
    pub duplicate_dropped: u64,
}

#[derive(Debug, Deserialize)]
pub struct SimulateIngressRequest {
    pub binding_key: BindingKey,
    pub message: String,
    #[serde(default)]
    pub preferred_session_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimulateIngressResponse {
    pub session_id: String,
    pub response: String,
    pub tools_used: bool,
    pub duration_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct OverrideBindingRequest {
    pub binding_key: BindingKey,
    pub session_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EnvelopeSimulateRequest {
    pub envelope: GatewayIngressEnvelope,
}

#[derive(Debug, Deserialize)]
pub struct ClusterRegisterRequest {
    pub agent_id: String,
    pub cwd: String,
    pub base_url: String,
    #[serde(default)]
    pub is_main: bool,
}

#[derive(Debug, Serialize)]
pub struct ClusterNodeResponse {
    pub agent_id: String,
    pub cwd: String,
    pub base_url: String,
    pub is_main: bool,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
pub struct PinClusterRouteRequest {
    pub route_key: String,
    pub target_agent_id: String,
}

#[derive(Debug, Deserialize)]
pub struct UnpinClusterRouteRequest {
    pub route_key: String,
}

#[derive(Debug, Serialize)]
pub struct ClusterRouteResponse {
    pub route_key: String,
    pub target_agent_id: String,
    pub updated_at: String,
}

pub async fn list_adapters_handler(
    State(adapter): State<SessionsState>,
) -> Result<Json<Vec<GatewayAdapterResponse>>, (StatusCode, String)> {
    let items = adapter
        .gateway_adapter_registry()
        .list()
        .into_iter()
        .map(|row| GatewayAdapterResponse {
            adapter_id: row.adapter_id,
            channel_kind: format!("{:?}", row.channel_kind),
            enabled: row.enabled,
            priority: row.priority,
        })
        .collect::<Vec<_>>();
    Ok(Json(items))
}

pub async fn enable_adapter_handler(
    State(adapter): State<SessionsState>,
    Path(adapter_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if adapter.gateway_set_adapter_enabled(&adapter_id, true) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("adapter not found: {adapter_id}"),
        ))
    }
}

pub async fn disable_adapter_handler(
    State(adapter): State<SessionsState>,
    Path(adapter_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if adapter.gateway_set_adapter_enabled(&adapter_id, false) {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("adapter not found: {adapter_id}"),
        ))
    }
}

pub async fn register_cluster_node_handler(
    headers: HeaderMap,
    crate::middleware::OptionalUserContext(user): crate::middleware::OptionalUserContext,
    Json(request): Json<ClusterRegisterRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    crate::handlers::gateway_validation::authorize_cluster_register(&headers, user.as_ref())
        .map_err(|(status, msg)| (status, msg.to_string()))?;
    let validated =
        crate::handlers::gateway_validation::validate_cluster_base_url(&request.base_url)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid base_url: {e}")))?;
    let normalized_base_url = validated.as_str().trim_end_matches('/').to_string();

    let pool = cluster_db_pool().await?;
    sqlx::query(
        r#"
        INSERT INTO gateway_cluster_nodes (agent_id, cwd, base_url, is_main, updated_at)
        VALUES (?1, ?2, ?3, ?4, datetime('now'))
        ON CONFLICT(agent_id) DO UPDATE SET
            cwd = excluded.cwd,
            base_url = excluded.base_url,
            is_main = excluded.is_main,
            updated_at = datetime('now')
        "#,
    )
    .bind(request.agent_id)
    .bind(request.cwd)
    .bind(normalized_base_url)
    .bind(request.is_main)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn list_cluster_nodes_handler()
-> Result<Json<Vec<ClusterNodeResponse>>, (StatusCode, String)> {
    let pool = cluster_db_pool().await?;
    let rows = sqlx::query(
        r#"
        SELECT agent_id, cwd, base_url, is_main, updated_at
        FROM gateway_cluster_nodes
        ORDER BY updated_at DESC
        "#,
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let nodes = rows
        .iter()
        .map(|row| {
            Ok(ClusterNodeResponse {
                agent_id: cluster_column(row, "agent_id")?,
                cwd: cluster_column(row, "cwd")?,
                base_url: cluster_column(row, "base_url")?,
                is_main: cluster_column(row, "is_main")?,
                updated_at: cluster_column(row, "updated_at")?,
            })
        })
        .collect::<Result<Vec<_>, (StatusCode, String)>>()?;
    Ok(Json(nodes))
}

pub async fn list_cluster_routes_handler()
-> Result<Json<Vec<ClusterRouteResponse>>, (StatusCode, String)> {
    let pool = cluster_db_pool().await?;
    let rows = sqlx::query(
        r#"
        SELECT route_key, target_agent_id, updated_at
        FROM gateway_cluster_routes
        ORDER BY updated_at DESC
        "#,
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let routes = rows
        .iter()
        .map(|row| {
            Ok(ClusterRouteResponse {
                route_key: cluster_column(row, "route_key")?,
                target_agent_id: cluster_column(row, "target_agent_id")?,
                updated_at: cluster_column(row, "updated_at")?,
            })
        })
        .collect::<Result<Vec<_>, (StatusCode, String)>>()?;
    Ok(Json(routes))
}

pub async fn pin_cluster_route_handler(
    Json(request): Json<PinClusterRouteRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let pool = cluster_db_pool().await?;

    let worker_exists = sqlx::query(
        r#"
        SELECT agent_id
        FROM gateway_cluster_nodes
        WHERE agent_id = ?1 AND is_main = 0
        LIMIT 1
        "#,
    )
    .bind(&request.target_agent_id)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .is_some();

    if !worker_exists {
        return Err((
            StatusCode::NOT_FOUND,
            format!("target worker not found: {}", request.target_agent_id),
        ));
    }

    sqlx::query(
        r#"
        INSERT INTO gateway_cluster_routes (route_key, target_agent_id, updated_at)
        VALUES (?1, ?2, datetime('now'))
        ON CONFLICT(route_key) DO UPDATE SET
            target_agent_id = excluded.target_agent_id,
            updated_at = datetime('now')
        "#,
    )
    .bind(request.route_key)
    .bind(request.target_agent_id)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn unpin_cluster_route_handler(
    Json(request): Json<UnpinClusterRouteRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let pool = cluster_db_pool().await?;
    let result = sqlx::query(
        r#"
        DELETE FROM gateway_cluster_routes
        WHERE route_key = ?1
        "#,
    )
    .bind(request.route_key)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if result.rows_affected() == 0 {
        return Err((StatusCode::NOT_FOUND, "route key not found".to_string()));
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn list_bindings_handler(
    State(adapter): State<SessionsState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<caliche::gateway::SessionBinding>>, (StatusCode, String)> {
    let rows = adapter
        .gateway_list_bindings(&session_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(rows))
}

pub async fn override_binding_handler(
    State(adapter): State<SessionsState>,
    Json(request): Json<OverrideBindingRequest>,
) -> Result<Json<caliche::gateway::SessionBinding>, (StatusCode, String)> {
    let binding = adapter
        .gateway_override_binding(request.binding_key, request.session_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(binding))
}

pub async fn simulate_ingress_handler(
    State(adapter): State<SessionsState>,
    Json(request): Json<SimulateIngressRequest>,
) -> Result<Json<SimulateIngressResponse>, (StatusCode, String)> {
    let envelope = GatewayIngressEnvelope {
        event_id: uuid::Uuid::now_v7().to_string(),
        idempotency_key: None,
        channel_kind: request.binding_key.channel_kind,
        channel_id: request.binding_key.channel_id.clone(),
        adapter_id: "simulate".to_string(),
        tenant_id: request.binding_key.tenant_id.clone(),
        user_id: request.binding_key.user_id.clone(),
        user_display_name: None,
        received_at: chrono::Utc::now(),
        session_ref: GatewaySessionRef {
            session_id: request.preferred_session_id.clone(),
            thread_id: request.binding_key.thread_id.clone(),
            conversation_id: None,
        },
        payload: IngressPayload {
            text: request.message.clone(),
            attachments: Vec::new(),
            is_command: false,
            reply_to_message_id: None,
        },
        metadata: std::collections::HashMap::new(),
    };
    let fairness_key = format!(
        "{}:{}:{}",
        request
            .binding_key
            .tenant_id
            .as_deref()
            .unwrap_or("default"),
        request.binding_key.channel_id,
        request.binding_key.user_id
    );
    let _admission = adapter
        .gateway_runtime()
        .submit(&fairness_key, envelope)
        .await
        .map_err(|e| (StatusCode::TOO_MANY_REQUESTS, e.to_string()))?;

    let session_id = Box::pin(
        adapter.resolve_gateway_session(request.binding_key, request.preferred_session_id),
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let turn = adapter
        .execute_turn(&session_id, &request.message)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(SimulateIngressResponse {
        session_id,
        response: turn.response,
        tools_used: turn.tools_used,
        duration_ms: turn.duration_ms,
    }))
}

pub async fn simulate_envelope_ingress_handler(
    State(adapter): State<SessionsState>,
    Json(request): Json<EnvelopeSimulateRequest>,
) -> Result<Json<SimulateIngressResponse>, (StatusCode, String)> {
    let result = Box::pin(execute_gateway_envelope(&adapter, request.envelope, false)).await?;
    Ok(Json(result))
}

fn escape_xml(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn fairness_key_from_envelope(envelope: &GatewayIngressEnvelope) -> String {
    format!(
        "{}:{}:{}",
        envelope.tenant_id.as_deref().unwrap_or("default"),
        envelope.channel_id,
        envelope.user_id
    )
}

fn binding_key_from_envelope(envelope: &GatewayIngressEnvelope) -> BindingKey {
    BindingKey {
        tenant_id: envelope.tenant_id.clone(),
        channel_kind: envelope.channel_kind,
        channel_id: envelope.channel_id.clone(),
        user_id: envelope.user_id.clone(),
        thread_id: envelope.session_ref.thread_id.clone(),
    }
}

async fn execute_gateway_envelope(
    adapter: &SessionsState,
    envelope: GatewayIngressEnvelope,
    allow_forwarding: bool,
) -> Result<SimulateIngressResponse, (StatusCode, String)> {
    if allow_forwarding
        && is_main_gateway()
        && let Some(forwarded) = maybe_forward_to_worker(envelope.clone()).await?
    {
        return Ok(forwarded);
    }

    let fairness_key = fairness_key_from_envelope(&envelope);
    let _admission = adapter
        .gateway_runtime()
        .submit(&fairness_key, envelope.clone())
        .await
        .map_err(|e| (StatusCode::TOO_MANY_REQUESTS, e.to_string()))?;

    let binding_key = binding_key_from_envelope(&envelope);
    let session_id = Box::pin(
        adapter.resolve_gateway_session(binding_key, envelope.session_ref.session_id.clone()),
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let turn = adapter
        .execute_turn(&session_id, &envelope.payload.text)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(SimulateIngressResponse {
        session_id,
        response: turn.response,
        tools_used: turn.tools_used,
        duration_ms: turn.duration_ms,
    })
}

/// Public Telegram webhook ingress endpoint.
///
/// Accepts Telegram update JSON and processes it through the Telegram adapter.
pub async fn telegram_webhook_ingress_handler(
    State(adapter): State<SessionsState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_telegram_secret(&headers)?;

    let telegram_adapter = adapter
        .gateway_adapter_registry()
        .resolve_for_channel(ChannelKind::Telegram)
        .into_iter()
        .next()
        .ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "no enabled Telegram gateway adapter found".to_string(),
            )
        })?;

    let raw_payload: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid Telegram JSON payload: {e}"),
        )
    })?;
    let envelope = telegram_adapter
        .normalize_ingress(raw_payload)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("failed to normalize Telegram ingress: {e}"),
            )
        })?;

    let result = Box::pin(execute_gateway_envelope(&adapter, envelope, true)).await?;
    Ok((StatusCode::OK, Json(result)))
}

/// Public Discord webhook ingress endpoint.
///
/// Accepts Discord message/create-style JSON payload and processes it through
/// the Discord adapter.
pub async fn discord_webhook_ingress_handler(
    State(adapter): State<SessionsState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_discord_signature(&headers, &body)?;

    let discord_adapter = adapter
        .gateway_adapter_registry()
        .resolve_for_channel(ChannelKind::Discord)
        .into_iter()
        .next()
        .ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "no enabled Discord gateway adapter found".to_string(),
            )
        })?;

    let raw_payload: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid Discord JSON payload: {e}"),
        )
    })?;
    let envelope = discord_adapter
        .normalize_ingress(raw_payload)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("failed to normalize Discord ingress: {e}"),
            )
        })?;

    let result = Box::pin(execute_gateway_envelope(&adapter, envelope, true)).await?;
    Ok((StatusCode::OK, Json(result)))
}

/// Public Twilio webhook ingress endpoint.
///
/// Accepts Twilio's form-encoded webhook payload, normalizes through the Twilio
/// adapter, executes a turn, and returns TwiML so Twilio can send the reply.
pub async fn twilio_webhook_ingress_handler(
    State(adapter): State<SessionsState>,
    headers: HeaderMap,
    uri: Uri,
    body: Bytes,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let form: HashMap<String, String> = form_urlencoded::parse(&body).into_owned().collect();
    verify_twilio_signature(&headers, &uri, &form)?;

    // Twilio may send non-message webhooks (e.g., Debugger/IP status callbacks) to the same URL.
    // Acknowledge those quickly so they do not trigger agent turns with empty input.
    if !is_twilio_message_ingress(&form) {
        tracing::info!(
            event = "gateway.twilio.callback_ack",
            reason = "non-message callback",
            from = form
                .get("From")
                .map(std::string::String::as_str)
                .unwrap_or(""),
            to = form
                .get("To")
                .map(std::string::String::as_str)
                .unwrap_or(""),
            has_body = form
                .get("Body")
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false),
            num_media = form
                .get("NumMedia")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0),
            message_sid = form
                .get("MessageSid")
                .or_else(|| form.get("SmsSid"))
                .map(std::string::String::as_str)
                .unwrap_or(""),
            "Twilio callback acknowledged without agent execution"
        );
        let twiml = r#"<?xml version="1.0" encoding="UTF-8"?><Response></Response>"#.to_string();
        return Ok((
            StatusCode::ACCEPTED,
            [("content-type", "application/xml")],
            twiml,
        ));
    }

    let twilio_adapter = adapter
        .gateway_adapter_registry()
        .resolve_for_channel(ChannelKind::Twilio)
        .into_iter()
        .next()
        .ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "no enabled Twilio gateway adapter found".to_string(),
            )
        })?;

    let raw_payload = serde_json::to_value(form).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid Twilio payload: {e}"),
        )
    })?;
    let envelope = twilio_adapter
        .normalize_ingress(raw_payload)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("failed to normalize Twilio ingress: {e}"),
            )
        })?;

    let turn = Box::pin(execute_gateway_envelope(&adapter, envelope, true)).await?;

    let twiml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><Response><Message>{}</Message></Response>"#,
        escape_xml(&turn.response)
    );

    Ok((StatusCode::OK, [("content-type", "application/xml")], twiml))
}

fn is_twilio_message_ingress(form: &HashMap<String, String>) -> bool {
    let has_sender = form
        .get("From")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let has_recipient = form
        .get("To")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let has_text = form
        .get("Body")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let has_media = form
        .get("NumMedia")
        .and_then(|value| value.parse::<usize>().ok())
        .map(|count| count > 0)
        .unwrap_or(false);

    has_sender && has_recipient && (has_text || has_media)
}

/// Public Slack Events API webhook ingress endpoint.
///
/// Accepts Slack event payloads, handles URL verification challenge, and routes
/// normalized message events through the Slack adapter.
pub async fn slack_webhook_ingress_handler(
    State(adapter): State<SessionsState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    verify_slack_signature(&headers, &body)?;

    let raw_payload: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid Slack JSON payload: {e}"),
        )
    })?;

    // Slack URL verification handshake
    if raw_payload
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|kind| kind == "url_verification")
    {
        let challenge = raw_payload
            .get("challenge")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "slack url_verification payload missing challenge".to_string(),
                )
            })?;
        let response = (
            StatusCode::OK,
            Json(serde_json::json!({ "challenge": challenge })),
        )
            .into_response();
        return Ok(response);
    }

    let slack_adapter = adapter
        .gateway_adapter_registry()
        .resolve_for_channel(ChannelKind::Slack)
        .into_iter()
        .next()
        .ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "no enabled Slack gateway adapter found".to_string(),
            )
        })?;

    let envelope = slack_adapter
        .normalize_ingress(raw_payload)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("failed to normalize Slack ingress: {e}"),
            )
        })?;

    let result = Box::pin(execute_gateway_envelope(&adapter, envelope, true)).await?;
    Ok((StatusCode::OK, Json(result)).into_response())
}

/// Public Microsoft Teams webhook ingress endpoint.
///
/// Accepts Bot Framework-style activity JSON and processes it through the Teams
/// adapter.
pub async fn teams_webhook_ingress_handler(
    State(adapter): State<SessionsState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    verify_teams_signature(&headers)
        .await
        .map_err(|err| webhook_client_reject(StatusCode::UNAUTHORIZED, err))?;
    let teams_adapter = adapter
        .gateway_adapter_registry()
        .resolve_for_channel(ChannelKind::Teams)
        .into_iter()
        .next()
        .ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "no enabled Teams gateway adapter found".to_string(),
            )
        })?;

    let raw_payload: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid Teams JSON payload: {e}"),
        )
    })?;
    let envelope = teams_adapter
        .normalize_ingress(raw_payload)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("failed to normalize Teams ingress: {e}"),
            )
        })?;

    let result = Box::pin(execute_gateway_envelope(&adapter, envelope, true)).await?;
    Ok((StatusCode::OK, Json(result)))
}

fn is_main_gateway() -> bool {
    caliche_contracts::credential_env::var("CALICHE_GATEWAY_ROLE")
        .map(|v| v.eq_ignore_ascii_case("main"))
        .unwrap_or(true)
}

/// Read the required `column` from a gateway cluster row. A NULL or mistyped
/// value is a 500 that names the column, not a panic. sqlx decodes a NULL
/// into `String` as `""` and into `bool` as `false`, so the value is read as
/// an `Option` and a NULL is refused here.
fn cluster_column<'r, T>(row: &'r SqliteRow, column: &str) -> Result<T, (StatusCode, String)>
where
    T: sqlx::Decode<'r, sqlx::Sqlite> + sqlx::Type<sqlx::Sqlite>,
{
    let internal = |reason: String| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("gateway cluster db: {reason}"),
        )
    };
    row.try_get::<Option<T>, _>(column)
        .map_err(|e| internal(e.to_string()))?
        .ok_or_else(|| internal(format!("column \"{column}\" is NULL")))
}

fn cluster_db_path() -> Result<std::path::PathBuf, (StatusCode, String)> {
    let home = dirs::home_dir().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to resolve home directory".to_string(),
        )
    })?;
    Ok(home.join(".caliche").join("gateway-cluster.db"))
}

async fn cluster_db_pool() -> Result<SqlitePool, (StatusCode, String)> {
    static POOL: OnceLock<SqlitePool> = OnceLock::new();
    if let Some(existing) = POOL.get() {
        return Ok(existing.clone());
    }

    let db_path = cluster_db_path()?;
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to create gateway cluster directory: {e}"),
            )
        })?;
    }
    let dsn = format!("sqlite://{}", db_path.display());
    let connect_options = SqliteConnectOptions::from_str(&dsn)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .create_if_missing(true);
    let pool = SqlitePool::connect_with(connect_options)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to open gateway cluster db: {e}"),
            )
        })?;
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS gateway_cluster_nodes (
            agent_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            base_url TEXT NOT NULL,
            is_main INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        "#,
    )
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS gateway_cluster_routes (
            route_key TEXT PRIMARY KEY,
            target_agent_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        "#,
    )
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if POOL.set(pool.clone()).is_ok() {
        return Ok(pool);
    }
    Ok(POOL.get().cloned().unwrap_or(pool))
}

#[derive(Debug, Clone)]
struct ClusterNode {
    agent_id: String,
    base_url: String,
}

async fn maybe_forward_to_worker(
    envelope: GatewayIngressEnvelope,
) -> Result<Option<SimulateIngressResponse>, (StatusCode, String)> {
    let route_key = route_key_from_envelope(&envelope);
    let Some(target) = resolve_worker_node_for_route(&route_key).await? else {
        return Ok(None);
    };

    // Revalidate the stored base_url on every forward in case the DB row
    // predates new security rules (e.g. after env changes or an operator
    // rotating the allowlist).
    let (client, validated) = crate::handlers::gateway_validation::cluster_forward_client(
        &target.base_url,
    )
    .map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!(
                "cluster node {} base_url rejected by validator: {e}",
                target.agent_id
            ),
        )
    })?;
    let endpoint = format!(
        "{}/api/gateway/ingress/simulate-envelope",
        validated.as_str().trim_end_matches('/')
    );
    let response = client
        .post(endpoint)
        .json(&EnvelopeSimulateRequest {
            envelope: envelope.clone(),
        })
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!("failed to forward ingress to worker: {e}"),
            )
        })?;
    if !response.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "worker {} rejected forwarded ingress with {}",
                target.agent_id,
                response.status()
            ),
        ));
    }

    // Cap response body size to avoid a malicious worker filling memory.
    let body_bytes = response.bytes().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("failed to read worker response body: {e}"),
        )
    })?;
    if body_bytes.len() > crate::handlers::gateway_validation::MAX_WORKER_BODY_BYTES {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "worker response body exceeded {} bytes cap",
                crate::handlers::gateway_validation::MAX_WORKER_BODY_BYTES
            ),
        ));
    }
    let payload = serde_json::from_slice::<SimulateIngressResponse>(&body_bytes).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("invalid worker response payload: {e}"),
        )
    })?;
    Ok(Some(payload))
}

fn route_key_from_envelope(envelope: &GatewayIngressEnvelope) -> String {
    let key = binding_key_from_envelope(envelope);
    format!(
        "{}|{:?}|{}|{}|{}",
        key.tenant_id.as_deref().unwrap_or("default"),
        key.channel_kind,
        key.channel_id,
        key.user_id,
        key.thread_id.unwrap_or_default()
    )
}

fn hash_to_index(seed: &str, size: usize) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % size
}

async fn resolve_worker_node_for_route(
    route_key: &str,
) -> Result<Option<ClusterNode>, (StatusCode, String)> {
    let pool = cluster_db_pool().await?;
    let workers_rows = sqlx::query(
        r#"
        SELECT agent_id, base_url
        FROM gateway_cluster_nodes
        WHERE is_main = 0
        ORDER BY updated_at DESC
        "#,
    )
    .fetch_all(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if workers_rows.is_empty() {
        return Ok(None);
    }
    let workers = workers_rows
        .iter()
        .map(|row| {
            Ok(ClusterNode {
                agent_id: cluster_column(row, "agent_id")?,
                base_url: cluster_column(row, "base_url")?,
            })
        })
        .collect::<Result<Vec<_>, (StatusCode, String)>>()?;

    let existing = sqlx::query(
        r#"
        SELECT target_agent_id
        FROM gateway_cluster_routes
        WHERE route_key = ?1
        "#,
    )
    .bind(route_key)
    .fetch_optional(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    if let Some(row) = existing {
        let target_id: String = cluster_column(&row, "target_agent_id")?;
        if let Some(node) = workers.iter().find(|n| n.agent_id == target_id) {
            return Ok(Some(node.clone()));
        }
    }

    let selected = workers[hash_to_index(route_key, workers.len())].clone();
    sqlx::query(
        r#"
        INSERT INTO gateway_cluster_routes (route_key, target_agent_id, updated_at)
        VALUES (?1, ?2, datetime('now'))
        ON CONFLICT(route_key) DO UPDATE SET
            target_agent_id = excluded.target_agent_id,
            updated_at = datetime('now')
        "#,
    )
    .bind(route_key)
    .bind(&selected.agent_id)
    .execute(&pool)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Some(selected))
}

fn verify_telegram_secret(headers: &HeaderMap) -> Result<(), (StatusCode, String)> {
    let expected =
        match crate::webhook_verify::resolve_webhook_secret("TELEGRAM_WEBHOOK_SECRET_TOKEN") {
            Ok(Some(value)) => value,
            Ok(None) => return Ok(()),
            Err(err) => return Err(webhook_server_error(err.to_string())),
        };
    let received = headers
        .get("x-telegram-bot-api-secret-token")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            webhook_client_reject(
                StatusCode::UNAUTHORIZED,
                "missing Telegram webhook secret header",
            )
        })?;
    if !signatures_equal(received, &expected) {
        return Err(webhook_client_reject(
            StatusCode::UNAUTHORIZED,
            "Telegram webhook secret mismatch",
        ));
    }
    Ok(())
}

fn verify_discord_signature(headers: &HeaderMap, body: &[u8]) -> Result<(), (StatusCode, String)> {
    let public_key_hex = match crate::webhook_verify::resolve_webhook_secret("DISCORD_PUBLIC_KEY") {
        Ok(Some(value)) => value,
        Ok(None) => return Ok(()),
        Err(err) => return Err(webhook_server_error(err.to_string())),
    };

    let sig_hex = headers
        .get("x-signature-ed25519")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            webhook_client_reject(StatusCode::UNAUTHORIZED, "missing Discord signature header")
        })?;
    let timestamp = headers
        .get("x-signature-timestamp")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            webhook_client_reject(StatusCode::UNAUTHORIZED, "missing Discord timestamp header")
        })?;

    let pk_bytes = hex::decode(public_key_hex)
        .map_err(|_| webhook_server_error("DISCORD_PUBLIC_KEY must be valid hex"))?;
    let pk_arr: [u8; 32] = pk_bytes
        .try_into()
        .map_err(|_| webhook_server_error("DISCORD_PUBLIC_KEY must be 32 bytes (hex-encoded)"))?;
    let verifying_key = VerifyingKey::from_bytes(&pk_arr)
        .map_err(|_| webhook_server_error("invalid Discord public key from DISCORD_PUBLIC_KEY"))?;

    let sig_bytes = hex::decode(sig_hex).map_err(|_| {
        webhook_client_reject(
            StatusCode::UNAUTHORIZED,
            "invalid Discord signature encoding",
        )
    })?;
    let sig_arr: [u8; 64] = sig_bytes.try_into().map_err(|_| {
        webhook_client_reject(StatusCode::UNAUTHORIZED, "invalid Discord signature length")
    })?;
    let signature = Signature::from_bytes(&sig_arr);

    let message = discord_signing_payload(timestamp, body);

    verifying_key.verify(&message, &signature).map_err(|_| {
        webhook_client_reject(
            StatusCode::UNAUTHORIZED,
            "Discord signature verification failed",
        )
    })
}

fn verify_twilio_signature(
    headers: &HeaderMap,
    uri: &Uri,
    form: &HashMap<String, String>,
) -> Result<(), (StatusCode, String)> {
    let should_validate = crate::webhook_verify::webhook_verification_required()
        || caliche_contracts::credential_env::var("TWILIO_WEBHOOK_VALIDATE_SIGNATURE")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
    if !should_validate {
        return Ok(());
    }

    let auth_token = caliche_contracts::credential_env::var("TWILIO_AUTH_TOKEN").map_err(|_| {
        webhook_server_error("TWILIO_AUTH_TOKEN is required when signature validation is enabled")
    })?;
    let signature = headers
        .get("x-twilio-signature")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            webhook_client_reject(StatusCode::UNAUTHORIZED, "missing Twilio signature header")
        })?;

    let url = inbound_public_url(headers, uri);
    let expected = twilio_expected_signature(&url, form, &auth_token)
        .map_err(|e| webhook_server_error(format!("failed to compute Twilio signature: {e}")))?;

    if !signatures_equal(signature, &expected) {
        return Err(webhook_client_reject(
            StatusCode::UNAUTHORIZED,
            "Twilio signature mismatch",
        ));
    }
    Ok(())
}

fn verify_slack_signature(headers: &HeaderMap, body: &[u8]) -> Result<(), (StatusCode, String)> {
    let signing_secret = match crate::webhook_verify::resolve_webhook_secret("SLACK_SIGNING_SECRET")
    {
        Ok(Some(value)) => value,
        Ok(None) => return Ok(()),
        Err(err) => return Err(webhook_server_error(err.to_string())),
    };

    let timestamp = headers
        .get("x-slack-request-timestamp")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            webhook_client_reject(
                StatusCode::UNAUTHORIZED,
                "missing Slack request timestamp header",
            )
        })?;

    // Replay-window guard (5 minutes), matching Slack guidance. Non-numeric
    // timestamps are rejected rather than skipping the check.
    if !slack_timestamp_is_fresh(timestamp, chrono::Utc::now().timestamp(), 300) {
        return Err(webhook_client_reject(
            StatusCode::UNAUTHORIZED,
            "stale or invalid Slack webhook timestamp",
        ));
    }

    let signature = headers
        .get("x-slack-signature")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            webhook_client_reject(StatusCode::UNAUTHORIZED, "missing Slack signature header")
        })?;

    let expected = slack_signature_v0(&signing_secret, timestamp, body).map_err(|e| {
        webhook_server_error(format!(
            "failed to initialize Slack signature verifier: {e}"
        ))
    })?;

    if !signatures_equal(signature, &expected) {
        return Err(webhook_client_reject(
            StatusCode::UNAUTHORIZED,
            "Slack signature mismatch",
        ));
    }
    Ok(())
}

pub async fn delivery_diagnostics_handler(
    State(adapter): State<SessionsState>,
) -> Result<Json<DeliveryDiagnosticsResponse>, (StatusCode, String)> {
    let dead_letter_count = adapter.gateway_dead_letter_store().list().await.len();
    let runtime = adapter.gateway_runtime_metrics();
    Ok(Json(DeliveryDiagnosticsResponse {
        supported_targets: vec![
            "file".to_string(),
            "webhook".to_string(),
            "channel".to_string(),
            "event_stream".to_string(),
            "callback".to_string(),
        ],
        retry_supported: true,
        dead_letter_queue: "memory_dead_letter_store".to_string(),
        dead_letter_count,
        runtime_metrics: GatewayRuntimeMetricsResponse {
            accepted: runtime.accepted,
            rejected_rate_limited: runtime.rejected_rate_limited,
            rejected_backpressure: runtime.rejected_backpressure,
            duplicate_dropped: runtime.duplicate_dropped,
        },
    }))
}
