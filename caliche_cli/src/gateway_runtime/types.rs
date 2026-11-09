use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRuntimeEntry {
    #[serde(default)]
    pub agent_id: String,
    pub cwd: String,
    pub pid: u32,
    pub port: u16,
    #[serde(default = "default_gateway_role")]
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub main_gateway_url: Option<String>,
    pub started_at: String,
    pub log_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forwarding: Option<ForwardSessionRecord>,
}

fn default_gateway_role() -> String {
    GatewayRuntimeRole::Main.as_str().to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct GatewayRuntimeRegistry {
    pub(super) entries: Vec<GatewayRuntimeEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GatewayRuntimeStatus {
    pub entry: GatewayRuntimeEntry,
    pub running: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GatewayRuntimeRole {
    Main,
    Worker,
}

impl GatewayRuntimeRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Worker => "worker",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardSessionRecord {
    pub provider: String,
    pub tunnel_pid: u32,
    pub started_at: String,
    pub local_port: u16,
    pub log_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForwardStatus {
    pub cwd: String,
    pub provider: ForwardProviderKind,
    pub tunnel_pid: u32,
    pub local_port: u16,
    pub public_url: Option<String>,
    pub log_path: String,
    pub running: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRouteInfo {
    pub route_key: String,
    pub target_agent_id: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwilioWebhookSyncResult {
    pub phone_number: String,
    pub incoming_phone_number_sid: String,
    pub sms_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwilioWebhookStatus {
    pub phone_number: String,
    pub incoming_phone_number_sid: String,
    pub sms_url: Option<String>,
    pub sms_method: Option<String>,
    pub expected_sms_url: Option<String>,
    pub matches_expected: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GatewayLogsSnapshot {
    pub cwd: String,
    pub runtime_log_path: String,
    pub runtime_log_tail: String,
    pub forwarding_log_path: Option<String>,
    pub forwarding_log_tail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GatewayDoctorRecord {
    pub cwd: String,
    pub role: String,
    pub port: u16,
    pub recorded_pid: u32,
    pub recorded_running: bool,
    pub listener_pid: Option<u32>,
    pub listener_matches_recorded: Option<bool>,
    pub forwarding_pid: Option<u32>,
    pub forwarding_running: Option<bool>,
    pub forwarding_url: Option<String>,
    pub runtime_log_path: String,
    pub forwarding_log_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GatewayLlmConfigStatus {
    pub cwd: String,
    pub configured_provider: Option<String>,
    pub configured_model: Option<String>,
    pub effective_provider: Option<String>,
    pub effective_model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ForwardProviderKind {
    CloudflareQuick,
}

impl ForwardProviderKind {
    pub fn parse(input: &str) -> Option<Self> {
        match input.to_ascii_lowercase().as_str() {
            "cloudflare" | "cloudflare-quick" | "quick" => Some(Self::CloudflareQuick),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CloudflareQuick => "cloudflare",
        }
    }
}
