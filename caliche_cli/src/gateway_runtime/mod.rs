//! Gateway runtime orchestration: local daemon lifecycle, tunnel forwarding, health checks, and Twilio ingress setup.

mod api;
mod doctor;
mod forward;
mod lifecycle;
mod logs;
mod process;
mod registry;
mod twilio;
mod types;

pub use types::{
    ForwardProviderKind, ForwardSessionRecord, ForwardStatus, GatewayDoctorRecord,
    GatewayLlmConfigStatus, GatewayLogsSnapshot, GatewayRouteInfo, GatewayRuntimeEntry,
    GatewayRuntimeRole, GatewayRuntimeStatus, TwilioWebhookStatus, TwilioWebhookSyncResult,
};

pub use doctor::doctor_gateway;
pub use forward::{
    cloudflare_tunnel_install_instructions, list_gateway_forwardings, restart_gateway_forwarding,
    start_gateway_forwarding, status_gateway_forwarding, stop_gateway_forwarding,
};
pub use lifecycle::{
    clear_gateway_llm_config, gateway_llm_config_status, list_gateway_routes,
    list_gateway_runtimes, pin_gateway_route, read_gateway_logs, read_gateway_logs_fallback,
    restart_gateway_runtime, set_gateway_llm_config, start_gateway_runtime, status_gateway_runtime,
    stop_all_gateway_runtimes, stop_gateway_runtime, unpin_gateway_route,
};
pub use twilio::{sync_twilio_webhook, twilio_webhook_status};
