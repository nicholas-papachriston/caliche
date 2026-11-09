//! Gateway core for multi-channel personal-agent interactions.
//!
//! Channel and webhook contracts for:
//! - ingress normalization from channel adapters
//! - session binding and continuity
//! - routing into agent execution
//! - outbound delivery with retry and dead-letter hooks

pub mod adapters;
pub mod delivery;
pub mod egress;
pub mod registry;
pub mod router;
pub mod routing_rules;
pub mod runtime;
pub mod session_binding;
pub mod types;
pub mod webhook_verify;

pub use adapters::{
    AdapterCapabilities, AdapterTransport, DiscordAdapter, DiscordAdapterConfig, GatewayAdapter,
    GenericWebhookAdapter, SlackAdapter, SlackAdapterConfig, TeamsAdapter, TeamsAdapterConfig,
    TelegramAdapter, TelegramAdapterConfig, TwilioAdapter, TwilioAdapterConfig,
};
pub use delivery::{
    DeliveryAttempt, DeliveryErrorKind, DeliveryOutcome, DeliveryResult, GatewayDeadLetterStore,
    GatewayDelivery, MemoryDeadLetterStore,
};
pub use registry::{AdapterRegistration, GatewayAdapterRegistry};
pub use router::{
    DefaultGatewayRouter, GatewayRouteDecision, GatewayRouter, InMemoryDedupeStore,
    MessageDedupeStore, RouterError,
};
pub use routing_rules::{
    FirstMatchEngine, MatchCondition, NoOpRoutingEngine, RoutingOverrides, RoutingRule,
    RoutingRuleEngine, engine_from_rules, load_engine_from_path, load_rules_from_path,
};
pub use runtime::{
    GatewayAdmission, GatewayRateLimiter, GatewayRuntime, GatewayRuntimeConfig,
    GatewayRuntimeMetrics, GatewayRuntimeMetricsSnapshot,
};
#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
pub use webhook_verify::verify_teams_signature;
pub use webhook_verify::{
    PublicUrlOptions, discord_signing_payload, inbound_public_url, inbound_public_url_with,
    resolve_webhook_secret, signatures_equal, slack_signature_v0, slack_timestamp_is_fresh,
    teams_verification_required, twilio_expected_signature, verify_teams_hmac_bearer,
    webhook_verification_required,
};

pub use egress::{
    DEFAULT_PROVIDER_HOSTS, EgressClass, EgressMode, EgressPolicy, current_policy, mcp_http_policy,
    provider_http_policy, tool_http_policy,
};
pub use session_binding::{
    BindingKey, MemorySessionBindingStore, SessionBinding, SessionBindingStore, SessionFactory,
};
pub use types::{
    ChannelKind, DeliveryPolicy, GatewayEventEnvelope, GatewayIngressEnvelope,
    GatewayOutboundEnvelope, GatewaySessionRef, IngressPayload, OutboundPayload, RouteContext,
};
