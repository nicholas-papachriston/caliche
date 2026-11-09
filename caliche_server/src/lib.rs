#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "HTTP service surfaces request/operator diagnostics on stdout/stderr"
)]

pub mod boot_provider;
mod bootstrap;
pub mod durable_settings;
pub mod env_file;
mod jwks_cache;
mod provider_cache;
mod redact;
mod router;
pub mod shutdown;
mod state;

pub mod paths;

pub mod acp;
pub mod adapter;
pub mod handlers;
pub mod middleware;
pub mod models;
pub mod rpc;
pub mod security;
pub mod transport;
pub mod webhook_verify;
pub mod workflow_executor;

pub use security::SecurityPolicy;

pub use handlers::endpoints::EndpointManager;
pub use handlers::harnesses::HarnessManager;

pub use adapter::{ApiAdapter, HarnessFactory, SessionId, SessionInfo, SessionState};

#[cfg(feature = "router")]
pub use provider_cache::select_provider;
pub use provider_cache::{get_or_create_provider, get_or_create_tool_registry};

pub use router::{create_router, create_router_with_shutdown};

pub use state::{
    AgentRouterState, AppState, CollectionMetadata, ProviderCacheKey, RouterOptions,
    ToolRegistryCacheKey, WorkflowExecution,
};

#[cfg(feature = "lambda")]
pub mod lambda;

#[cfg(feature = "lambda")]
pub use lambda::{create_lambda_handler, lambda_handler, run_lambda_server};
#[cfg(feature = "stream-lambda")]
pub use lambda::{lambda_streaming_handler, run_lambda_streaming_server};

#[cfg(feature = "embedded-ui")]
pub mod embedded_ui;
