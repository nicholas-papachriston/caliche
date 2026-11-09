pub mod agent_stream;
pub mod chat;
mod chat_contracts;
mod chat_history;
mod chat_service;
pub mod chat_stream;
pub mod context;
pub mod estimate;
pub mod files;
pub mod gateway;
pub mod gateway_validation;
pub mod health;
pub mod mcp_config;
pub mod mcp_policy;
pub mod memory;
pub mod metrics;
pub mod models;
pub mod multipart_chat;
pub mod multipart_limits;
pub mod prometheus_metrics;
mod session_sse;
pub mod sessions;
pub mod settings;
pub mod storage;
pub mod tools;
mod turn_failure;

// Advanced feature handlers
#[cfg(feature = "reasoning-patterns")]
pub mod reasoning;

#[cfg(feature = "chains")]
pub mod chains;

#[cfg(any(feature = "document-loaders", feature = "tools-rag"))]
pub mod documents;

pub mod endpoints;
pub mod harnesses;
pub mod orchestration;
pub mod plugins;
pub mod policy;

#[cfg(feature = "templates")]
pub mod templates;

#[cfg(feature = "mcp")]
pub mod mcp;
