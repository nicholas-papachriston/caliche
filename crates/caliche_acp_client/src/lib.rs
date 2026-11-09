//! ACP v1 client for spawning a peer agent over stdio.
//!
//! The peer is the program with the model (`opencode acp`, `hermes acp`,
//! `omp acp`, or the fixture in this crate's tests). This process is the
//! client: it writes JSON-RPC to the child stdin and reads JSON-RPC from the
//! child stdout. Logs stay off that stdout pipe.
//!
//! # Auth
//!
//! [`reject_peer_auth`] accepts an empty `authMethods` list, or a list whose
//! every method id is exactly `opencode-login` or `hermes-setup`. Any other
//! id stops the client before `session/new`. The client does not call
//! `authenticate`. If the peer later returns `auth_required`, the client stops
//! and names the out-of-band BYO setup. Provider API keys and local-model
//! settings are whatever the operator already put in the child environment.
//! This crate does not run `opencode auth login`, `hermes setup`, or
//! `hermes acp --setup`.
//!
//! # Smoke
//!
//! [`smoke`] drives this client against the in-repo fixture or a real peer
//! (`opencode acp`, `hermes acp`, `hermes-acp`). The binary is `acp-peer-smoke`.
//! Pass operator-supplied keys through [`AcpSpawn::env`], or opt in with
//! [`AcpSpawn::inherit_env`] when those keys are already exported.
//! A list of only `opencode-login` and `hermes-setup` does not abort the turn.
//! This crate answers `session/request_permission`
//! with `cancelled` and answers `fs/*` and `terminal/*` with method-not-found.
//! Serving those v1 client methods, or speaking ACP v2, is a separate change.
//!
//! # Example
//!
//! ```no_run
//! use caliche_acp_client::{AcpClient, AcpSpawn};
//!
//! # async fn run() -> Result<(), caliche_acp_client::AcpClientError> {
//! let client = AcpClient::spawn(
//!     AcpSpawn::new(["opencode", "acp"], "/absolute/workspace")
//!         .env("OPENAI_API_KEY", "operator-supplied"),
//! )?;
//! let _init = client.initialize().await?;
//! let session = client.new_session("/absolute/workspace").await?;
//! let turn = client
//!     .prompt_text(session.session_id.0.as_ref(), "Summarize the repo")
//!     .await?;
//! let _ = turn.stop_reason;
//! # Ok(())
//! # }
//! ```

mod auth;
mod client;
mod error;
mod notes;
mod rpc;
pub mod smoke;
mod stdio;
mod transcript;

pub use auth::reject_peer_auth;
pub use client::{ACP_PROTOCOL_VERSION, AcpClient, AcpSpawn, PromptOutcome};
pub use error::AcpClientError;
