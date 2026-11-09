use agent_client_protocol_schema::v1::Implementation;
use anyhow::Result;
use caliche::AgentService;
use caliche_server::acp::{AcpFrameSink, AcpServer, AcpTurnDriver, StdFrameSink, serve_acp};
use caliche_server::adapter::ApiAdapter;
use std::sync::Arc;
use tokio::io::BufReader;
use tokio::sync::RwLock;

use crate::server::build_harness_config_from_cli;

/// ACP stdio agent (JSON-RPC protocol version 1).
///
/// Stdout is newline-delimited JSON-RPC frames only. `session/prompt` does not
/// block the stdin read, so `session/cancel` can arrive while updates are
/// still being written. Tracing stays on the process subscriber (stderr when
/// this process is not a REPL).
pub async fn handle_acp_stdio(
    shared_service: Arc<RwLock<AgentService>>,
    args: &crate::commands::Args,
    cli_config: Option<&crate::config::Config>,
) -> Result<()> {
    let harness_config =
        build_harness_config_from_cli(shared_service.clone(), args, cli_config).await?;
    let adapter = Arc::new(ApiAdapter::new_with_env(harness_config, Some(shared_service)).await);
    let driver: Arc<dyn AcpTurnDriver> = adapter;
    let server = Arc::new(AcpServer::new(
        driver,
        Implementation::new("caliche", env!("CARGO_PKG_VERSION")).title("Caliche"),
    ));
    let frames: Arc<dyn AcpFrameSink> = Arc::new(StdFrameSink::new(std::io::stdout()));
    let stdin = BufReader::new(tokio::io::stdin());
    serve_acp(server, stdin, frames).await
}
