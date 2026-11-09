use agent_client_protocol_schema::v1::McpServer;
use anyhow::Context;

pub fn validate_mcp_servers(servers: &[McpServer]) -> anyhow::Result<()> {
    for server in servers {
        match server {
            McpServer::Stdio(stdio) => {
                if !stdio.command.is_absolute() {
                    anyhow::bail!(
                        "mcp stdio command must be an absolute path: {}",
                        stdio.command.display()
                    );
                }
            }
            McpServer::Http(http) => validate_mcp_url(&http.url)?,
            McpServer::Sse(sse) => validate_mcp_url(&sse.url)?,
            _ => anyhow::bail!("unsupported MCP transport"),
        }
    }
    Ok(())
}

fn validate_mcp_url(url: &str) -> anyhow::Result<()> {
    let parsed = url::Url::parse(url).with_context(|| format!("invalid MCP URL: {url}"))?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        anyhow::bail!("MCP URL must use http or https: {url}");
    }
    caliche::gateway::egress::current_policy()?
        .validate_static(caliche::gateway::egress::EgressClass::Mcp, url)
        .with_context(|| format!("MCP URL denied by egress policy: {url}"))?;
    Ok(())
}
