mod llms_probe;
mod tool;

use crate::agent::tools::{Tool, Toolkit};
use std::sync::Arc;

pub use llms_probe::{
    DomainProbeCache, ProbeResult, ensure_domain_probed, extract_domain, new_probe_cache,
};
pub use tool::HttpTool;

/// Toolkit that bundles HTTP-related tools
pub struct HttpToolkit {
    tool: Arc<HttpTool>,
}

impl HttpToolkit {
    /// Create a new HTTP toolkit
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool: Arc::new(HttpTool::new()),
        }
    }

    /// Create with credentials
    #[must_use]
    pub fn with_credentials(credentials: crate::agent::tools::credentials::ToolCredential) -> Self {
        Self {
            tool: Arc::new(HttpTool::with_credentials(credentials)),
        }
    }

    /// Create with base URL
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            tool: Arc::new(HttpTool::new().with_base_url(base_url)),
        }
    }
}

impl Default for HttpToolkit {
    fn default() -> Self {
        Self::new()
    }
}

impl Toolkit for HttpToolkit {
    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::clone(&self.tool) as Arc<dyn Tool>]
    }

    fn name(&self) -> &'static str {
        "http"
    }
}
