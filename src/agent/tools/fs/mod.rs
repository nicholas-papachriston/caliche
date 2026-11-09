use crate::agent::tools::{FsTool, Tool, Toolkit};
use std::sync::Arc;

/// Toolkit that bundles filesystem-related tools
pub struct FileSystemToolkit;

impl Toolkit for FileSystemToolkit {
    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![Arc::new(FsTool::new())]
    }

    fn name(&self) -> &'static str {
        "filesystem"
    }
}
