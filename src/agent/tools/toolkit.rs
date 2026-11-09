use crate::agent::tools::Tool;
use std::sync::Arc;

/// A toolkit is a collection of related tools that can be registered together
/// This simplifies agent setup by bundling commonly-used tools
pub trait Toolkit: Send + Sync {
    /// Get all tools in this toolkit
    fn get_tools(&self) -> Vec<Arc<dyn Tool>>;

    /// Get the name/identifier of this toolkit
    fn name(&self) -> &str;
}
