#[cfg(feature = "tools-sql")]
use crate::agent::tools::SqlExecutionTool;
#[cfg(feature = "tools-rag")]
use crate::agent::tools::SqlGenerationTool;
use crate::agent::tools::{Tool, Toolkit};
use std::sync::Arc;

/// Toolkit that bundles SQL-related tools
pub struct SqlToolkit {
    #[cfg(feature = "tools-sql")]
    execution_tool: Option<Arc<SqlExecutionTool>>,
    #[cfg(feature = "tools-rag")]
    generation_tool: Option<Arc<SqlGenerationTool>>,
}

impl SqlToolkit {
    /// Create a new SQL toolkit
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "tools-sql")]
            execution_tool: None,
            #[cfg(feature = "tools-rag")]
            generation_tool: None,
        }
    }

    /// Add SQL execution tool
    #[cfg(feature = "tools-sql")]
    #[must_use]
    pub fn with_execution_tool(mut self, tool: Arc<SqlExecutionTool>) -> Self {
        self.execution_tool = Some(tool);
        self
    }

    /// Add SQL generation tool
    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn with_generation_tool(mut self, tool: Arc<SqlGenerationTool>) -> Self {
        self.generation_tool = Some(tool);
        self
    }
}

impl Default for SqlToolkit {
    fn default() -> Self {
        Self::new()
    }
}

impl Toolkit for SqlToolkit {
    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        #[cfg(any(feature = "tools-sql", feature = "tools-rag"))]
        {
            let mut tools = Vec::new();
            #[cfg(feature = "tools-sql")]
            if let Some(ref tool) = self.execution_tool {
                tools.push(Arc::clone(tool) as Arc<dyn Tool>);
            }
            #[cfg(feature = "tools-rag")]
            if let Some(ref tool) = self.generation_tool {
                tools.push(Arc::clone(tool) as Arc<dyn Tool>);
            }
            tools
        }
        #[cfg(not(any(feature = "tools-sql", feature = "tools-rag")))]
        {
            Vec::new()
        }
    }

    fn name(&self) -> &'static str {
        "sql"
    }
}
