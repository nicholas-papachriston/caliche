#[cfg(feature = "tools-document")]
use crate::agent::tools::ProcessDocumentTool;
use crate::agent::tools::{DataManipulationTool, Tool, Toolkit};
use std::sync::Arc;

/// Toolkit that bundles data manipulation and document processing tools
pub struct DataToolkit;

impl Toolkit for DataToolkit {
    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        #[cfg(feature = "tools-document")]
        {
            return vec![
                Arc::new(DataManipulationTool),
                Arc::new(ProcessDocumentTool),
            ];
        }
        #[cfg(not(feature = "tools-document"))]
        {
            vec![Arc::new(DataManipulationTool)]
        }
    }

    fn name(&self) -> &'static str {
        "data"
    }
}
