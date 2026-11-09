//! LSP (Language Server Protocol) client integration.
//!
//! Provides real-time diagnostics, type information, and symbol search
//! by connecting to language servers (rust-analyzer, pyright, etc.).

pub mod client;
pub mod config;
pub mod diagnostics;
pub mod manager;
pub mod registry;
pub mod tool_inject;
pub mod types;

pub use config::{LspConfig, ServerConfig};
pub use diagnostics::DiagnosticFormatter;
pub use manager::LspManager;
pub use registry::LspServerRegistry;
pub use tool_inject::{
    collect_diagnostics_for_path, inject_path_diagnostics, inject_paths_diagnostics,
};
pub use types::{Diagnostic, DiagnosticSeverity, Position, Range, SymbolInfo, SymbolKind};

/// Re-export commonly used types.
pub mod prelude {
    pub use crate::lsp::{
        config::{LspConfig, ServerConfig},
        diagnostics::DiagnosticFormatter,
        manager::LspManager,
        registry::LspServerRegistry,
        types::{Diagnostic, DiagnosticSeverity, Position, Range, SymbolInfo, SymbolKind},
    };
}
