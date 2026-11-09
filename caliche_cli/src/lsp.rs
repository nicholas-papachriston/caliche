//! CLI helpers for the core Language Server Protocol (LSP) integration.

use anyhow::Result;
use tokio::process::Command;

pub use caliche::lsp::{LspConfig, LspManager};

/// Format LSP status as a string (for REPL popup or other use).
pub fn format_lsp_status(manager: &LspManager) -> String {
    let mut out = String::new();
    out.push_str("\nEnabled: ");
    out.push_str(if manager.is_enabled() { "yes" } else { "no" });
    out.push_str("\n\nSupported Languages:\n");
    for language in manager.supported_languages() {
        let available = if manager.is_server_available(&language) {
            "[AVAILABLE]"
        } else {
            "[NOT FOUND]"
        };
        if let Some(config) = manager.registry().get_server(&language) {
            out.push_str(&format!(
                "  {:<12} {} - {}\n",
                language, available, config.command
            ));
        }
    }
    out.push('\n');
    out
}

/// Print LSP status information to stdout.
pub fn print_lsp_status(manager: &LspManager) {
    print!("{}", format_lsp_status(manager));
}

/// CLI command to install a language server
pub async fn install_language_server(language: &str) -> Result<()> {
    let install_cmd = match language {
        "rust" => Some(("rustup", vec!["component", "add", "rust-analyzer"])),
        "typescript" => Some((
            "npm",
            vec!["install", "-g", "typescript-language-server", "typescript"],
        )),
        "python" => Some(("pip", vec!["install", "pyright"])),
        "go" => Some(("go", vec!["install", "golang.org/x/tools/gopls@latest"])),
        _ => None,
    };

    match install_cmd {
        Some((cmd, args)) => {
            println!("Installing {} language server...", language);
            println!("Running: {} {}", cmd, args.join(" "));

            let status = Command::new(cmd).args(&args).status().await?;

            if status.success() {
                println!("Installation complete!");
            } else {
                anyhow::bail!("Installation failed");
            }
            Ok(())
        }
        None => {
            anyhow::bail!(
                "No automatic installation available for {}. Please install manually.",
                language
            );
        }
    }
}
