//! Welcome message and project display for interactive mode.

use std::path::Path;

/// Print the welcome message at REPL startup.
pub fn print_welcome_message(
    provider: &str,
    model: &str,
    token_budget: &crate::cost::TokenBudget,
    project_type: &str,
    project_config_path: Option<&Path>,
    tool_count: usize,
    custom_command_count: usize,
) {
    println!();
    println!("\x1b[1;36m=== Caliche ===\x1b[0m");
    println!();

    println!(
        "\x1b[1mProvider:\x1b[0m {} \x1b[2m|\x1b[0m \x1b[1mModel:\x1b[0m {}",
        provider, model
    );
    println!(
        "\x1b[1mContext:\x1b[0m {} \x1b[2m|\x1b[0m \x1b[1mProject:\x1b[0m {}",
        crate::cost::format_tokens(token_budget.max_tokens),
        project_type
    );

    if let Some(path) = project_config_path {
        println!(
            "\x1b[32mConfig:\x1b[0m {} loaded",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    println!();
    let mut capabilities = Vec::new();
    capabilities.push(format!("{} tools", tool_count));
    if custom_command_count > 0 {
        capabilities.push(format!("{} custom commands", custom_command_count));
    }
    println!("\x1b[1mCapabilities:\x1b[0m {}", capabilities.join(", "));

    println!();
    println!("\x1b[2mStart here:\x1b[0m");
    println!("  Ask a task directly, or run \x1b[1m/help\x1b[0m for commands");
    println!(
        "  Attach context with \x1b[1m@path\x1b[0m, \x1b[1m@diff\x1b[0m, or \x1b[1m@status\x1b[0m"
    );
    println!();
    println!("\x1b[2mQuick keys:\x1b[0m");
    println!(
        "  \x1b[1m/\x1b[0m commands  \x1b[1m@\x1b[0m context  \x1b[1mTab\x1b[0m complete  \x1b[1mexit\x1b[0m quit"
    );
    println!(
        "  \x1b[1m/sessions\x1b[0m list  \x1b[1m/resume\x1b[0m continue  \x1b[1mCtrl+C\x1b[0m cancel"
    );
    println!(
        "  \x1b[1m/mode\x1b[0m status  \x1b[1m/plan\x1b[0m read-only  \x1b[1m/agent\x1b[0m full-exec"
    );
    println!();
}
