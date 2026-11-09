//! Grouped help, lists, and token popups.

use super::text::render_wrapped_lines;

pub(super) fn render_help(renderer: &mut super::ReplRenderer, commands: &[(String, String)]) {
    let mut by_group: std::collections::HashMap<&str, Vec<(String, String)>> =
        std::collections::HashMap::new();
    for (cmd, desc) in commands {
        let key = cmd
            .split_whitespace()
            .next()
            .unwrap_or_else(|| cmd.as_str());
        let group = crate::slash_catalog::slash_command_group_for(key);
        by_group
            .entry(group)
            .or_default()
            .push((cmd.clone(), desc.clone()));
    }
    let total_cmds: usize = by_group.values().map(|entries| entries.len()).sum();
    renderer
        .output
        .print("Available Commands (Tab to complete):");
    for (group_name, _) in crate::slash_catalog::slash_command_groups() {
        if let Some(entries) = by_group.get(*group_name) {
            renderer
                .output
                .print(&format!("\n  \x1b[1m{group_name}\x1b[0m"));
            for (cmd, desc) in entries {
                renderer.output.print(&format!("    {cmd:<24} {desc}"));
            }
        }
    }
    if let Some(entries) = by_group.get("Other") {
        renderer.output.print("\n  \x1b[1mOther\x1b[0m");
        for (cmd, desc) in entries {
            renderer.output.print(&format!("    {cmd:<24} {desc}"));
        }
    }
    if total_cmds <= 12 {
        renderer
            .output
            .print("\n  \x1b[2mType /help all for full command list.\x1b[0m");
    }
    renderer
        .output
        .print("\n  \x1b[2mCtrl+C to cancel in-flight request.\x1b[0m");
    renderer.output.print("");
}

pub(super) fn render_list(
    renderer: &mut super::ReplRenderer,
    title: &str,
    items: &[String],
    descriptions: &[Option<String>],
) {
    renderer.output.print(&format!("\n=== {title} ==="));
    renderer.output.print(&format!("{} item(s)", items.len()));
    let term_width = crate::repl::terminal::terminal_size()
        .map(|(width, _)| width as usize)
        .unwrap_or(100);
    let desc_width = term_width.saturating_sub(8).clamp(40, 120);
    for (index, item) in items.iter().enumerate() {
        renderer.output.print(&format!("  • {item}"));
        if let Some(desc) = descriptions.get(index).and_then(Option::as_ref) {
            for line in render_wrapped_lines(desc, desc_width) {
                renderer.output.print(&format!("      {line}"));
            }
        }
    }
    renderer.output.print("");
}

pub(super) fn render_token_usage_popup(
    renderer: &mut super::ReplRenderer,
    session_tokens: u64,
    context_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
    session_prompt_tokens: u64,
) {
    use crate::cost::format_tokens;
    let mut body = format!(
        "Token Usage\n===========\n\nSession Tokens: {}\nContext Tokens: {}",
        format_tokens(session_tokens),
        format_tokens(context_tokens)
    );
    if cache_read_tokens > 0 || cache_creation_tokens > 0 || session_prompt_tokens > 0 {
        body.push_str(&format!(
            "\nPrompt tokens: {}\nPrompt cache read: {}\nPrompt cache write: {}",
            format_tokens(session_prompt_tokens),
            format_tokens(cache_read_tokens),
            format_tokens(cache_creation_tokens)
        ));
        if session_prompt_tokens > 0 {
            let hit_pct = (cache_read_tokens as f64 / session_prompt_tokens as f64) * 100.0;
            body.push_str(&format!("\nPrompt cache hit: {hit_pct:.1}%"));
        }
    }
    renderer.output.print(&body);
}
