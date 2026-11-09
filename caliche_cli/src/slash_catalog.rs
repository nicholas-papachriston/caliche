//! CLI slash command catalog: contract-backed specs plus CLI-only extensions.

use caliche_contracts::slash::{
    SlashCommandSpec, SlashSurface, all_slash_commands,
    slash_command_groups as contract_slash_command_groups,
};
use std::collections::HashSet;
use std::sync::LazyLock;

/// Slash commands that need a Cargo feature, and whether this build has it.
/// A command left out is not offered (catalog, `/help`, completion); typing
/// it still reaches its handler, which says how to build it in.
pub(crate) const FEATURE_GATED_COMMANDS: &[(&str, bool)] = &[
    ("/local-model", cfg!(feature = "local-inference-base")),
    ("/local-models", cfg!(feature = "local-inference-base")),
];

/// True unless `command` needs a feature this build left out.
#[must_use]
pub fn command_in_this_build(command: &str) -> bool {
    FEATURE_GATED_COMMANDS
        .iter()
        .all(|(gated, built)| *gated != command || *built)
}

/// CLI-only slash commands (and completion variants) not yet in `SLASH_COMMAND_SPECS`.
const CLI_EXTRA_SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/save", "Save conversation to file"),
    ("/load", "Load conversation from file"),
    (
        "/resume",
        "Resume session by id/prefix, /resume last, or show recent sessions",
    ),
    ("/status", "Show session status"),
    ("/context", "Show current context"),
    ("/track", "Track a file in context"),
    ("/add", "Add files to context by glob pattern"),
    ("/recent", "Show recently modified files"),
    ("/related", "Find related files (imports, tests)"),
    (
        "/models",
        "List available models (optionally for a provider)",
    ),
    (
        "/subagent-provider",
        "Set provider for subagents (or 'clear'); persists across sessions",
    ),
    (
        "/subagent-model",
        "Set model for subagents (or 'clear'); persists across sessions",
    ),
    ("/embeddings", "List available embedding providers"),
    (
        "/embedding",
        "Set embedding provider/model (e.g. openai or openai/text-embedding-3-small)",
    ),
    ("/cost", "Show session cost breakdown"),
    ("/budget", "Show token budget status"),
    ("/stats", "Show usage statistics for current session"),
    ("/skill", "Apply a skill to the agent"),
    (
        "/diff",
        "Session diff overview (agent mode) or unified diff: /diff or /diff <path>",
    ),
    (
        "/export",
        "Export conversation to a file; add --include-thinking to keep chain-of-thought",
    ),
    ("/compress", "Compress conversation history to save context"),
    (
        "/autocompact",
        "Configure auto-compact (on|off|<threshold>)",
    ),
    (
        "/memory",
        "Show or switch memory strategy (buffer|summary|entity|vector|enhanced)",
    ),
    ("/remember", "Store a fact in knowledge store"),
    ("/recall", "Search knowledge store for relevant information"),
    ("/knowledge", "Show knowledge store summary"),
    ("/forget", "Remove a knowledge entry by ID"),
    ("/changes", "Show files modified during session"),
    ("/pr", "Prepare a pull request"),
    ("/commit", "Generate intelligent commit message"),
    ("/review", "Review staged changes"),
    ("/lsp", "Show LSP status and available servers"),
    (
        "/compact",
        "Set quiet output (answer-focused; toggle back to normal)",
    ),
    (
        "/verbose",
        "Set output verbosity: /verbose [quiet|normal|thinking|verbose|debug]",
    ),
    (
        "/verbosity",
        "Alias for /verbose [quiet|normal|thinking|verbose|debug]",
    ),
    (
        "/thinking",
        "Show thought chains (same as /verbose thinking)",
    ),
    ("/editor", "Open external editor for input"),
    ("/copy", "Copy last agent response to clipboard"),
    (
        "/attach",
        "Attach image/audio/video/document to next message: /attach <file> [file...]",
    ),
    ("/commands", "List custom user commands"),
    ("/creds", "Show credentials available in this directory"),
    ("/creds setup", "Run interactive credential setup wizard"),
    ("/gateway", "Manage gateway runtime, forwarding, and routes"),
    ("/gateway setup", "Run gateway-only credential setup wizard"),
    (
        "/gateway setup --forward [--provider cloudflare]",
        "Run gateway setup and start forwarding",
    ),
    (
        "/gateway start [--port N]",
        "Start background gateway runtime for this directory",
    ),
    ("/gateway stop", "Stop background gateway runtime"),
    (
        "/gateway stop --all",
        "Stop all registered gateway runtimes and forwarders",
    ),
    (
        "/gateway restart [--port N]",
        "Restart background gateway runtime",
    ),
    ("/gateway status", "Show background gateway runtime status"),
    ("/gateway list", "List all background gateway runtimes"),
    (
        "/gateway forward start [--provider cloudflare] [--port N]",
        "Start Cloudflare quick tunnel forwarding",
    ),
    ("/gateway forward stop", "Stop gateway forwarding tunnel"),
    (
        "/gateway forward status",
        "Show gateway forwarding tunnel status",
    ),
    ("/gateway forward list", "List gateway forwarding sessions"),
    (
        "/gateway route list",
        "List pinned gateway route assignments",
    ),
    (
        "/gateway route pin <route_key> --agent <agent_id>",
        "Pin a route key to a specific worker agent",
    ),
    (
        "/gateway route unpin <route_key>",
        "Remove a pinned route assignment",
    ),
    (
        "/gateway llm status [--dir <path>]",
        "Show gateway-local provider/model configuration",
    ),
    (
        "/gateway llm set [--provider <provider>] [--model <model>]",
        "Set provider/model for this gateway runtime only",
    ),
    (
        "/gateway llm clear [--dir <path>]",
        "Clear gateway-local provider/model overrides",
    ),
    (
        "/gateway twilio webhook sync [--url <public_base_url>]",
        "Sync Twilio number webhook to /api/gateway/ingress/twilio",
    ),
    (
        "/gateway twilio webhook status [--url <public_base_url>]",
        "Show current Twilio webhook config and expected gateway URL",
    ),
    (
        "/gateway logs [--lines N] [--forward|--no-forward] [--dir <path>]",
        "Show recent gateway runtime and forwarding logs",
    ),
    (
        "/gateway doctor [--all] [--dir <path>]",
        "Diagnose gateway/runtime PID, listener, and forwarding health",
    ),
    (
        "/creds providers setup",
        "Run provider-only credential setup wizard",
    ),
    (
        "/creds set <provider> <key>",
        "Store an API key in the keychain",
    ),
    (
        "/creds import",
        "Import credentials from .env file into keychain",
    ),
    ("/creds list", "List all credential scopes"),
    (
        "/creds link <dir>",
        "Use credentials from <dir> in this directory",
    ),
    ("/exit", "Exit the session"),
    ("/quit", "Exit the session"),
];

/// Base commands shown by default in `/help`.
pub const QUICK_HELP_COMMANDS: &[&str] = &[
    "/help",
    "/welcome",
    "/mode",
    "/undo",
    "/redo",
    "/share",
    "/unshare",
    "/diff",
    "/tools",
    "/doctor",
    "/init",
    "/sessions",
    "/resume",
    "/suspend",
    "/save",
    "/load",
    "/export",
    "/compact",
    "/provider",
    "/model",
    "/status",
    "/exit",
];

/// Extra help groups for CLI-only commands (merged with contract `slash_command_groups`).
const CLI_EXTRA_SLASH_COMMAND_GROUPS: &[(&str, &[&str])] = &[
    (
        "Session",
        &[
            "/save", "/load", "/resume", "/suspend", "/status", "/context", "/track", "/add",
            "/recent", "/related",
        ],
    ),
    (
        "Provider Commands",
        &[
            "/models",
            "/subagent-provider",
            "/subagent-model",
            "/embeddings",
            "/embedding",
        ],
    ),
    ("Cost & stats", &["/cost", "/budget", "/stats"]),
    ("Export & diff", &["/skill", "/diff", "/export"]),
    (
        "Context Management",
        &["/compress", "/autocompact", "/memory", "/attach"],
    ),
    (
        "Knowledge",
        &["/remember", "/recall", "/knowledge", "/forget"],
    ),
    ("Files", &["/changes"]),
    ("Git", &["/pr", "/commit", "/review"]),
    ("LSP", &["/lsp"]),
    (
        "Display",
        &[
            "/compact",
            "/verbose",
            "/verbosity",
            "/thinking",
            "/editor",
            "/copy",
        ],
    ),
    ("Custom", &["/commands"]),
    ("Credentials", &["/creds", "/gateway"]),
    ("Exit", &["/exit", "/quit"]),
];

static CLI_SLASH_COMMANDS: LazyLock<Vec<(&'static str, &'static str)>> =
    LazyLock::new(build_slash_commands);

static CLI_SLASH_COMMAND_GROUPS: LazyLock<Vec<(&'static str, Vec<&'static str>)>> =
    LazyLock::new(build_slash_command_groups);

fn push_command(
    out: &mut Vec<(&'static str, &'static str)>,
    seen: &mut HashSet<&'static str>,
    name: &'static str,
    description: &'static str,
) {
    if command_in_this_build(name) && seen.insert(name) {
        out.push((name, description));
    }
}

fn push_spec(
    out: &mut Vec<(&'static str, &'static str)>,
    seen: &mut HashSet<&'static str>,
    spec: &'static SlashCommandSpec,
) {
    push_command(out, seen, spec.primary_name, spec.description);
    for alias in spec.aliases {
        push_command(out, seen, alias, spec.description);
    }
}

fn build_slash_commands() -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for spec in all_slash_commands(SlashSurface::Cli) {
        push_spec(&mut out, &mut seen, spec);
    }
    for &(name, desc) in CLI_EXTRA_SLASH_COMMANDS {
        push_command(&mut out, &mut seen, name, desc);
    }
    out
}

fn merge_group(
    groups: &mut Vec<(&'static str, Vec<&'static str>)>,
    group: &'static str,
    commands: &[&'static str],
) {
    if let Some((_, existing)) = groups.iter_mut().find(|(name, _)| *name == group) {
        for cmd in commands {
            if !existing.contains(cmd) {
                existing.push(cmd);
            }
        }
    } else {
        groups.push((group, commands.to_vec()));
    }
}

fn build_slash_command_groups() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut groups: Vec<(&'static str, Vec<&'static str>)> =
        contract_slash_command_groups(SlashSurface::Cli)
            .into_iter()
            .map(|(group, specs)| {
                let names = specs
                    .iter()
                    .map(|spec| spec.primary_name)
                    .filter(|name| command_in_this_build(name))
                    .collect();
                (group, names)
            })
            .collect();

    for &(group, commands) in CLI_EXTRA_SLASH_COMMAND_GROUPS {
        merge_group(&mut groups, group, commands);
    }

    groups
}

/// All CLI slash commands (contract specs + CLI extensions), including aliases for completion.
#[must_use]
pub fn slash_commands() -> &'static [(&'static str, &'static str)] {
    CLI_SLASH_COMMANDS.as_slice()
}

/// Help rendering groups in display order.
#[must_use]
pub fn slash_command_groups() -> &'static [(&'static str, Vec<&'static str>)] {
    CLI_SLASH_COMMAND_GROUPS.as_slice()
}

/// Resolve a help group name for a slash command token (primary name only).
#[must_use]
pub fn slash_command_group_for(command: &str) -> &'static str {
    let key = command.split_whitespace().next().unwrap_or(command);
    slash_command_groups()
        .iter()
        .find(|(_, cmds)| cmds.iter().any(|cmd| *cmd == key))
        .map(|(name, _)| *name)
        .unwrap_or("Other")
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche_contracts::slash::find_slash_command;

    #[test]
    fn catalog_includes_all_contract_cli_commands() {
        let built = all_slash_commands(SlashSurface::Cli)
            .into_iter()
            .filter(|spec| command_in_this_build(spec.primary_name));
        for spec in built {
            assert!(
                slash_commands()
                    .iter()
                    .any(|(name, _)| *name == spec.primary_name),
                "missing primary {}",
                spec.primary_name
            );
            for alias in spec.aliases {
                assert!(
                    slash_commands().iter().any(|(name, _)| *name == *alias),
                    "missing alias {alias}"
                );
            }
        }
    }

    #[test]
    fn catalog_includes_cli_only_status() {
        assert!(slash_commands().iter().any(|(name, _)| *name == "/status"));
    }

    #[test]
    fn find_slash_command_matches_catalog_primaries() {
        for spec in all_slash_commands(SlashSurface::Cli) {
            assert!(find_slash_command(spec.primary_name, SlashSurface::Cli).is_some());
        }
    }
}
