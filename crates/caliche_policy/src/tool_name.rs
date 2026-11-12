//! The name a tool call runs under.
//!
//! The registry trims a requested name and maps a few aliases to the tool
//! they run. Every gate decides on that same name, so a call cannot reach a
//! tool under a spelling the gate does not check.

/// Requested names that run another tool, matched without regard to case.
pub const TOOL_ALIASES: [(&str, &str); 4] = [
    ("shell", "bash"),
    ("web", "http"),
    ("memory", "workspace"),
    ("session", "sessions_send"),
];

/// The tool a requested name runs: surrounding space trimmed and an alias
/// from [`TOOL_ALIASES`] replaced by its tool.
///
/// # Examples
///
/// ```
/// use caliche_policy::tool_name_as_run;
///
/// assert_eq!(tool_name_as_run(" fs "), "fs");
/// assert_eq!(tool_name_as_run("Memory"), "workspace");
/// assert_eq!(tool_name_as_run("git_log"), "git_log");
/// ```
#[must_use]
pub fn tool_name_as_run(name: &str) -> &str {
    let trimmed = name.trim();
    TOOL_ALIASES
        .iter()
        .find(|(alias, _)| trimmed.eq_ignore_ascii_case(alias))
        .map_or(trimmed, |(_, tool)| tool)
}
