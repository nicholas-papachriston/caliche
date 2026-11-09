//! Shadow comparison. Logs a diff and does not change the live effect.
//!
//! Enable from the engine with `CALICHE_POLICY_SHADOW=1`. The hook path
//! keeps its own allow/deny. This module only evaluates the shadow snapshot
//! (floor + workstation `standard` preset, human waiter present) and logs when
//! the two effects differ. Arguments are not logged.

use crate::action::{Action, FsOp, HttpMethod, ResolvedPath, ToolClass, ToolFacts};
use crate::effect::Effect;
use crate::policy::{CompiledPolicy, PolicyBuilder};
use crate::posture::ShellDialect;
use crate::principal::{ApprovalWaiter, HumanChannel, Principal};
use crate::reason::ReasonCode;
use crate::request::{Request, SessionScope};
use serde_json::Value;
use std::sync::OnceLock;

/// Environment variable that turns shadow comparison on.
pub const SHADOW_ENV: &str = "CALICHE_POLICY_SHADOW";

/// True when `raw` is `1`, `true`, `yes`, or `on` (case-insensitive).
#[must_use]
pub fn shadow_enabled_value(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Live effect versus the shadow decision.
///
/// Callers keep using [`Self::live`]. [`Self::shadow`] is diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowComparison {
    /// Effect the existing hook path already chose.
    pub live: Effect,
    /// Effect [`CompiledPolicy::decide`](crate::CompiledPolicy::decide) would choose.
    pub shadow: Effect,
    /// Whether the two effects differ.
    pub differs: bool,
    /// Reason code from the shadow decision.
    pub shadow_reason: ReasonCode,
}

/// Compare without applying the shadow effect.
#[must_use]
pub fn compare_shadow(live: Effect, decision: &crate::Decision) -> ShadowComparison {
    let shadow = decision.effect;
    ShadowComparison {
        live,
        shadow,
        differs: live != shadow,
        shadow_reason: decision.reason.code(),
    }
}

/// Evaluate the shadow snapshot for one action.
///
/// [`Effect::Deny`](crate::Effect::Deny) is a hard floor. [`Effect::Ask`](crate::Effect::Ask)
/// stays ask when a human waiter is present.
#[must_use]
pub fn evaluate_action(action: Action, args: &Value) -> crate::Decision {
    let principal = shadow_principal();
    let session = shadow_session();
    let request = Request::new(
        &principal,
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        action,
        args,
    );
    shadow_policy().decide(&request)
}

/// Evaluate the shadow snapshot for one tool call.
#[must_use]
pub fn evaluate_shadow(tool: &str, args: &Value) -> crate::Decision {
    evaluate_action(shadow_action(tool, args), args)
}

/// Live tool gate.
///
/// The deployment default is [`crate::Preset::Trusted`] so a default overlay
/// does not ask for every call. [`crate::posture::ModeOverlay::Interactive`] still raises a
/// non-read allow to ask, including HTTP GET and MCP. The floor still denies.
/// A missing waiter turns ask into deny. [`crate::fs::FsRoots`] resolves
/// filesystem and patch paths before the decision. An empty workspace skips
/// the inside and outside check. A rejected path still denies.
#[must_use]
pub fn evaluate_tool_gate(
    tool: &str,
    args: &Value,
    overlay: crate::posture::ModeOverlay,
    waiter: crate::principal::ApprovalWaiter,
    roots: crate::fs::FsRoots<'_>,
) -> crate::Decision {
    let principal = Principal::human("operator", HumanChannel::CliTty);
    let mut session = SessionScope::new("tool");
    session.overlay = overlay;
    session.cwd = roots.cwd.to_owned();
    let mut action = shadow_action(tool, args);
    crate::fs::annotate_fs_action(&mut action, roots);
    attach_shell_plan(trusted_policy(), &mut action, args);
    let request = Request::new(&principal, waiter, &session, action, args);
    trusted_policy().decide(&request)
}

/// Evaluate and log. Does not return a replacement effect.
///
/// Diffs log at info with the tool name and reason code. Arguments stay out
/// of the log line.
pub fn observe_shadow(tool: &str, args: &Value, live: Effect) {
    let decision = evaluate_shadow(tool, args);
    let comparison = compare_shadow(live, &decision);
    if comparison.differs {
        tracing::info!(
            target: "caliche_policy::shadow",
            tool,
            live = comparison.live.as_str(),
            shadow = comparison.shadow.as_str(),
            reason = comparison.shadow_reason.as_str(),
            "unified policy shadow diff; live decision unchanged"
        );
    } else {
        tracing::debug!(
            target: "caliche_policy::shadow",
            tool,
            effect = comparison.live.as_str(),
            "unified policy shadow match"
        );
    }
}

fn shell_dialect(tool: &str) -> ShellDialect {
    match tool {
        "powershell" | "pwsh" => ShellDialect::PowerShell,
        "cmd" => ShellDialect::Cmd,
        _ => ShellDialect::Posix,
    }
}

fn attach_shell_plan(policy: &CompiledPolicy, action: &mut Action, args: &Value) {
    let Action::Tool {
        facts: ToolFacts::Shell { dialect, plan },
        ..
    } = action
    else {
        return;
    };
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return;
    };
    let parsed = match *dialect {
        ShellDialect::Posix => policy.analyze_shell(&crate::posix::PosixShellAnalyzer, command),
        ShellDialect::PowerShell => policy.analyze_shell(&crate::pwsh::PowerShellAnalyzer, command),
        ShellDialect::Cmd => policy.analyze_shell(&crate::cmd::CmdShellAnalyzer, command),
    };
    match parsed {
        None => {}
        Some(Ok(parsed)) => *plan = Some(parsed),
        Some(Err(_)) => {
            *plan = Some(crate::action::ShellPlan {
                opaque: true,
                segment_count: 0,
                denied: false,
            });
        }
    }
}

fn trusted_policy() -> &'static CompiledPolicy {
    static POLICY: OnceLock<CompiledPolicy> = OnceLock::new();
    POLICY.get_or_init(|| PolicyBuilder::preset(crate::Preset::Trusted).build_or_missing())
}

fn shadow_policy() -> &'static CompiledPolicy {
    static POLICY: OnceLock<CompiledPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        PolicyBuilder::preset(crate::Preset::workstation_default()).build_or_missing()
    })
}

fn shadow_principal() -> Principal {
    Principal::human("shadow", HumanChannel::CliTty)
}

fn shadow_session() -> SessionScope {
    SessionScope::new("shadow")
}

fn shadow_action(tool: &str, args: &Value) -> Action {
    let class = tool_class_from_name(tool);
    let facts = facts_from_args(class, tool, args);
    Action::tool(class, tool, facts)
}

fn tool_class_from_name(tool: &str) -> ToolClass {
    match tool {
        "fs" | "edit_file" | "lsp" => ToolClass::Fs,
        "bash" | "shell" | "powershell" | "pwsh" | "cmd" => ToolClass::Shell,
        "http" => ToolClass::Http,
        "web_fetch" | "web_search" => ToolClass::WebFetch,
        "execute_python" => ToolClass::CodeExec,
        "apply_patch" => ToolClass::Patch,
        "git" => ToolClass::Git,
        "sql" | "sql_execution" => ToolClass::Sql,
        "workspace" => ToolClass::Workspace,
        other if other.contains("__") => ToolClass::Mcp,
        _ => ToolClass::Other,
    }
}

fn facts_from_args(class: ToolClass, tool: &str, args: &Value) -> ToolFacts {
    match class {
        ToolClass::Fs => ToolFacts::Fs {
            op: fs_op_from_args(args),
            paths: paths_from_args(args),
        },
        ToolClass::Patch => ToolFacts::Patch {
            paths: paths_from_args(args),
        },
        ToolClass::Shell => ToolFacts::Shell {
            dialect: shell_dialect(tool),
            plan: None,
        },
        ToolClass::Http | ToolClass::WebFetch => ToolFacts::Http {
            method: HttpMethod::Get,
            host: host_from_args(args).unwrap_or_default(),
        },
        ToolClass::Mcp => mcp_facts(tool),
        _ => ToolFacts::Opaque,
    }
}

fn mcp_facts(tool: &str) -> ToolFacts {
    let (server, name) = match tool.split_once("__") {
        Some((server, name)) if !server.is_empty() && !name.is_empty() => (server, name),
        _ => ("mcp", tool),
    };
    ToolFacts::Mcp {
        server: server.to_string(),
        tool: name.to_string(),
    }
}

fn fs_op_from_args(args: &Value) -> FsOp {
    match args.get("op").and_then(Value::as_str) {
        Some("read" | "grep" | "glob") => FsOp::Read,
        Some("write" | "edit" | "append") => FsOp::Write,
        Some("delete" | "remove") => FsOp::Delete,
        Some("list") => FsOp::List,
        _ => FsOp::Other,
    }
}

fn paths_from_args(args: &Value) -> Vec<ResolvedPath> {
    crate::fs::raw_paths_from_args(args)
        .into_iter()
        .map(ResolvedPath::new)
        .collect()
}

fn host_from_args(args: &Value) -> Option<String> {
    let url = args.get("url").and_then(Value::as_str)?;
    host_of_url(url)
}

fn host_of_url(raw: &str) -> Option<String> {
    let parsed = url::Url::parse(raw)
        .or_else(|_| url::Url::parse(&format!("http://{raw}")))
        .ok()?;
    let host = parsed.host_str()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::Effect;
    use crate::reason::ReasonCode;

    #[test]
    fn mcp_tool_name_keeps_server_and_tool() {
        let action = shadow_action("acme__search", &Value::Null);
        match action {
            Action::Tool {
                facts: ToolFacts::Mcp { server, tool },
                ..
            } => {
                assert_eq!(server, "acme");
                assert_eq!(tool, "search");
            }
            other => panic!("expected mcp facts, got {other:?}"),
        }
    }

    #[test]
    fn vendor_auth_read_is_denied() {
        let decision = evaluate_shadow(
            "fs",
            &serde_json::json!({"op": "read", "path": "/tmp/.codex/auth.json"}),
        );
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::VendorAuth);
    }

    #[test]
    fn interactive_http_get_asks_when_a_human_can_answer() {
        let decision = evaluate_tool_gate(
            "http",
            &serde_json::json!({"method": "GET", "url": "https://example.com/health"}),
            crate::posture::ModeOverlay::Interactive,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new(".", ""),
        );
        assert_eq!(decision.effect, Effect::Ask);
    }

    #[test]
    fn interactive_mcp_tool_asks() {
        let decision = evaluate_tool_gate(
            "github__create_issue",
            &serde_json::json!({"title": "bug"}),
            crate::posture::ModeOverlay::Interactive,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new(".", ""),
        );
        assert_eq!(decision.effect, Effect::Ask);
    }

    #[test]
    fn default_overlay_allows_http_get() {
        let decision = evaluate_tool_gate(
            "http",
            &serde_json::json!({"method": "GET", "url": "https://example.com/health"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new(".", ""),
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn interactive_without_a_waiter_denies_http_get() {
        let decision = evaluate_tool_gate(
            "http",
            &serde_json::json!({"method": "GET", "url": "https://example.com/health"}),
            crate::posture::ModeOverlay::Interactive,
            crate::principal::ApprovalWaiter::Absent,
            crate::fs::FsRoots::new(".", ""),
        );
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::NoApprover);
    }

    #[test]
    fn floor_still_denies_vendor_auth_on_the_live_gate() {
        let decision = evaluate_tool_gate(
            "fs",
            &serde_json::json!({"op": "read", "path": "/tmp/.codex/auth.json"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new(".", ""),
        );
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::VendorAuth);
    }

    #[test]
    fn relative_escape_asks_when_a_human_can_answer() {
        let decision = evaluate_tool_gate(
            "fs",
            &serde_json::json!({"op": "read", "path": "../../secret"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new("/work/sub", "/work"),
        );
        assert_eq!(decision.effect, Effect::Ask);
        assert_eq!(decision.reason.code(), ReasonCode::OutsideWorkspace);
    }

    #[test]
    fn relative_escape_without_a_waiter_denies() {
        let decision = evaluate_tool_gate(
            "fs",
            &serde_json::json!({"op": "read", "path": "../../secret"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Absent,
            crate::fs::FsRoots::new("/work/sub", "/work"),
        );
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::NoApprover);
    }

    #[test]
    fn path_inside_the_workspace_stays_allowed() {
        let decision = evaluate_tool_gate(
            "fs",
            &serde_json::json!({"op": "read", "path": "src/lib.rs"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Absent,
            crate::fs::FsRoots::new("/work", "/work"),
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn alternate_data_stream_denies_on_the_live_gate() {
        let decision = evaluate_tool_gate(
            "fs",
            &serde_json::json!({"op": "read", "path": "notes.txt:Zone.Identifier"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new("/work", "/work"),
        );
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::PathRejected);
    }

    #[test]
    fn patch_escape_asks() {
        let decision = evaluate_tool_gate(
            "apply_patch",
            &serde_json::json!({"patch": "*** Update File: ../secret\n"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new("/work", "/work"),
        );
        assert_eq!(decision.effect, Effect::Ask);
        assert_eq!(decision.reason.code(), ReasonCode::OutsideWorkspace);
    }

    #[test]
    fn writes_into_git_metadata_are_denied_under_accept_all() {
        for (tool, args) in [
            (
                "fs",
                serde_json::json!({"op": "write", "path": ".git/config"}),
            ),
            (
                "fs",
                serde_json::json!({"op": "edit", "path": "/work/.git/hooks/pre-commit"}),
            ),
            (
                "fs",
                serde_json::json!({"op": "write", "path": ".GIT/config"}),
            ),
            (
                "fs",
                serde_json::json!({"op": "write", "path": "vendor/dep/.git/config"}),
            ),
            ("fs", serde_json::json!({"op": "delete", "path": ".git"})),
            (
                "fs",
                serde_json::json!({"op": "write", "path": "sub/../.git/info/attributes"}),
            ),
            ("edit_file", serde_json::json!({"path": ".git/config"})),
            (
                "fs",
                serde_json::json!({"op": "write", "path": ".git./config"}),
            ),
            (
                "fs",
                serde_json::json!({"op": "write", "path": ".git /hooks/pre-commit"}),
            ),
            (
                "apply_patch",
                serde_json::json!({"patch": "*** Update File: .git/config\n"}),
            ),
            (
                "apply_patch",
                serde_json::json!({
                    "patch": "*** Begin Patch\n*** Update File: README.md\n*** Move to: .git/config\n@@\n-a\n+b\n*** End Patch\n"
                }),
            ),
        ] {
            let decision = evaluate_tool_gate(
                tool,
                &args,
                crate::posture::ModeOverlay::AcceptAll,
                crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
                crate::fs::FsRoots::new("/work", "/work"),
            );
            assert_eq!(
                (decision.effect, decision.reason.code()),
                (Effect::Deny, ReasonCode::GitMetadata),
                "{tool} {args}"
            );
        }
    }

    #[test]
    fn git_metadata_reads_and_lookalike_names_stay_allowed() {
        for args in [
            serde_json::json!({"op": "read", "path": ".git/config"}),
            serde_json::json!({"op": "write", "path": ".gitignore"}),
            serde_json::json!({"op": "write", "path": ".github/workflows/x.yml"}),
            serde_json::json!({"op": "write", "path": "docs/git/notes.md"}),
        ] {
            let decision = evaluate_tool_gate(
                "fs",
                &args,
                crate::posture::ModeOverlay::AcceptAll,
                crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
                crate::fs::FsRoots::new("/work", "/work"),
            );
            assert_eq!(decision.effect, Effect::Allow, "{args}");
        }
    }

    #[test]
    fn a_git_write_outside_the_workspace_still_asks() {
        let decision = evaluate_tool_gate(
            "fs",
            &serde_json::json!({"op": "write", "path": "/elsewhere/.git/config"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new("/work", "/work"),
        );
        assert_eq!(
            (decision.effect, decision.reason.code()),
            (Effect::Ask, ReasonCode::OutsideWorkspace)
        );
    }

    fn gate(
        tool: &str,
        args: &Value,
        overlay: crate::posture::ModeOverlay,
        waiter: crate::principal::ApprovalWaiter,
        workspace: &str,
    ) -> crate::Decision {
        evaluate_tool_gate(
            tool,
            args,
            overlay,
            waiter,
            crate::fs::FsRoots::new("/work/sub", workspace),
        )
    }

    #[test]
    fn accept_all_cannot_lift_an_outside_write() {
        let human = crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty);
        for (tool, args) in [
            (
                "fs",
                serde_json::json!({"op": "write", "path": "../../etc/cron.d/x", "content": "x"}),
            ),
            (
                "fs",
                serde_json::json!({"op": "delete", "path": "/etc/hosts"}),
            ),
            ("edit_file", serde_json::json!({"path": "/tmp/outside.rs"})),
            (
                "apply_patch",
                serde_json::json!({"patch": "*** Update File: ../../secret\n"}),
            ),
        ] {
            for overlay in [
                crate::posture::ModeOverlay::AcceptAll,
                crate::posture::ModeOverlay::AcceptEdits,
            ] {
                let asked = gate(tool, &args, overlay, human, "/work");
                assert_eq!(asked.effect, Effect::Ask, "{tool} {args} {overlay:?}");
                assert_eq!(asked.reason.code(), ReasonCode::OutsideWorkspace);
                let closed = gate(
                    tool,
                    &args,
                    overlay,
                    crate::principal::ApprovalWaiter::Absent,
                    "/work",
                );
                assert_eq!(closed.effect, Effect::Deny, "{tool} {args} {overlay:?}");
                assert_eq!(closed.reason.code(), ReasonCode::NoApprover);
            }
        }
    }

    #[test]
    fn accept_all_cannot_lift_an_outside_read_or_list() {
        for (op, path) in [
            ("read", "~/.ssh/id_ed25519"),
            ("read", "/etc/passwd"),
            ("list", "~/.ssh"),
            ("read", "../../notes.txt"),
            ("search", "/etc"),
            ("glob", "/etc"),
            ("grep", "/etc"),
        ] {
            let args = serde_json::json!({"op": op, "path": path});
            for overlay in [
                crate::posture::ModeOverlay::AcceptAll,
                crate::posture::ModeOverlay::AcceptEdits,
            ] {
                let asked = gate(
                    "fs",
                    &args,
                    overlay,
                    crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
                    "/work",
                );
                assert_eq!(asked.effect, Effect::Ask, "{args} {overlay:?}");
                assert_eq!(asked.reason.code(), ReasonCode::OutsideWorkspace);
                let closed = gate(
                    "fs",
                    &args,
                    overlay,
                    crate::principal::ApprovalWaiter::Absent,
                    "/work",
                );
                assert_eq!(closed.effect, Effect::Deny, "{args} {overlay:?}");
                assert_eq!(closed.reason.code(), ReasonCode::NoApprover);
            }
        }
    }

    #[test]
    fn accept_all_still_allows_a_write_inside_the_workspace() {
        let decision = gate(
            "fs",
            &serde_json::json!({"op": "write", "path": "src/lib.rs", "content": "x"}),
            crate::posture::ModeOverlay::AcceptAll,
            crate::principal::ApprovalWaiter::Absent,
            "/work",
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn accept_all_still_allows_a_read_inside_the_workspace() {
        let decision = gate(
            "fs",
            &serde_json::json!({"op": "read", "path": "src/lib.rs"}),
            crate::posture::ModeOverlay::AcceptAll,
            crate::principal::ApprovalWaiter::Absent,
            "/work",
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn missing_workspace_denies_every_filesystem_action() {
        let human = crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty);
        for workspace in ["", "   "] {
            for (tool, args) in [
                (
                    "fs",
                    serde_json::json!({"op": "read", "path": "src/lib.rs"}),
                ),
                (
                    "fs",
                    serde_json::json!({"op": "write", "path": "src/lib.rs"}),
                ),
                ("fs", serde_json::json!({"op": "list"})),
                ("edit_file", serde_json::json!({"path": "src/lib.rs"})),
                (
                    "apply_patch",
                    serde_json::json!({"patch": "*** Update File: src/lib.rs\n"}),
                ),
            ] {
                for overlay in [
                    crate::posture::ModeOverlay::Default,
                    crate::posture::ModeOverlay::AcceptAll,
                ] {
                    let decision = gate(tool, &args, overlay, human, workspace);
                    assert_eq!(decision.effect, Effect::Deny, "{tool} {args} {overlay:?}");
                    assert_eq!(decision.reason.code(), ReasonCode::NoWorkspace);
                }
            }
        }
    }

    #[test]
    fn missing_workspace_does_not_block_a_non_filesystem_tool() {
        let decision = gate(
            "http",
            &serde_json::json!({"method": "GET", "url": "https://example.com/health"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Absent,
            "",
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn trusted_gate_does_not_deny_recursive_rm() {
        let decision = evaluate_tool_gate(
            "bash",
            &serde_json::json!({"command": "rm -rf /"}),
            crate::posture::ModeOverlay::Default,
            crate::principal::ApprovalWaiter::Present(HumanChannel::CliTty),
            crate::fs::FsRoots::new(".", ""),
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn unmatched_mcp_tool_asks() {
        let decision = evaluate_shadow("acme__search", &Value::Null);
        assert_eq!(decision.effect, Effect::Ask);
    }

    #[test]
    fn mcp_connect_asks_on_the_workstation_preset() {
        let decision = evaluate_action(
            Action::McpConnect {
                server: "acme".to_string(),
            },
            &Value::Null,
        );
        assert_eq!(decision.effect, Effect::Ask);
    }
}
