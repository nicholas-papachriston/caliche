//! Join sandbox obligations with host capabilities.
//!
//! `decide` does not probe the OS. A named host that cannot meet an obligation
//! changes shell and code execution. An unnamed host does not.

use crate::policy::CompiledPolicy;
use crate::posture::{FsConfine, HostCapability, NetConfine, Obligations, Preset, ProcConfine};

/// Obligations the named host cannot enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObligationGap {
    /// Network obligation is set and the host cannot enforce it.
    pub network: bool,
    /// Filesystem obligation is set and the host cannot enforce it.
    pub filesystem: bool,
    /// Process obligation is set and the host cannot enforce it.
    pub process: bool,
}

impl ObligationGap {
    /// Shell and code execution change when files or the network are unmet.
    #[must_use]
    pub const fn limits_shell(self) -> bool {
        self.network || self.filesystem
    }
}

/// Compare `obligations` with what `host` can enforce.
///
/// [`HostCapability::Unspecified`] produces an empty gap. The caller did not
/// name a backend, so `decide` does not invent one.
#[must_use]
pub const fn obligation_gap(obligations: Obligations, host: HostCapability) -> ObligationGap {
    if matches!(host, HostCapability::Unspecified) {
        return ObligationGap {
            network: false,
            filesystem: false,
            process: false,
        };
    }
    ObligationGap {
        network: !matches!(obligations.net, NetConfine::None) && !host_enforces_network(host),
        filesystem: !matches!(obligations.fs, FsConfine::None) && !host_enforces_filesystem(host),
        process: !matches!(obligations.proc, ProcConfine::None) && !host_enforces_process(host),
    }
}

/// Whether this host confines filesystem reads and writes.
#[must_use]
pub const fn host_enforces_filesystem(host: HostCapability) -> bool {
    match host {
        HostCapability::MacOsSeatbelt
        | HostCapability::LinuxLandlock { .. }
        | HostCapability::WindowsDockerOrWsl2
        | HostCapability::Docker => true,
        HostCapability::Unspecified | HostCapability::WindowsJobOnly | HostCapability::Noop => {
            false
        }
    }
}

/// Whether this host can deny or allow outbound network.
#[must_use]
pub const fn host_enforces_network(host: HostCapability) -> bool {
    match host {
        HostCapability::MacOsSeatbelt
        | HostCapability::WindowsDockerOrWsl2
        | HostCapability::Docker => true,
        HostCapability::LinuxLandlock {
            network_enforceable,
        } => network_enforceable,
        HostCapability::Unspecified | HostCapability::WindowsJobOnly | HostCapability::Noop => {
            false
        }
    }
}

/// Whether this host can kill the process tree.
#[must_use]
pub const fn host_enforces_process(host: HostCapability) -> bool {
    match host {
        HostCapability::MacOsSeatbelt
        | HostCapability::WindowsJobOnly
        | HostCapability::WindowsDockerOrWsl2
        | HostCapability::Docker => true,
        HostCapability::Unspecified
        | HostCapability::LinuxLandlock { .. }
        | HostCapability::Noop => false,
    }
}

/// Capability panel for `policy show`.
///
/// A Windows Job host includes the line `network not sandboxed` and the
/// shell effect for `standard` and `locked`.
#[must_use]
pub fn policy_show(policy: &CompiledPolicy, host: HostCapability) -> String {
    let mut text = String::new();
    push_header(&mut text, policy);
    push_host(&mut text, host);
    push_gaps(&mut text, obligation_gap(policy.obligations(), host));
    text
}

fn push_header(text: &mut String, policy: &CompiledPolicy) {
    let preset = policy.preset().map_or("none", Preset::as_str);
    let obligations = policy.obligations();
    text.push_str("preset: ");
    text.push_str(preset);
    text.push('\n');
    text.push_str("obligations: fs=");
    text.push_str(fs_label(obligations.fs));
    text.push_str(" net=");
    text.push_str(net_label(obligations.net));
    text.push_str(" proc=");
    text.push_str(proc_label(obligations.proc));
    text.push('\n');
}

fn push_host(text: &mut String, host: HostCapability) {
    text.push_str("host: ");
    text.push_str(host_label(host));
    text.push('\n');
    text.push_str("filesystem: ");
    text.push_str(sandbox_word(host_enforces_filesystem(host)));
    text.push('\n');
    text.push_str("network: ");
    text.push_str(sandbox_word(host_enforces_network(host)));
    text.push('\n');
    text.push_str("process: ");
    text.push_str(sandbox_word(host_enforces_process(host)));
    text.push('\n');
}

fn push_gaps(text: &mut String, gap: ObligationGap) {
    if gap.network {
        text.push_str(
            "network not sandboxed; shell/code_exec => Ask under standard, Deny under locked\n",
        );
    }
    if gap.filesystem {
        text.push_str("filesystem not sandboxed\n");
    }
}

const fn sandbox_word(enforced: bool) -> &'static str {
    if enforced {
        "sandboxed"
    } else {
        "not sandboxed"
    }
}

const fn fs_label(fs: FsConfine) -> &'static str {
    match fs {
        FsConfine::None => "none",
        FsConfine::WorkspaceWrite => "workspace_write",
        FsConfine::ReadOnly => "read_only",
    }
}

const fn net_label(net: NetConfine) -> &'static str {
    match net {
        NetConfine::None => "none",
        NetConfine::DenyAll => "deny_all",
        NetConfine::AllowPublic => "allow_public",
    }
}

const fn proc_label(proc: ProcConfine) -> &'static str {
    match proc {
        ProcConfine::None => "none",
        ProcConfine::KillOnDrop => "kill_on_drop",
    }
}

fn host_label(host: HostCapability) -> &'static str {
    host.label()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, ToolClass, ToolFacts};
    use crate::effect::Effect;
    use crate::policy::PolicyBuilder;
    use crate::posture::ModeOverlay;
    use crate::principal::{ApprovalWaiter, HumanChannel, Principal};
    use crate::reason::ReasonCode;
    use crate::request::{Request, SessionScope};

    fn shell_decision(
        preset: Preset,
        host: HostCapability,
        overlay: ModeOverlay,
    ) -> crate::Decision {
        let policy = PolicyBuilder::preset(preset).build().expect("preset");
        let action = Action::tool(
            ToolClass::Shell,
            "bash",
            ToolFacts::Shell {
                dialect: crate::posture::ShellDialect::Posix,
                plan: None,
            },
        );
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let mut session = SessionScope::new("cap");
        session.host = host;
        session.overlay = overlay;
        let args = serde_json::json!({"command": "git status"});
        policy.decide(&Request::new(
            &principal,
            ApprovalWaiter::Present(HumanChannel::CliTty),
            &session,
            action,
            &args,
        ))
    }

    #[test]
    fn windows_job_show_names_the_network_gap() {
        let policy = PolicyBuilder::preset(Preset::Standard)
            .build()
            .expect("standard");
        let text = policy_show(&policy, HostCapability::WindowsJobOnly);
        assert!(text.contains("network not sandboxed"));
        assert!(text.contains("Ask under standard, Deny under locked"));
        assert!(text.contains("filesystem not sandboxed"));
        assert!(text.contains("process: sandboxed"));
    }

    #[test]
    fn seatbelt_show_does_not_call_the_network_unmet() {
        let policy = PolicyBuilder::preset(Preset::Standard)
            .build()
            .expect("standard");
        let text = policy_show(&policy, HostCapability::MacOsSeatbelt);
        assert!(!text.contains("network not sandboxed"));
        assert!(text.contains("network: sandboxed"));
    }

    #[test]
    fn trusted_shell_on_windows_job_asks() {
        let decision = shell_decision(
            Preset::Trusted,
            HostCapability::WindowsJobOnly,
            ModeOverlay::Default,
        );
        assert_eq!(decision.effect, Effect::Ask);
        assert_eq!(decision.reason.code(), ReasonCode::UnmetSandbox);
        assert!(decision.reason.message().contains("network not sandboxed"));
    }

    #[test]
    fn landlock_with_network_allows_trusted_shell() {
        let decision = shell_decision(
            Preset::Trusted,
            HostCapability::LinuxLandlock {
                network_enforceable: true,
            },
            ModeOverlay::Default,
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn trusted_shell_on_seatbelt_stays_allowed() {
        let decision = shell_decision(
            Preset::Trusted,
            HostCapability::MacOsSeatbelt,
            ModeOverlay::Default,
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn unnamed_host_does_not_change_trusted_shell() {
        let decision = shell_decision(
            Preset::Trusted,
            HostCapability::Unspecified,
            ModeOverlay::Default,
        );
        assert_eq!(decision.effect, Effect::Allow);
    }

    #[test]
    fn landlock_without_network_asks_for_code_exec() {
        let policy = PolicyBuilder::preset(Preset::Trusted)
            .build()
            .expect("trusted");
        let action = Action::tool(
            ToolClass::CodeExec,
            "execute_python",
            ToolFacts::CodeExec {
                lang: "python".to_owned(),
            },
        );
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let mut session = SessionScope::new("cap");
        session.host = HostCapability::LinuxLandlock {
            network_enforceable: false,
        };
        let args = serde_json::json!({"code": "print(1)"});
        let decision = policy.decide(&Request::new(
            &principal,
            ApprovalWaiter::Present(HumanChannel::CliTty),
            &session,
            action,
            &args,
        ));
        assert_eq!(decision.effect, Effect::Ask);
        assert_eq!(decision.reason.code(), ReasonCode::UnmetSandbox);
    }

    #[test]
    fn locked_shell_on_windows_job_denies() {
        let decision = shell_decision(
            Preset::Locked,
            HostCapability::WindowsJobOnly,
            ModeOverlay::Default,
        );
        assert_eq!(decision.effect, Effect::Deny);
    }

    #[test]
    fn accept_all_cannot_lift_the_windows_job_ask() {
        let decision = shell_decision(
            Preset::Trusted,
            HostCapability::WindowsJobOnly,
            ModeOverlay::AcceptAll,
        );
        assert_eq!(decision.effect, Effect::Ask);
        assert_eq!(decision.reason.code(), ReasonCode::UnmetSandbox);
    }

    #[test]
    fn windows_job_does_not_replace_a_posix_denylist_deny() {
        let policy = PolicyBuilder::preset(Preset::Standard)
            .build()
            .expect("standard");
        let plan = policy
            .analyze_shell(&crate::posix::PosixShellAnalyzer, "rm -rf /")
            .expect("facet")
            .expect("plan");
        let action = Action::tool(
            ToolClass::Shell,
            "bash",
            ToolFacts::Shell {
                dialect: crate::posture::ShellDialect::Posix,
                plan: Some(plan),
            },
        );
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let mut session = SessionScope::new("cap");
        session.host = HostCapability::WindowsJobOnly;
        let args = serde_json::json!({"command": "rm -rf /"});
        let decision = policy.decide(&Request::new(
            &principal,
            ApprovalWaiter::Present(HumanChannel::CliTty),
            &session,
            action,
            &args,
        ));
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::PosixDenylist);
    }

    #[test]
    fn danger_on_noop_has_no_gap() {
        let decision = shell_decision(Preset::Danger, HostCapability::Noop, ModeOverlay::Default);
        assert_eq!(decision.effect, Effect::Allow);
        let policy = PolicyBuilder::preset(Preset::Danger)
            .build()
            .expect("danger");
        let text = policy_show(&policy, HostCapability::Noop);
        assert!(!text.contains("network not sandboxed"));
    }

    #[test]
    fn windows_job_does_not_ask_for_http() {
        let policy = PolicyBuilder::preset(Preset::Trusted)
            .build()
            .expect("trusted");
        let action = Action::tool(
            ToolClass::Http,
            "http",
            ToolFacts::Http {
                method: crate::action::HttpMethod::Get,
                host: "example.com".to_owned(),
            },
        );
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let mut session = SessionScope::new("cap");
        session.host = HostCapability::WindowsJobOnly;
        let args = serde_json::json!({"url": "https://example.com"});
        let decision = policy.decide(&Request::new(
            &principal,
            ApprovalWaiter::Present(HumanChannel::CliTty),
            &session,
            action,
            &args,
        ));
        assert_eq!(decision.effect, Effect::Allow);
    }
}
