//! Vault fail-closed invariants for the typed policy core.
//!
//! These tests are the merge gate for the rules in DESIGN §2 / the vault
//! review: missing policy denies, deny wins, lower layers cannot raise a
//! deployment deny, `AcceptAll` cannot lift that deny, no waiter turns ask into
//! deny, a checker error denies, only humans answer approvals, and decisions
//! bind tool arguments.

use caliche_policy::{
    Action, ActionBinding, ApprovalWaiter, BindingRecheck, BotOrigin, CredentialClass, Decide,
    Effect, FsOp, HostCapability, HttpMethod, HumanChannel, Layer, Matcher, ModeOverlay, PendingId,
    PolicyBuilder, PolicyError, PolicySource, Preset, Principal, Request, ResolvedPath, Rule,
    RuleSet, RuleSource, SandboxRequirement, SessionScope, ShellAnalyzer, ShellDialect, ShellPlan,
    StubShellAnalyzer, ToolClass, ToolFacts, compare_shadow, evaluate_shadow, ingest_source,
    parse_effect, parse_mode_overlay, parse_preset, shadow_enabled_value,
};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

fn human() -> Principal {
    Principal::human("operator", HumanChannel::CliTty)
}

fn args() -> serde_json::Value {
    json!({})
}

fn decide(
    policy: &caliche_policy::CompiledPolicy,
    principal: &Principal,
    waiter: ApprovalWaiter,
    session: &SessionScope,
    action: Action,
    raw_args: &serde_json::Value,
) -> caliche_policy::Decision {
    let request = Request::new(principal, waiter, session, action, raw_args);
    policy.decide(&request)
}

fn bash(command: &str) -> Action {
    Action::tool(
        ToolClass::Shell,
        "bash",
        ToolFacts::Shell {
            dialect: ShellDialect::Posix,
            plan: Some(ShellPlan {
                opaque: false,
                segment_count: 1,
                denied: false,
            }),
        },
    )
    .tap_command(command)
}

trait Tap {
    fn tap_command(self, _command: &str) -> Self;
}

impl Tap for Action {
    fn tap_command(self, _command: &str) -> Self {
        self
    }
}

fn tool_name_rule(layer: Layer, effect: Effect, name: &str, id: &str) -> Rule {
    Rule {
        id: id.to_owned(),
        layer,
        effect,
        hard: effect == Effect::Deny,
        source: RuleSource::Builtin { id: "caller" },
        matcher: Matcher::ToolName(name.to_owned()),
    }
}

#[test]
fn effect_join_deny_wins() {
    assert_eq!(Effect::Allow.join(Effect::Deny), Effect::Deny);
    assert_eq!(Effect::Deny.join(Effect::Ask), Effect::Deny);
    assert_eq!(Effect::Allow.join(Effect::Ask), Effect::Ask);
    assert!(Effect::Allow < Effect::Ask && Effect::Ask < Effect::Deny);
}

#[test]
fn missing_policy_denies() {
    let policy = PolicyBuilder::new().build().expect("empty snapshot");
    assert_eq!(policy.deployment_default(), None);
    let session = SessionScope::new("s");
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        Action::tool(ToolClass::Shell, "bash", ToolFacts::Opaque),
        &args(),
    );
    assert_eq!(decision.effect, Effect::Deny);
    assert_eq!(
        decision.reason.code(),
        caliche_policy::ReasonCode::MissingPolicy
    );
}

#[test]
fn unknown_preset_and_permission_mode_error() {
    assert!(matches!(
        parse_preset("yolo"),
        Err(PolicyError::UnknownPreset(_))
    ));
    assert!(matches!(
        parse_mode_overlay(Some("nope")),
        Err(PolicyError::UnknownPermissionMode(_))
    ));
    assert!(matches!(
        parse_effect("permit"),
        Err(PolicyError::UnknownEffect(_))
    ));
    assert_eq!(
        parse_mode_overlay(Some("accept-edits")).expect("parse"),
        ModeOverlay::AcceptEdits
    );
    assert_eq!(
        parse_mode_overlay(None).expect("absent"),
        ModeOverlay::Default
    );
    let unknown = serde_json::from_str::<ModeOverlay>("\"yolo\"");
    assert!(unknown.is_err());
}

#[test]
fn preset_defaults_match_documented_table() {
    assert_eq!(Preset::workstation_default(), Preset::Standard);
    assert_eq!(Preset::new_deploy_default(), Preset::Locked);
    assert_eq!(Preset::Locked.default_effect(), Effect::Deny);
    assert_eq!(Preset::Standard.default_effect(), Effect::Ask);
    assert_eq!(Preset::Trusted.default_effect(), Effect::Allow);
    assert_eq!(
        Preset::Trusted.sandbox(),
        SandboxRequirement::WorkspaceWrite
    );
    assert_ne!(Preset::Trusted.sandbox(), SandboxRequirement::None);
    let trusted = PolicyBuilder::preset(Preset::Trusted)
        .build()
        .expect("trusted");
    assert!(trusted.active_facets().bits() == 0);
}

#[test]
fn deny_beats_allow_across_layers() {
    let policy = PolicyBuilder::preset(Preset::Trusted)
        .rule(Rule::deny_tool_class(
            Layer::Project,
            ToolClass::Shell,
            "no-shell",
        ))
        .build()
        .expect("compile");
    let session = SessionScope::new("s");
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("ls"),
        &json!({"command": "ls"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
}

#[test]
fn deny_beats_a_more_specific_rule_in_the_same_layer() {
    let principal = human();
    let waiter = ApprovalWaiter::Present(HumanChannel::CliTty);
    let raw = json!({"command": "ls"});
    let mut session = SessionScope::new("s");

    let deployment = PolicyBuilder::preset(Preset::Trusted)
        .rule(Rule::deny_tool_class(
            Layer::Deployment,
            ToolClass::Shell,
            "deny-shell",
        ))
        .rule(tool_name_rule(
            Layer::Deployment,
            Effect::Allow,
            "bash",
            "allow-bash",
        ))
        .build()
        .expect("deployment");
    assert_eq!(
        decide(&deployment, &principal, waiter, &session, bash("ls"), &raw).effect,
        Effect::Deny
    );

    let profile = PolicyBuilder::preset(Preset::Trusted)
        .rule(Rule::deny_tool_class(
            Layer::Profile,
            ToolClass::Shell,
            "deny-shell",
        ))
        .rule(tool_name_rule(
            Layer::Profile,
            Effect::Ask,
            "bash",
            "ask-bash",
        ))
        .build()
        .expect("profile");
    session.overlay = ModeOverlay::AcceptAll;
    assert_eq!(
        decide(&profile, &principal, waiter, &session, bash("ls"), &raw).effect,
        Effect::Deny
    );

    let session_policy = PolicyBuilder::preset(Preset::Trusted)
        .rule(Rule::deny_tool_class(
            Layer::Session,
            ToolClass::Shell,
            "deny-shell",
        ))
        .rule(tool_name_rule(
            Layer::Session,
            Effect::Ask,
            "bash",
            "ask-bash",
        ))
        .build()
        .expect("session");
    assert_eq!(
        decide(
            &session_policy,
            &principal,
            waiter,
            &session,
            bash("ls"),
            &raw
        )
        .effect,
        Effect::Deny
    );

    let floor = PolicyBuilder::preset(Preset::Trusted)
        .floor_rule(Rule::deny_tool_class(
            Layer::Floor,
            ToolClass::Shell,
            "deny-shell",
        ))
        .floor_rule(tool_name_rule(
            Layer::Floor,
            Effect::Ask,
            "bash",
            "ask-bash",
        ))
        .build()
        .expect("floor");
    session.overlay = ModeOverlay::Default;
    assert_eq!(
        decide(&floor, &principal, waiter, &session, bash("ls"), &raw).effect,
        Effect::Deny
    );
}

#[test]
fn floor_and_deployment_deny_beat_a_later_layer_ask() {
    let principal = human();
    let waiter = ApprovalWaiter::Present(HumanChannel::CliTty);
    let raw = json!({"command": "ls"});
    let mut session = SessionScope::new("s");
    session.overlay = ModeOverlay::AcceptAll;

    let floor = PolicyBuilder::preset(Preset::Trusted)
        .floor_rule(Rule::deny_tool_class(
            Layer::Floor,
            ToolClass::Shell,
            "floor-deny-shell",
        ))
        .rule(tool_name_rule(
            Layer::Profile,
            Effect::Ask,
            "bash",
            "profile-ask-bash",
        ))
        .build()
        .expect("floor");
    assert_eq!(
        decide(&floor, &principal, waiter, &session, bash("ls"), &raw).effect,
        Effect::Deny
    );

    let deployment = PolicyBuilder::preset(Preset::Trusted)
        .rule(Rule::deny_tool_class(
            Layer::Deployment,
            ToolClass::Shell,
            "deploy-deny-shell",
        ))
        .rule(tool_name_rule(
            Layer::Session,
            Effect::Ask,
            "bash",
            "session-ask-bash",
        ))
        .build()
        .expect("deployment");
    assert_eq!(
        decide(&deployment, &principal, waiter, &session, bash("ls"), &raw).effect,
        Effect::Deny
    );
}

#[test]
fn profile_and_project_allows_cannot_raise_rights() {
    let profile = PolicyBuilder::preset(Preset::Locked).rule(Rule::allow_tool_class(
        Layer::Profile,
        ToolClass::Shell,
        "profile-allow",
    ));
    assert!(matches!(
        profile.build(),
        Err(PolicyError::ProfileAllowRaisesRights(_))
    ));

    let project = PolicyBuilder::preset(Preset::Locked).rule(Rule::project_allow_file(
        ToolClass::Shell,
        "repo/caliche.toml",
        4,
    ));
    assert!(matches!(
        project.build(),
        Err(PolicyError::ProjectAllowRaisesRights(_))
    ));

    let opted_in = PolicyBuilder::preset(Preset::Locked)
        .trust_project_allows(["repo/**"])
        .rule(Rule::project_allow_file(
            ToolClass::Shell,
            "repo/caliche.toml",
            4,
        ))
        .build()
        .expect("opt-in loads");
    assert_eq!(opted_in.recorded_project_allows(), 1);
    let session = SessionScope::new("s");
    let decision = decide(
        &opted_in,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("ls"),
        &json!({"command": "ls"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
}

#[test]
fn accept_all_cannot_lift_deployment_deny() {
    let policy = PolicyBuilder::preset(Preset::Standard)
        .rule(Rule::deny_tool_class(
            Layer::Deployment,
            ToolClass::Shell,
            "deploy-deny-shell",
        ))
        .build()
        .expect("compile");
    let mut session = SessionScope::new("s");
    session.overlay = ModeOverlay::AcceptAll;
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("rm -rf /"),
        &json!({"command": "rm -rf /"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
    assert_eq!(
        decision.reason.code(),
        caliche_policy::ReasonCode::DeploymentDeny
    );
}

#[test]
fn accept_all_can_lower_a_non_hard_ask_when_a_human_is_present() {
    let policy = PolicyBuilder::preset(Preset::Standard)
        .build()
        .expect("standard");
    let mut session = SessionScope::new("s");
    session.overlay = ModeOverlay::AcceptAll;
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        Action::tool(ToolClass::Fs, "fs", ToolFacts::Opaque),
        &json!({"op": "read", "path": "README.md"}),
    );
    assert_eq!(decision.effect, Effect::Allow);
}

#[test]
fn absent_waiter_turns_ask_into_deny() {
    let policy = PolicyBuilder::preset(Preset::Standard)
        .build()
        .expect("standard");
    let session = SessionScope::new("s");
    let decision = decide(
        &policy,
        &Principal::bot("cron", BotOrigin::Cron),
        ApprovalWaiter::Absent,
        &session,
        bash("ls"),
        &json!({"command": "ls"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
    assert_eq!(
        decision.reason.code(),
        caliche_policy::ReasonCode::NoApprover
    );
}

#[test]
fn checker_error_denies_even_with_accept_all() {
    let policy = PolicyBuilder::preset(Preset::Trusted)
        .build()
        .expect("trusted");
    let mut session = SessionScope::new("s");
    session.overlay = ModeOverlay::AcceptAll;
    let action = Action::tool(
        ToolClass::Fs,
        "fs",
        ToolFacts::CheckerError {
            detail: "canonicalize failed".into(),
        },
    );
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        action,
        &json!({"path": "../etc/passwd"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
    assert_eq!(
        decision.reason.code(),
        caliche_policy::ReasonCode::CheckerError
    );
}

#[test]
fn only_humans_may_answer_approvals() {
    let policy = PolicyBuilder::preset(Preset::Trusted)
        .build()
        .expect("trusted");
    let session = SessionScope::new("s");
    let pending = PendingId::parse("apr_01").expect("id");
    let action = Action::ApprovalAnswer {
        pending: pending.clone(),
    };
    for principal in [
        Principal::bot("hook", BotOrigin::Webhook),
        Principal::Peer {
            harness: "pi".into(),
        },
        Principal::Plugin { id: "ext".into() },
        Principal::Subagent {
            parent: "s".into(),
            depth: 1,
        },
    ] {
        let decision = decide(
            &policy,
            &principal,
            ApprovalWaiter::Absent,
            &session,
            action.clone(),
            &args(),
        );
        assert_eq!(decision.effect, Effect::Deny, "{principal:?}");
        assert_eq!(
            decision.reason.code(),
            caliche_policy::ReasonCode::NonHumanApprover
        );
    }
    let human_decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::UiSession),
        &session,
        Action::ApprovalAnswer { pending },
        &args(),
    );
    assert_ne!(
        human_decision.reason.code(),
        caliche_policy::ReasonCode::NonHumanApprover
    );
    assert_eq!(human_decision.effect, Effect::Allow);
}

#[test]
fn floor_denies_vendor_auth_and_subscription_credentials() {
    let policy = PolicyBuilder::preset(Preset::Trusted)
        .build()
        .expect("trusted");
    let session = SessionScope::new("s");
    let read = Action::tool(
        ToolClass::Fs,
        "fs",
        ToolFacts::Fs {
            op: FsOp::Read,
            paths: vec![ResolvedPath::new("/home/dev/.codex/auth.json")],
        },
    );
    let denied = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        read,
        &json!({"op": "read", "path": "/home/dev/.codex/auth.json"}),
    );
    assert_eq!(denied.effect, Effect::Deny);
    assert_eq!(denied.reason.code(), caliche_policy::ReasonCode::VendorAuth);

    let keychain = Action::tool(
        ToolClass::Other,
        "keychain",
        ToolFacts::Keychain {
            owned_by_caliche: false,
        },
    );
    let key_denied = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        keychain,
        &args(),
    );
    assert_eq!(key_denied.effect, Effect::Deny);

    let subscription = Action::ModelUse {
        provider: "openai".into(),
        model: "gpt".into(),
        cred: CredentialClass::VendorSubscription,
    };
    let sub_denied = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        subscription,
        &args(),
    );
    assert_eq!(sub_denied.effect, Effect::Deny);
    assert_eq!(
        sub_denied.reason.code(),
        caliche_policy::ReasonCode::ByoCredential
    );

    let byo = Action::ModelUse {
        provider: "openai".into(),
        model: "gpt".into(),
        cred: CredentialClass::ByoApiKey,
    };
    let byo_ok = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        byo,
        &args(),
    );
    assert_eq!(byo_ok.effect, Effect::Allow);
}

#[test]
fn floor_denies_metadata_and_private_egress() {
    let policy = PolicyBuilder::preset(Preset::Danger)
        .build()
        .expect("danger outside production");
    let session = SessionScope::new("s");
    let action = Action::tool(
        ToolClass::Http,
        "http",
        ToolFacts::Http {
            method: HttpMethod::Get,
            host: "169.254.169.254".into(),
        },
    );
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        action,
        &json!({"url": "http://169.254.169.254/latest/meta-data/"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
    assert_eq!(decision.reason.code(), caliche_policy::ReasonCode::Ssrf);
}

#[test]
fn binding_tracks_tool_args_and_rejects_mutation() {
    let policy = PolicyBuilder::preset(Preset::Standard)
        .build()
        .expect("standard");
    let session = SessionScope::new("s");
    let raw = json!({"command": "git status"});
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("git status"),
        &raw,
    );
    assert_eq!(decision.effect, Effect::Ask);
    let approved = decision.binding.expect("ask binds args");
    let same = ActionBinding::compute("bash", &raw, ".", Some("posix"), session.policy_version);
    assert_eq!(approved.recheck(same), BindingRecheck::Fresh);
    let mutated = json!({"command": "curl https://evil"});
    let actual =
        ActionBinding::compute("bash", &mutated, ".", Some("posix"), session.policy_version);
    assert!(matches!(
        approved.recheck(actual),
        BindingRecheck::Stale { .. }
    ));
    assert_ne!(
        ActionBinding::compute("bash", &raw, ".", Some("posix"), 1),
        ActionBinding::compute("bash", &raw, ".", Some("posix"), 2)
    );
}

#[test]
fn hard_floor_ask_survives_accept_all() {
    let policy = PolicyBuilder::preset(Preset::Standard)
        .floor_rule(Rule::ask_tool_class(Layer::Floor, ToolClass::Shell, "hard"))
        .build()
        .expect("compile");
    let mut session = SessionScope::new("s");
    session.overlay = ModeOverlay::AcceptAll;
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("ls"),
        &json!({"command": "ls"}),
    );
    assert_eq!(decision.effect, Effect::Ask);
}

#[test]
fn session_grant_does_not_lift_deployment_deny() {
    let policy = PolicyBuilder::preset(Preset::Standard)
        .rule(Rule::deny_tool_class(
            Layer::Deployment,
            ToolClass::Shell,
            "no-shell",
        ))
        .rule(Rule::allow_tool_class(
            Layer::Session,
            ToolClass::Shell,
            "grant",
        ))
        .build()
        .expect("compile");
    let session = SessionScope::new("s");
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("ls"),
        &json!({"command": "ls"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
}

#[test]
fn production_refuses_danger_and_noop() {
    let danger = PolicyBuilder::preset(Preset::Danger).production(true);
    assert!(matches!(
        danger.build(),
        Err(PolicyError::DangerRefusedInProduction)
    ));
    let noop = PolicyBuilder::preset(Preset::Locked)
        .sandbox(SandboxRequirement::Noop)
        .production(true);
    assert!(matches!(
        noop.build(),
        Err(PolicyError::NoopRefusedInProduction)
    ));

    let compiled = PolicyBuilder::preset(Preset::Danger)
        .build()
        .expect("workstation danger");
    let mut session = SessionScope::new("bot");
    session.production = true;
    let decision = decide(
        &compiled,
        &Principal::bot("deploy", BotOrigin::UnattendedPost),
        ApprovalWaiter::Absent,
        &session,
        bash("ls"),
        &json!({"command": "ls"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
    assert_eq!(
        decision.reason.code(),
        caliche_policy::ReasonCode::ProductionUnsafe
    );
}

#[test]
fn windows_unattended_requires_docker_or_wsl2() {
    assert!(!HostCapability::WindowsJobOnly.unattended_shell_supported());
    assert!(HostCapability::WindowsDockerOrWsl2.unattended_shell_supported());
    assert!(!HostCapability::Noop.unattended_shell_supported());
}

struct CountingAnalyzer {
    calls: AtomicUsize,
}

impl ShellAnalyzer for CountingAnalyzer {
    fn dialect(&self) -> ShellDialect {
        ShellDialect::Posix
    }

    fn analyze(&self, _command: &str) -> Result<ShellPlan, PolicyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ShellPlan {
            opaque: true,
            segment_count: 0,
            denied: false,
        })
    }
}

#[test]
fn shell_analyzer_is_not_called_when_the_facet_is_off() {
    let trusted = PolicyBuilder::preset(Preset::Trusted)
        .build()
        .expect("trusted");
    let counter = CountingAnalyzer {
        calls: AtomicUsize::new(0),
    };
    assert!(trusted.analyze_shell(&counter, "rm -rf /").is_none());
    assert_eq!(counter.calls.load(Ordering::SeqCst), 0);

    let stub = StubShellAnalyzer::new(ShellDialect::PowerShell);
    assert!(
        trusted
            .analyze_shell(&stub, "Remove-Item -Recurse ~")
            .is_none()
    );

    let with_rule = PolicyBuilder::preset(Preset::Standard)
        .rule(Rule::shell_content(
            Layer::Deployment,
            ShellDialect::Posix,
            Effect::Deny,
            "content",
        ))
        .build()
        .expect("content rule");
    assert!(
        with_rule
            .active_facets()
            .contains(caliche_policy::ActiveFacets::SHELL_POSIX)
    );
    assert!(
        with_rule
            .active_facets()
            .contains(caliche_policy::ActiveFacets::SHELL_PWSH)
    );
    assert!(
        with_rule
            .active_facets()
            .contains(caliche_policy::ActiveFacets::SHELL_CMD)
    );
    assert!(
        !trusted
            .active_facets()
            .contains(caliche_policy::ActiveFacets::SHELL_PWSH)
    );
    let plan = with_rule
        .analyze_shell(&counter, "git status && rm -rf ~")
        .expect("facet on");
    assert!(plan.expect("stub").opaque);
    assert_eq!(counter.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn opaque_shell_is_not_allowed() {
    let policy = PolicyBuilder::preset(Preset::Trusted)
        .build()
        .expect("trusted");
    let mut session = SessionScope::new("s");
    session.overlay = ModeOverlay::AcceptAll;
    let action = Action::tool(
        ToolClass::Shell,
        "bash",
        ToolFacts::Shell {
            dialect: ShellDialect::Posix,
            plan: Some(ShellPlan {
                opaque: true,
                segment_count: 0,
                denied: false,
            }),
        },
    );
    let asked = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        action,
        &json!({"command": "bash -c 'rm -rf ~'"}),
    );
    assert_eq!(asked.effect, Effect::Ask);
    assert_eq!(asked.reason.code(), caliche_policy::ReasonCode::OpaqueShell);
}

struct DeploymentFile;

impl PolicySource for DeploymentFile {
    fn layer(&self) -> Layer {
        Layer::Deployment
    }

    fn load(&self) -> Result<RuleSet, PolicyError> {
        Ok(RuleSet {
            rules: vec![Rule::deny_tool_class(
                Layer::Deployment,
                ToolClass::CodeExec,
                "from-source",
            )],
        })
    }
}

struct FloorSource;

impl PolicySource for FloorSource {
    fn layer(&self) -> Layer {
        Layer::Floor
    }

    fn load(&self) -> Result<RuleSet, PolicyError> {
        Ok(RuleSet::new())
    }
}

#[test]
fn policy_source_cannot_supply_the_floor_or_decide() {
    assert!(matches!(
        ingest_source(&FloorSource),
        Err(PolicyError::SourceCannotSupplyFloor)
    ));
    let rules = ingest_source(&DeploymentFile).expect("deployment source");
    assert_eq!(rules.rules.len(), 1);
    let policy = PolicyBuilder::preset(Preset::Trusted)
        .rule(rules.rules.into_iter().next().expect("rule"))
        .build()
        .expect("compile");
    let session = SessionScope::new("s");
    let decision = decide(
        &policy,
        &human(),
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        Action::tool(ToolClass::CodeExec, "execute_python", ToolFacts::Opaque),
        &json!({"code": "print(1)"}),
    );
    assert_eq!(decision.effect, Effect::Deny);
    let principal = human();
    let raw = json!({"code": "print(1)"});
    let request = Request::new(
        &principal,
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        Action::tool(ToolClass::CodeExec, "execute_python", ToolFacts::Opaque),
        &raw,
    );
    let sealed: &dyn Decide = &policy;
    assert_eq!(Decide::decide(sealed, &request).effect, Effect::Deny);
}

#[test]
fn shadow_compare_keeps_the_live_effect() {
    assert!(shadow_enabled_value("1"));
    assert!(shadow_enabled_value(" true "));
    assert!(!shadow_enabled_value("0"));
    let shadow = evaluate_shadow(
        "fs",
        &json!({"op": "read", "path": "/Users/dev/.codex/auth.json"}),
    );
    assert_eq!(shadow.effect, Effect::Deny);
    let comparison = compare_shadow(Effect::Allow, &shadow);
    assert!(comparison.differs);
    assert_eq!(comparison.live, Effect::Allow);
    assert_eq!(comparison.shadow, Effect::Deny);

    let ordinary = evaluate_shadow("bash", &json!({"command": "ls"}));
    assert_eq!(ordinary.effect, Effect::Ask);
    let same = compare_shadow(Effect::Ask, &ordinary);
    assert!(!same.differs);
    assert_eq!(same.live, Effect::Ask);

    let userinfo = evaluate_shadow(
        "http",
        &json!({"url": "http://user:secret@169.254.169.254/latest/meta-data"}),
    );
    assert_eq!(userinfo.effect, Effect::Deny);
}

#[test]
fn bad_glob_is_a_load_error() {
    let built = PolicyBuilder::preset(Preset::Standard).trust_project_allows(["repo/[a]"]);
    assert!(matches!(built.build(), Err(PolicyError::BadGlob(_))));
}

#[test]
fn sealed_decide_matches_inherent_method() {
    let policy = PolicyBuilder::preset(Preset::Locked)
        .build()
        .expect("locked");
    let session = SessionScope::new("s");
    let principal = human();
    let raw = json!({"command": "ls"});
    let request = Request::new(
        &principal,
        ApprovalWaiter::Present(HumanChannel::CliTty),
        &session,
        bash("ls"),
        &raw,
    );
    assert_eq!(
        Decide::decide(&policy, &request).effect,
        policy.decide(&request).effect
    );
    assert_eq!(Decide::decide(&policy, &request).effect, Effect::Deny);
}
