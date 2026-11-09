//! Deployment `policy.toml`.
//!
//! Version 1 names a preset, an optional default effect, a sandbox
//! requirement, and tool rules. Unknown keys are a load error. This file does
//! not replace the live tool gate. That gate stays on [`Preset::Trusted`].

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::action::ToolClass;
use crate::effect::Effect;
use crate::error::PolicyError;
use crate::policy::{CompiledPolicy, PolicyBuilder};
use crate::posture::{HostCapability, Layer, Preset, SandboxRequirement, ShellDialect};
use crate::rule::{Matcher, Rule, RuleSource};

/// One deployment policy file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyDocument {
    /// Must be 1.
    pub version: u64,
    /// `locked`, `review`, `standard`, `trusted`, or `danger`.
    pub preset: String,
    /// Overrides the preset default when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    /// Production profile. Refuses `danger` and a noop sandbox.
    #[serde(default, skip_serializing_if = "is_false")]
    pub production: bool,
    /// Sandbox obligation override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxDocument>,
    /// Deployment tool rules. A rule sets `class` or `name`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolRuleDocument>,
}

/// `[sandbox]` table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxDocument {
    /// `none`, `workspace_write`, `workspace-write`, `required`, or `noop`.
    pub require: String,
}

/// One `[[tools]]` rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRuleDocument {
    /// Tool family. Mutually exclusive with [`Self::name`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// Exact tool name. Mutually exclusive with [`Self::class`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `allow`, `ask`, or `deny`.
    pub effect: String,
}

/// One row in the Permissions table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveRule {
    /// Rule id.
    pub id: String,
    /// Layer name.
    pub layer: String,
    /// Effect name.
    pub effect: String,
    /// Provenance (`builtin:…` or `path:line`).
    pub source: String,
    /// Matcher label.
    pub matcher: String,
}

/// Serializable view shared by `policy show` and the Permissions page.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectivePolicy {
    /// False when the path does not exist yet.
    pub file_present: bool,
    /// Path this view was loaded from.
    pub source_path: String,
    /// Document that a save writes back.
    pub document: PolicyDocument,
    /// Compiled preset name.
    pub preset: String,
    /// Compiled default effect.
    pub default_effect: String,
    /// Compiled sandbox requirement.
    pub sandbox: String,
    /// Production flag from the document.
    pub production: bool,
    /// Obligation labels.
    pub obligations_fs: String,
    /// Obligation labels.
    pub obligations_net: String,
    /// Obligation labels.
    pub obligations_proc: String,
    /// Host label used for the capability panel.
    pub host: String,
    /// Text from [`crate::policy_show`].
    pub capability_panel: String,
    /// Floor, deployment, profile, project, and session rules.
    pub rules: Vec<EffectiveRule>,
    /// Session grants. Empty until a grant is recorded on the snapshot.
    pub grants: Vec<EffectiveRule>,
    /// Pending approval ids. The file does not store them.
    pub pending: Vec<String>,
    /// Preset the live tool gate uses. This file does not change it.
    pub live_gate: String,
}

impl PolicyDocument {
    /// New-deploy document. Preset is [`Preset::Locked`].
    #[must_use]
    pub fn new_deploy() -> Self {
        Self {
            version: 1,
            preset: Preset::Locked.as_str().to_owned(),
            default: None,
            production: false,
            sandbox: None,
            tools: Vec::new(),
        }
    }

    /// Compile this document.
    ///
    /// `source_path` is stored on tool-rule provenance.
    ///
    /// # Errors
    ///
    /// Returns a [`PolicyError`] for an unknown preset, effect, sandbox, tool
    /// target, or a production refusal.
    pub fn compile(&self, source_path: &str) -> Result<CompiledPolicy, PolicyError> {
        if self.version != 1 {
            return Err(PolicyError::UnknownPolicyVersion(self.version));
        }
        let preset = crate::parse_preset(&self.preset)?;
        let mut builder = PolicyBuilder::preset(preset)
            .policy_version(self.version)
            .production(self.production);
        if let Some(default) = &self.default {
            builder = builder.deployment_default(crate::parse_effect(default)?);
        }
        if let Some(sandbox) = &self.sandbox {
            builder = builder.sandbox(parse_sandbox(&sandbox.require)?);
        }
        for (index, tool) in self.tools.iter().enumerate() {
            builder = builder.rule(tool.to_rule(source_path, index)?);
        }
        builder.build()
    }

    /// Serialize this document as TOML.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError::InvalidPolicy`] when encoding fails.
    pub fn to_toml_text(&self) -> Result<String, PolicyError> {
        toml::to_string_pretty(self).map_err(|err| PolicyError::InvalidPolicy(err.to_string()))
    }
}

impl ToolRuleDocument {
    fn to_rule(&self, source_path: &str, index: usize) -> Result<Rule, PolicyError> {
        let effect = crate::parse_effect(&self.effect)?;
        let matcher = tool_matcher(self)?;
        Ok(Rule {
            id: format!("deployment.tool.{index}"),
            layer: Layer::Deployment,
            effect,
            hard: effect == Effect::Deny,
            source: RuleSource::File {
                path: source_path.to_owned(),
                line: 1,
            },
            matcher,
        })
    }
}

/// Parse version-1 TOML.
///
/// # Errors
///
/// Returns [`PolicyError::InvalidPolicy`] for syntax errors and unknown keys,
/// and [`PolicyError::UnknownPolicyVersion`] when `version` is not 1.
pub fn parse_policy_toml(text: &str) -> Result<PolicyDocument, PolicyError> {
    let document: PolicyDocument =
        toml::from_str(text).map_err(|err| PolicyError::InvalidPolicy(err.to_string()))?;
    if document.version != 1 {
        return Err(PolicyError::UnknownPolicyVersion(document.version));
    }
    Ok(document)
}

/// Build the shared view for a compiled document.
///
/// # Errors
///
/// Returns the same errors as [`PolicyDocument::compile`].
pub fn view_policy(
    document: &PolicyDocument,
    source_path: &str,
    file_present: bool,
    host: HostCapability,
) -> Result<EffectivePolicy, PolicyError> {
    let policy = document.compile(source_path)?;
    Ok(effective_view(
        document,
        &policy,
        source_path,
        file_present,
        host,
    ))
}

/// Load `path`. A missing file returns the locked new-deploy view and does not
/// write it.
///
/// # Errors
///
/// Returns a [`PolicyError`] when the file exists but is not valid version-1
/// policy.
pub fn open_policy_file(path: &Path, host: HostCapability) -> Result<EffectivePolicy, PolicyError> {
    let source_path = path.display().to_string();
    if !path.exists() {
        let document = PolicyDocument::new_deploy();
        return view_policy(&document, &source_path, false, host);
    }
    let text =
        fs::read_to_string(path).map_err(|err| PolicyError::InvalidPolicy(err.to_string()))?;
    let document = parse_policy_toml(&text)?;
    view_policy(&document, &source_path, true, host)
}

/// Compile `document`, create parent directories, and write TOML.
///
/// # Errors
///
/// Returns a [`PolicyError`] when compile fails or the write fails. A compile
/// error leaves the previous file in place.
pub fn write_policy_file(
    path: &Path,
    document: &PolicyDocument,
    host: HostCapability,
) -> Result<EffectivePolicy, PolicyError> {
    let source_path = path.display().to_string();
    let view = view_policy(document, &source_path, true, host)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| PolicyError::PolicyWrite(err.to_string()))?;
    }
    let text = document.to_toml_text()?;
    fs::write(path, text).map_err(|err| PolicyError::PolicyWrite(err.to_string()))?;
    Ok(view)
}

/// `CALICHE_POLICY_PATH`, else `$HOME/.config/caliche/policy.toml`.
///
/// When neither home variable is set, the path is `policy.toml` in the
/// process working directory.
#[must_use]
pub fn default_policy_path() -> PathBuf {
    if let Ok(raw) = std::env::var("CALICHE_POLICY_PATH") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    match home {
        Ok(home) if !home.trim().is_empty() => PathBuf::from(home.trim())
            .join(".config")
            .join("caliche")
            .join("policy.toml"),
        _ => PathBuf::from("policy.toml"),
    }
}

/// Parse a host label.
///
/// # Errors
///
/// Returns [`PolicyError::UnknownHost`] for an unknown token.
pub fn parse_host(value: &str) -> Result<HostCapability, PolicyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "unspecified" => Ok(HostCapability::Unspecified),
        "macos_seatbelt" => Ok(HostCapability::MacOsSeatbelt),
        "linux_landlock" => Ok(HostCapability::LinuxLandlock {
            network_enforceable: true,
        }),
        "linux_landlock_no_network" => Ok(HostCapability::LinuxLandlock {
            network_enforceable: false,
        }),
        "windows_job_only" => Ok(HostCapability::WindowsJobOnly),
        "windows_docker_or_wsl2" => Ok(HostCapability::WindowsDockerOrWsl2),
        "docker" => Ok(HostCapability::Docker),
        "noop" => Ok(HostCapability::Noop),
        other => Err(PolicyError::UnknownHost(other.to_owned())),
    }
}

/// Host label for the Permissions page when the caller does not name one.
///
/// This does not change [`crate::CompiledPolicy::decide`]. This crate does
/// not probe Landlock. Linux stays `network_enforceable: false` here.
/// [`host_for_process`] applies a caller probe.
#[must_use]
pub fn host_for_this_process() -> HostCapability {
    if cfg!(windows) {
        HostCapability::WindowsJobOnly
    } else if cfg!(target_os = "macos") {
        HostCapability::MacOsSeatbelt
    } else if cfg!(target_os = "linux") {
        HostCapability::LinuxLandlock {
            network_enforceable: false,
        }
    } else {
        HostCapability::Unspecified
    }
}

/// Same host as [`host_for_this_process`], with a Landlock network probe.
///
/// `network_enforceable` changes the result only on Linux. Other hosts keep
/// their own label.
#[must_use]
pub fn host_for_process(network_enforceable: bool) -> HostCapability {
    match host_for_this_process() {
        HostCapability::LinuxLandlock { .. } => HostCapability::LinuxLandlock {
            network_enforceable,
        },
        other => other,
    }
}

/// Text form of [`EffectivePolicy`] for `policy show`.
#[must_use]
pub fn render_effective(policy: &EffectivePolicy) -> String {
    let mut text = policy.capability_panel.clone();
    text.push_str("source: ");
    text.push_str(&policy.source_path);
    text.push('\n');
    text.push_str("file: ");
    text.push_str(if policy.file_present {
        "present"
    } else {
        "absent"
    });
    text.push('\n');
    text.push_str("live tool gate: ");
    text.push_str(&policy.live_gate);
    text.push('\n');
    push_rules(&mut text, "rules", &policy.rules);
    push_rules(&mut text, "grants", &policy.grants);
    text.push_str("pending: ");
    if policy.pending.is_empty() {
        text.push_str("none");
    } else {
        text.push_str(&policy.pending.join(", "));
    }
    text.push('\n');
    text
}

fn push_rules(text: &mut String, heading: &str, rules: &[EffectiveRule]) {
    text.push_str(heading);
    text.push_str(":\n");
    if rules.is_empty() {
        text.push_str("  none\n");
        return;
    }
    for rule in rules {
        text.push_str("  ");
        text.push_str(&rule.layer);
        text.push(' ');
        text.push_str(&rule.effect);
        text.push(' ');
        text.push_str(&rule.source);
        text.push(' ');
        text.push_str(&rule.matcher);
        text.push('\n');
    }
}

fn effective_view(
    document: &PolicyDocument,
    policy: &CompiledPolicy,
    source_path: &str,
    file_present: bool,
    host: HostCapability,
) -> EffectivePolicy {
    let obligations = policy.obligations();
    let preset = policy.preset().map_or("none", Preset::as_str).to_owned();
    let default_effect = policy
        .deployment_default()
        .map_or("deny", Effect::as_str)
        .to_owned();
    EffectivePolicy {
        file_present,
        source_path: source_path.to_owned(),
        document: document.clone(),
        preset,
        default_effect,
        sandbox: policy.sandbox().as_str().to_owned(),
        production: policy.production(),
        obligations_fs: fs_name(obligations.fs).to_owned(),
        obligations_net: net_name(obligations.net).to_owned(),
        obligations_proc: proc_name(obligations.proc).to_owned(),
        host: host.label().to_owned(),
        capability_panel: crate::policy_show(policy, host),
        rules: policy
            .layer_rules()
            .into_iter()
            .flat_map(|(layer, rules)| rules.iter().map(move |rule| rule_row(layer, rule)))
            .collect(),
        grants: policy
            .grant_rules()
            .iter()
            .map(|rule| rule_row(rule.layer, rule))
            .collect(),
        pending: Vec::new(),
        live_gate: Preset::Trusted.as_str().to_owned(),
    }
}

fn rule_row(layer: Layer, rule: &Rule) -> EffectiveRule {
    EffectiveRule {
        id: rule.id.clone(),
        layer: layer_name(layer).to_owned(),
        effect: rule.effect.as_str().to_owned(),
        source: source_label(&rule.source),
        matcher: matcher_label(&rule.matcher),
    }
}

fn tool_matcher(tool: &ToolRuleDocument) -> Result<Matcher, PolicyError> {
    match (&tool.class, &tool.name) {
        (Some(class), None) => Ok(Matcher::ToolClass(parse_tool_class(class)?)),
        (None, Some(name)) if !name.trim().is_empty() => Ok(Matcher::ToolName(name.clone())),
        _ => Err(PolicyError::ToolRuleTarget),
    }
}

fn parse_tool_class(value: &str) -> Result<ToolClass, PolicyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fs" => Ok(ToolClass::Fs),
        "shell" => Ok(ToolClass::Shell),
        "http" => Ok(ToolClass::Http),
        "web_fetch" => Ok(ToolClass::WebFetch),
        "mcp" => Ok(ToolClass::Mcp),
        "code_exec" => Ok(ToolClass::CodeExec),
        "patch" => Ok(ToolClass::Patch),
        "git" => Ok(ToolClass::Git),
        "sql" => Ok(ToolClass::Sql),
        "workspace" => Ok(ToolClass::Workspace),
        "other" => Ok(ToolClass::Other),
        other => Err(PolicyError::UnknownToolClass(other.to_owned())),
    }
}

fn parse_sandbox(value: &str) -> Result<SandboxRequirement, PolicyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(SandboxRequirement::None),
        "workspace_write" | "workspace-write" => Ok(SandboxRequirement::WorkspaceWrite),
        "required" => Ok(SandboxRequirement::Required),
        "noop" => Ok(SandboxRequirement::Noop),
        other => Err(PolicyError::UnknownSandbox(other.to_owned())),
    }
}

fn source_label(source: &RuleSource) -> String {
    match source {
        RuleSource::Builtin { id } => format!("builtin:{id}"),
        RuleSource::File { path, line } => format!("{path}:{line}"),
        RuleSource::Env { name } => format!("env:{name}"),
    }
}

fn matcher_label(matcher: &Matcher) -> String {
    match matcher {
        Matcher::ToolClass(class) => format!("tool_class:{}", class_name(*class)),
        Matcher::ToolName(name) => format!("tool_name:{name}"),
        Matcher::NonHumanApproval => "non_human_approval".to_owned(),
        Matcher::VendorAuth => "vendor_auth".to_owned(),
        Matcher::VendorSubscription => "vendor_subscription".to_owned(),
        Matcher::Ssrf => "ssrf".to_owned(),
        Matcher::CheckerError => "checker_error".to_owned(),
        Matcher::ShellContent { dialect } => {
            format!("shell_content:{}", dialect_name(*dialect))
        }
    }
}

const fn class_name(class: ToolClass) -> &'static str {
    match class {
        ToolClass::Fs => "fs",
        ToolClass::Shell => "shell",
        ToolClass::Http => "http",
        ToolClass::WebFetch => "web_fetch",
        ToolClass::Mcp => "mcp",
        ToolClass::CodeExec => "code_exec",
        ToolClass::Patch => "patch",
        ToolClass::Git => "git",
        ToolClass::Sql => "sql",
        ToolClass::Workspace => "workspace",
        ToolClass::Other => "other",
    }
}

const fn dialect_name(dialect: ShellDialect) -> &'static str {
    match dialect {
        ShellDialect::Posix => "posix",
        ShellDialect::PowerShell => "powershell",
        ShellDialect::Cmd => "cmd",
    }
}

const fn layer_name(layer: Layer) -> &'static str {
    match layer {
        Layer::Floor => "floor",
        Layer::Deployment => "deployment",
        Layer::Profile => "profile",
        Layer::Project => "project",
        Layer::Session => "session",
    }
}

const fn fs_name(fs: crate::posture::FsConfine) -> &'static str {
    match fs {
        crate::posture::FsConfine::None => "none",
        crate::posture::FsConfine::WorkspaceWrite => "workspace_write",
        crate::posture::FsConfine::ReadOnly => "read_only",
    }
}

const fn net_name(net: crate::posture::NetConfine) -> &'static str {
    match net {
        crate::posture::NetConfine::None => "none",
        crate::posture::NetConfine::DenyAll => "deny_all",
        crate::posture::NetConfine::AllowPublic => "allow_public",
    }
}

const fn proc_name(proc: crate::posture::ProcConfine) -> &'static str {
    match proc {
        crate::posture::ProcConfine::None => "none",
        crate::posture::ProcConfine::KillOnDrop => "kill_on_drop",
    }
}

const fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, HttpMethod, ToolFacts};
    use crate::effect::Effect;
    use crate::posture::ModeOverlay;
    use crate::principal::{ApprovalWaiter, HumanChannel, Principal};
    use crate::reason::ReasonCode;
    use crate::request::{Request, SessionScope};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn shell_effect(policy: &CompiledPolicy) -> Effect {
        let action = Action::tool(
            ToolClass::Shell,
            "bash",
            ToolFacts::Shell {
                dialect: ShellDialect::Posix,
                plan: None,
            },
        );
        decide(
            policy,
            action,
            &serde_json::json!({"command": "git status"}),
        )
    }

    fn decide(policy: &CompiledPolicy, action: Action, args: &serde_json::Value) -> Effect {
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let mut session = SessionScope::new("doc");
        session.overlay = ModeOverlay::Default;
        policy
            .decide(&Request::new(
                &principal,
                ApprovalWaiter::Present(HumanChannel::CliTty),
                &session,
                action,
                args,
            ))
            .effect
    }

    #[test]
    fn presets_select_the_default_effect() {
        let standard = parse_policy_toml("version = 1\npreset = \"standard\"\n").expect("parse");
        let standard_policy = standard.compile("policy.toml").expect("compile");
        assert_eq!(shell_effect(&standard_policy), Effect::Ask);

        let locked = parse_policy_toml("version = 1\npreset = \"locked\"\n").expect("parse");
        assert_eq!(
            shell_effect(&locked.compile("policy.toml").expect("compile")),
            Effect::Deny
        );

        let trusted = parse_policy_toml("version = 1\npreset = \"trusted\"\n").expect("parse");
        assert_eq!(
            shell_effect(&trusted.compile("policy.toml").expect("compile")),
            Effect::Allow
        );
    }

    #[test]
    fn default_override_and_tool_deny() {
        let text = r#"
version = 1
preset = "trusted"
default = "ask"

[[tools]]
class = "http"
effect = "deny"
"#;
        let document = parse_policy_toml(text).expect("parse");
        let policy = document.compile("policy.toml").expect("compile");
        assert_eq!(shell_effect(&policy), Effect::Ask);
        let action = Action::tool(
            ToolClass::Http,
            "http",
            ToolFacts::Http {
                method: HttpMethod::Get,
                host: "example.com".to_owned(),
            },
        );
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let session = SessionScope::new("doc");
        let args = serde_json::json!({"url": "https://example.com"});
        let decision = policy.decide(&Request::new(
            &principal,
            ApprovalWaiter::Present(HumanChannel::CliTty),
            &session,
            action,
            &args,
        ));
        assert_eq!(decision.effect, Effect::Deny);
        assert_eq!(decision.reason.code(), ReasonCode::DeploymentDeny);
    }

    #[test]
    fn unknown_preset_key_version_and_tool_target_fail() {
        let unknown_preset = parse_policy_toml("version = 1\npreset = \"yolo\"\n").expect("parse");
        assert!(matches!(
            unknown_preset.compile("policy.toml"),
            Err(PolicyError::UnknownPreset(_))
        ));
        assert!(matches!(
            parse_policy_toml("version = 2\npreset = \"locked\"\n"),
            Err(PolicyError::UnknownPolicyVersion(2))
        ));
        assert!(matches!(
            parse_policy_toml("version = 1\npreset = \"locked\"\nextra = 1\n"),
            Err(PolicyError::InvalidPolicy(_))
        ));
        assert!(matches!(
            parse_policy_toml("version = 1\npreset = \"locked\"\n[shell]\nrules = []\n"),
            Err(PolicyError::InvalidPolicy(_))
        ));
        let both = parse_policy_toml(
            "version = 1\npreset = \"locked\"\n[[tools]]\nclass = \"http\"\nname = \"curl\"\neffect = \"deny\"\n",
        )
        .expect("parse");
        assert!(matches!(
            both.compile("policy.toml"),
            Err(PolicyError::ToolRuleTarget)
        ));
        let bad_class = parse_policy_toml(
            "version = 1\npreset = \"locked\"\n[[tools]]\nclass = \"browser\"\neffect = \"deny\"\n",
        )
        .expect("parse");
        assert!(matches!(
            bad_class.compile("policy.toml"),
            Err(PolicyError::UnknownToolClass(_))
        ));
    }

    #[test]
    fn production_refuses_danger_and_noop() {
        let danger = parse_policy_toml("version = 1\npreset = \"danger\"\nproduction = true\n")
            .expect("parse");
        assert!(matches!(
            danger.compile("policy.toml"),
            Err(PolicyError::DangerRefusedInProduction)
        ));
        let noop = parse_policy_toml(
            "version = 1\npreset = \"locked\"\nproduction = true\n[sandbox]\nrequire = \"noop\"\n",
        )
        .expect("parse");
        assert!(matches!(
            noop.compile("policy.toml"),
            Err(PolicyError::NoopRefusedInProduction)
        ));
    }

    #[test]
    fn host_probe_changes_only_the_linux_landlock_flag() {
        let host = host_for_process(true);
        if cfg!(target_os = "linux") {
            assert_eq!(
                host,
                HostCapability::LinuxLandlock {
                    network_enforceable: true
                }
            );
        } else {
            assert!(!matches!(host, HostCapability::LinuxLandlock { .. }));
        }
        assert_eq!(host_for_process(false), host_for_this_process());
    }

    #[test]
    fn windows_job_panel_and_round_trip() {
        let document = parse_policy_toml("version = 1\npreset = \"standard\"\n").expect("parse");
        let view = view_policy(
            &document,
            "policy.toml",
            true,
            HostCapability::WindowsJobOnly,
        )
        .expect("view");
        assert!(view.capability_panel.contains("network not sandboxed"));
        assert!(
            view.capability_panel
                .contains("Ask under standard, Deny under locked")
        );
        assert_eq!(view.live_gate, "trusted");
        assert!(view.pending.is_empty());
        assert!(view.grants.is_empty());
        assert!(view.rules.iter().any(|rule| rule.layer == "floor"));
        let again = parse_policy_toml(&document.to_toml_text().expect("toml")).expect("parse");
        assert_eq!(again, document);
        let text = render_effective(&view);
        assert!(text.contains("live tool gate: trusted"));
        assert!(text.contains("file: present"));
    }

    #[test]
    fn missing_file_is_locked_and_write_round_trips() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("caliche-policy-{stamp}"));
        let path = dir.join("policy.toml");
        let missing = open_policy_file(&path, HostCapability::MacOsSeatbelt).expect("missing");
        assert!(!missing.file_present);
        assert_eq!(missing.preset, "locked");
        assert!(!missing.capability_panel.contains("network not sandboxed"));
        let mut document = missing.document;
        document.preset = "review".to_owned();
        write_policy_file(&path, &document, HostCapability::Unspecified).expect("write");
        let loaded = open_policy_file(&path, HostCapability::Unspecified).expect("load");
        assert!(loaded.file_present);
        assert_eq!(loaded.preset, "review");
        let _ = fs::remove_dir_all(&dir);
    }
}
