//! Compiled snapshot and the only `decide` implementation.

use crate::action::{Action, FsOp, PathPlace, ToolClass, ToolFacts};
use crate::analyzer::ShellAnalyzer;
use crate::effect::Effect;
use crate::error::PolicyError;
use crate::ids::{ActionBinding, ActionId};
use crate::matching::{best_match, content_rule_matched, shell_plan_opaque};
use crate::posture::{
    ActiveFacets, Layer, ModeOverlay, Obligations, Preset, SandboxRequirement, ShellDialect,
};
use crate::principal::{ApprovalWaiter, Principal};
use crate::reason::Reason;
use crate::request::Request;
use crate::rule::{Matcher, Rule, RuleRef, RuleSource};

/// Outcome of [`Decide::decide`]. Plugins cannot construct this by implementing
/// the trait; only [`CompiledPolicy`] can.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    /// Winning effect. Deny beats ask and allow.
    pub effect: Effect,
    /// Why.
    pub reason: Reason,
    /// Winning rule, when one was named.
    pub rule: Option<RuleRef>,
    /// Tool-plus-args hash when the effect is ask, a content rule matched, or
    /// audit is on.
    pub binding: Option<ActionBinding>,
    /// Sandbox obligations for this snapshot. Recorded here. `decide` also
    /// raises shell and code execution when a named host cannot meet them.
    pub obligations: Obligations,
    /// Action identity. Not the argument hash.
    pub action_id: ActionId,
}

/// Immutable policy snapshot.
#[derive(Clone, Debug)]
pub struct CompiledPolicy {
    preset: Option<Preset>,
    deployment_default: Option<Effect>,
    production: bool,
    sandbox: SandboxRequirement,
    obligations: Obligations,
    facets: ActiveFacets,
    version: u64,
    floor: Vec<Rule>,
    deployment: Vec<Rule>,
    profile: Vec<Rule>,
    project: Vec<Rule>,
    session_narrow: Vec<Rule>,
    grants: Vec<Rule>,
    project_allows: Vec<Rule>,
}

impl CompiledPolicy {
    /// Decide this request.
    ///
    /// This is the inherent method. The [`Decide`] trait is sealed so no other
    /// type in a downstream crate can implement it.
    #[must_use]
    pub fn decide(&self, request: &Request<'_>) -> Decision {
        evaluate(self, request)
    }

    /// Preset, when the snapshot was built from one.
    #[must_use]
    pub const fn preset(&self) -> Option<Preset> {
        self.preset
    }

    /// Whether the deployment was compiled as a production profile.
    #[must_use]
    pub const fn production(&self) -> bool {
        self.production
    }

    /// Deployment default. `None` means a missing policy, which denies.
    #[must_use]
    pub const fn deployment_default(&self) -> Option<Effect> {
        self.deployment_default
    }

    /// Active facets. Shell bits are clear unless a content rule exists.
    #[must_use]
    pub const fn active_facets(&self) -> ActiveFacets {
        self.facets
    }

    /// Obligations recorded for the preset.
    #[must_use]
    pub const fn obligations(&self) -> Obligations {
        self.obligations
    }

    /// Sandbox requirement.
    #[must_use]
    pub const fn sandbox(&self) -> SandboxRequirement {
        self.sandbox
    }

    /// Policy version mixed into bindings.
    #[must_use]
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Rules in layer order, excluding session grants.
    pub(crate) fn layer_rules(&self) -> [(Layer, &[Rule]); 5] {
        [
            (Layer::Floor, self.floor.as_slice()),
            (Layer::Deployment, self.deployment.as_slice()),
            (Layer::Profile, self.profile.as_slice()),
            (Layer::Project, self.project.as_slice()),
            (Layer::Session, self.session_narrow.as_slice()),
        ]
    }

    /// Session grants recorded on this snapshot.
    #[must_use]
    pub(crate) fn grant_rules(&self) -> &[Rule] {
        self.grants.as_slice()
    }

    /// Project allow rules accepted because the deployment opted in.
    ///
    /// They are recorded and cannot change the effect: a project allow does
    /// not lift a deployment deny and does not lower an ask.
    #[must_use]
    pub const fn recorded_project_allows(&self) -> usize {
        self.project_allows.len()
    }

    /// Run a shell analyzer only when that dialect's content facet is set.
    ///
    /// Returns `None` without calling `analyzer` when the bit is clear.
    pub fn analyze_shell(
        &self,
        analyzer: &impl ShellAnalyzer,
        command: &str,
    ) -> Option<Result<crate::action::ShellPlan, PolicyError>> {
        if !self.facets.contains(analyzer.dialect().facet()) {
            return None;
        }
        Some(analyzer.analyze(command))
    }
}

/// The only decider. The seal is crate-private, so plugins cannot implement this.
pub trait Decide: seal::Sealed {
    /// Authorize `request`.
    fn decide(&self, request: &Request<'_>) -> Decision;
}

mod seal {
    pub trait Sealed {}
}

impl seal::Sealed for CompiledPolicy {}

impl Decide for CompiledPolicy {
    fn decide(&self, request: &Request<'_>) -> Decision {
        Self::decide(self, request)
    }
}

/// Builds a [`CompiledPolicy`]. Floor rules are always included.
#[derive(Clone, Debug)]
pub struct PolicyBuilder {
    preset: Option<Preset>,
    explicit_default: Option<Effect>,
    production: bool,
    sandbox: Option<SandboxRequirement>,
    trust_project_allows: Vec<String>,
    rules: Vec<Rule>,
    audit: bool,
    version: u64,
}

impl Default for PolicyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyBuilder {
    /// Empty snapshot. No deployment default, so every action is denied.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            preset: None,
            explicit_default: None,
            production: false,
            sandbox: None,
            trust_project_allows: Vec::new(),
            rules: Vec::new(),
            audit: false,
            version: 1,
        }
    }

    /// Start from a named preset.
    #[must_use]
    pub const fn preset(preset: Preset) -> Self {
        Self::new().with_preset(preset)
    }

    /// Select a preset. The preset's default effect applies unless overridden.
    #[must_use]
    pub const fn with_preset(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Override the deployment default.
    #[must_use]
    pub const fn deployment_default(mut self, effect: Effect) -> Self {
        self.explicit_default = Some(effect);
        self
    }

    /// Mark the deployment as a production profile.
    #[must_use]
    pub const fn production(mut self, production: bool) -> Self {
        self.production = production;
        self
    }

    /// Override the sandbox requirement.
    #[must_use]
    pub const fn sandbox(mut self, sandbox: SandboxRequirement) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Globs of project files whose allow rules may be loaded.
    ///
    /// Loaded allows still cannot lift a deployment deny.
    #[must_use]
    pub fn trust_project_allows(
        mut self,
        globs: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.trust_project_allows
            .extend(globs.into_iter().map(Into::into));
        self
    }

    /// Add a rule. Layer and effect are checked in [`Self::build`].
    #[must_use]
    pub fn rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add a floor rule. Allows are rejected. The rule is hard.
    #[must_use]
    pub fn floor_rule(mut self, mut rule: Rule) -> Self {
        rule.layer = Layer::Floor;
        rule.hard = true;
        self.rules.push(rule);
        self
    }

    /// Hash bindings even when the effect is allow.
    #[must_use]
    pub const fn audit_bindings(mut self, enabled: bool) -> Self {
        self.audit = enabled;
        self
    }

    /// Version mixed into [`ActionBinding`].
    #[must_use]
    pub const fn policy_version(mut self, version: u64) -> Self {
        self.version = version;
        self
    }

    /// Compile.
    ///
    /// # Errors
    ///
    /// Returns a [`PolicyError`] when a lower layer would raise rights, a glob
    /// is invalid, or production refuses `danger` / noop.
    pub fn build(self) -> Result<CompiledPolicy, PolicyError> {
        compile(self)
    }

    pub(crate) fn build_or_missing(self) -> CompiledPolicy {
        self.build().unwrap_or_else(|_| missing_policy())
    }
}

fn missing_policy() -> CompiledPolicy {
    CompiledPolicy {
        preset: None,
        deployment_default: None,
        production: false,
        sandbox: SandboxRequirement::Required,
        obligations: Obligations::required(),
        facets: ActiveFacets::empty(),
        version: 0,
        floor: builtin_floor(),
        deployment: Vec::new(),
        profile: Vec::new(),
        project: Vec::new(),
        session_narrow: Vec::new(),
        grants: Vec::new(),
        project_allows: Vec::new(),
    }
}

fn push_shell_denylist(buckets: &mut Buckets) -> Result<(), PolicyError> {
    for (id, dialect) in [
        ("floor.posix_denylist", ShellDialect::Posix),
        ("floor.pwsh_denylist", ShellDialect::PowerShell),
        ("floor.cmd_denylist", ShellDialect::Cmd),
    ] {
        buckets.push(
            Rule::floor(id, Effect::Deny, true, Matcher::ShellContent { dialect }),
            &[],
        )?;
    }
    Ok(())
}

fn compile(builder: PolicyBuilder) -> Result<CompiledPolicy, PolicyError> {
    let sandbox = builder.sandbox.unwrap_or_else(|| {
        builder
            .preset
            .map_or(SandboxRequirement::Required, Preset::sandbox)
    });
    if builder.production && builder.preset == Some(Preset::Danger) {
        return Err(PolicyError::DangerRefusedInProduction);
    }
    if builder.production && sandbox == SandboxRequirement::Noop {
        return Err(PolicyError::NoopRefusedInProduction);
    }
    for glob in &builder.trust_project_allows {
        validate_glob(glob)?;
    }

    let mut buckets = Buckets::with_floor(builtin_floor());
    if matches!(
        builder.preset,
        Some(Preset::Standard | Preset::Locked | Preset::Review)
    ) {
        push_shell_denylist(&mut buckets)?;
    }
    for rule in builder.rules {
        buckets.push(rule, &builder.trust_project_allows)?;
    }

    let deployment_default = match builder.explicit_default {
        Some(effect) => Some(effect),
        None => builder.preset.map(Preset::default_effect),
    };
    let mut facets = buckets.facets();
    if builder.audit {
        facets = facets.union(ActiveFacets::AUDIT_BINDING);
    }
    let obligations = obligations_for(builder.preset, sandbox);

    Ok(CompiledPolicy {
        preset: builder.preset,
        deployment_default,
        production: builder.production,
        sandbox,
        obligations,
        facets,
        version: builder.version,
        floor: buckets.floor,
        deployment: buckets.deployment,
        profile: buckets.profile,
        project: buckets.project,
        session_narrow: buckets.session_narrow,
        grants: buckets.grants,
        project_allows: buckets.project_allows,
    })
}

struct Buckets {
    floor: Vec<Rule>,
    deployment: Vec<Rule>,
    profile: Vec<Rule>,
    project: Vec<Rule>,
    session_narrow: Vec<Rule>,
    grants: Vec<Rule>,
    project_allows: Vec<Rule>,
}

impl Buckets {
    const fn with_floor(floor: Vec<Rule>) -> Self {
        Self {
            floor,
            deployment: Vec::new(),
            profile: Vec::new(),
            project: Vec::new(),
            session_narrow: Vec::new(),
            grants: Vec::new(),
            project_allows: Vec::new(),
        }
    }

    fn push(&mut self, rule: Rule, trust: &[String]) -> Result<(), PolicyError> {
        match (rule.layer, rule.effect) {
            (Layer::Floor, Effect::Allow) => Err(PolicyError::FloorCannotAllow(rule.id)),
            (Layer::Floor, _) => {
                self.floor.push(rule);
                Ok(())
            }
            (Layer::Deployment, _) => {
                self.deployment.push(rule);
                Ok(())
            }
            (Layer::Profile, Effect::Allow) => Err(PolicyError::ProfileAllowRaisesRights(rule.id)),
            (Layer::Profile, _) => {
                self.profile.push(rule);
                Ok(())
            }
            (Layer::Project, Effect::Allow) => {
                if !project_allow_trusted(trust, &rule)? {
                    return Err(PolicyError::ProjectAllowRaisesRights(rule.id));
                }
                self.project_allows.push(rule);
                Ok(())
            }
            (Layer::Project, _) => {
                self.project.push(rule);
                Ok(())
            }
            (Layer::Session, Effect::Allow) => {
                self.grants.push(rule);
                Ok(())
            }
            (Layer::Session, _) => {
                self.session_narrow.push(rule);
                Ok(())
            }
        }
    }

    fn facets(&self) -> ActiveFacets {
        let mut facets = ActiveFacets::empty();
        for rule in self
            .floor
            .iter()
            .chain(self.deployment.iter())
            .chain(self.profile.iter())
            .chain(self.project.iter())
            .chain(self.session_narrow.iter())
            .chain(self.grants.iter())
            .chain(self.project_allows.iter())
        {
            if let Matcher::ShellContent { dialect } = &rule.matcher {
                facets = facets.union(dialect.facet());
            }
        }
        facets
    }
}

fn builtin_floor() -> Vec<Rule> {
    vec![
        Rule::floor(
            "floor.non_human_approval",
            Effect::Deny,
            true,
            Matcher::NonHumanApproval,
        ),
        Rule::floor("floor.vendor_auth", Effect::Deny, true, Matcher::VendorAuth),
        Rule::floor(
            "floor.byo_credential",
            Effect::Deny,
            true,
            Matcher::VendorSubscription,
        ),
        Rule::floor("floor.ssrf", Effect::Deny, true, Matcher::Ssrf),
        Rule::floor(
            "floor.checker_error",
            Effect::Deny,
            true,
            Matcher::CheckerError,
        ),
    ]
}

fn obligations_for(preset: Option<Preset>, sandbox: SandboxRequirement) -> Obligations {
    match sandbox {
        SandboxRequirement::None | SandboxRequirement::Noop => Obligations::none(),
        SandboxRequirement::WorkspaceWrite => Obligations::workspace_write(),
        SandboxRequirement::Required => {
            preset.map_or_else(Obligations::required, Preset::obligations)
        }
    }
}

fn project_allow_trusted(globs: &[String], rule: &Rule) -> Result<bool, PolicyError> {
    let path = match &rule.source {
        RuleSource::File { path, .. } => path.clone(),
        _ => rule.id.clone(),
    };
    let mut trusted = false;
    for glob in globs {
        validate_glob(glob)?;
        if glob_match(glob, &path) {
            trusted = true;
        }
    }
    Ok(trusted)
}

fn validate_glob(pattern: &str) -> Result<(), PolicyError> {
    if pattern.is_empty() || pattern.contains('[') || pattern.contains(']') {
        Err(PolicyError::BadGlob(pattern.to_owned()))
    } else {
        Ok(())
    }
}

fn glob_match(pattern: &str, text: &str) -> bool {
    glob_bytes(pattern.as_bytes(), text.as_bytes())
}

fn glob_bytes(pattern: &[u8], text: &[u8]) -> bool {
    let mut pattern_index = 0;
    let mut text_index = 0;
    let mut star_pattern = None;
    let mut star_text = 0;
    while text_index < text.len() {
        if pattern_index < pattern.len()
            && (pattern[pattern_index] == b'?' || pattern[pattern_index] == text[text_index])
        {
            pattern_index += 1;
            text_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
            star_pattern = Some(pattern_index);
            star_text = text_index;
            pattern_index += 1;
        } else if let Some(saved) = star_pattern {
            pattern_index = saved + 1;
            star_text += 1;
            text_index = star_text;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

struct Winner {
    effect: Effect,
    hard: bool,
    reason: Reason,
    rule: Option<RuleRef>,
    content: bool,
}

fn evaluate(policy: &CompiledPolicy, request: &Request<'_>) -> Decision {
    let mut winner = deployment_winner(policy, request);
    if let Some(floor) = floor_winner(policy, request) {
        winner = prefer(winner, floor, true);
    }
    for rules in [&policy.profile, &policy.project, &policy.session_narrow] {
        if let Some(matched) = best_match(rules, request) {
            winner = prefer(winner, winner_from_rule(matched, false), false);
        }
    }
    winner.content = winner.content || content_anywhere(policy, request);
    winner = apply_opaque_shell(request, winner);
    winner = apply_fs_place(request, winner);
    winner = apply_grants(policy, request, winner);
    winner = apply_overlay(request, winner);
    winner = apply_capability_join(policy, request, winner);
    winner = apply_waiter(request, winner);
    let binding = binding_for(policy, request, &winner);
    Decision {
        effect: winner.effect,
        reason: winner.reason,
        rule: winner.rule,
        binding,
        obligations: policy.obligations,
        action_id: request.action.id(),
    }
}

fn deployment_winner(policy: &CompiledPolicy, request: &Request<'_>) -> Winner {
    if let Some(matched) = best_match(&policy.deployment, request) {
        let mut winner = winner_from_rule(matched, false);
        if winner.effect == Effect::Deny {
            winner.reason = Reason::new(
                crate::reason::ReasonCode::DeploymentDeny,
                "deployment rule denies the action",
            );
            winner.hard = true;
        }
        return winner;
    }
    policy.deployment_default.map_or_else(
        || Winner {
            effect: Effect::Deny,
            hard: true,
            reason: Reason::missing_policy(),
            rule: Some(RuleRef::builtin(Layer::Floor, "floor.missing_policy")),
            content: false,
        },
        |effect| Winner {
            effect,
            hard: effect == Effect::Deny,
            reason: Reason::deployment_default(),
            rule: policy
                .preset
                .map(|preset| RuleRef::builtin(Layer::Deployment, preset.as_str())),
            content: false,
        },
    )
}

fn floor_winner(policy: &CompiledPolicy, request: &Request<'_>) -> Option<Winner> {
    let matched = best_match(&policy.floor, request).map(|rule| winner_from_rule(rule, true));
    match (matched, production_winner(policy, request)) {
        (Some(rule), Some(production)) => Some(prefer(rule, production, true)),
        (Some(rule), None) => Some(rule),
        (None, Some(production)) => Some(production),
        (None, None) => None,
    }
}

fn production_winner(policy: &CompiledPolicy, request: &Request<'_>) -> Option<Winner> {
    let bot = matches!(request.principal, Principal::Bot { .. });
    let unsafe_posture =
        policy.preset == Some(Preset::Danger) || policy.sandbox == SandboxRequirement::Noop;
    if request.session.production && bot && unsafe_posture {
        Some(Winner {
            effect: Effect::Deny,
            hard: true,
            reason: Reason::production_unsafe(),
            rule: Some(RuleRef::builtin(Layer::Floor, "floor.production_unsafe")),
            content: false,
        })
    } else {
        None
    }
}

fn winner_from_rule(rule: &Rule, floor: bool) -> Winner {
    let reason = match &rule.matcher {
        Matcher::NonHumanApproval => Reason::non_human_approver(),
        Matcher::VendorAuth => Reason::vendor_auth(),
        Matcher::VendorSubscription => Reason::byo_credential(),
        Matcher::Ssrf => Reason::ssrf(),
        Matcher::CheckerError => Reason::checker_error(),
        Matcher::ShellContent {
            dialect: ShellDialect::Posix,
        } if rule.effect == Effect::Deny => Reason::posix_denylist(),
        Matcher::ShellContent {
            dialect: ShellDialect::PowerShell,
        } if rule.effect == Effect::Deny => Reason::powershell_denylist(),
        Matcher::ShellContent {
            dialect: ShellDialect::Cmd,
        } if rule.effect == Effect::Deny => Reason::cmd_denylist(),
        _ => Reason::matched_rule(),
    };
    Winner {
        effect: rule.effect,
        hard: floor || rule.hard || rule.effect == Effect::Deny,
        reason,
        rule: Some(rule.rule_ref()),
        content: rule.matcher.is_shell_content(),
    }
}

fn prefer(current: Winner, incoming: Winner, incoming_wins_ties: bool) -> Winner {
    let content = current.content || incoming.content;
    let take_incoming = if incoming_wins_ties {
        incoming.effect >= current.effect
    } else {
        incoming.effect > current.effect
    };
    if take_incoming {
        Winner {
            hard: current.hard || incoming.hard || incoming.effect == Effect::Deny,
            content,
            ..incoming
        }
    } else {
        Winner {
            hard: current.hard || (incoming.hard && incoming.effect == current.effect),
            content,
            ..current
        }
    }
}

fn content_anywhere(policy: &CompiledPolicy, request: &Request<'_>) -> bool {
    content_rule_matched(&policy.floor, request)
        || content_rule_matched(&policy.deployment, request)
        || content_rule_matched(&policy.profile, request)
        || content_rule_matched(&policy.project, request)
        || content_rule_matched(&policy.session_narrow, request)
        || content_rule_matched(&policy.grants, request)
}

fn apply_opaque_shell(request: &Request<'_>, mut winner: Winner) -> Winner {
    if !shell_plan_opaque(request) {
        return winner;
    }
    if winner.effect == Effect::Allow {
        winner.effect = Effect::Ask;
        winner.reason = Reason::opaque_shell();
        winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.opaque_shell"));
    }
    if winner.effect == Effect::Ask {
        winner.hard = true;
    }
    winner
}

fn apply_fs_place(request: &Request<'_>, mut winner: Winner) -> Winner {
    match fs_place(request) {
        PathPlace::Rejected if winner.effect != Effect::Deny => {
            winner.effect = Effect::Deny;
            winner.hard = true;
            winner.reason = Reason::path_rejected();
            winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.path_rejected"));
        }
        PathPlace::NoWorkspace if winner.effect != Effect::Deny => {
            winner.effect = Effect::Deny;
            winner.hard = true;
            winner.reason = Reason::no_workspace();
            winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.no_workspace"));
        }
        PathPlace::GitMetadata if winner.effect != Effect::Deny => {
            // Host-side git runs commands named in repository config and
            // hooks, so a write there would run outside any sandbox.
            winner.effect = Effect::Deny;
            winner.hard = true;
            winner.reason = Reason::git_metadata();
            winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.git_metadata"));
        }
        PathPlace::Outside if winner.effect != Effect::Deny => {
            // Any filesystem action outside the workspace is a hard ask.
            // Accept-all and a session grant cannot lower it. Only a human answer can.
            let write = !is_readonly(&request.action);
            if winner.effect == Effect::Allow || write {
                winner.effect = Effect::Ask;
                winner.reason = Reason::outside_workspace();
                winner.rule = Some(RuleRef::builtin(Layer::Session, "fs.outside_workspace"));
            }
            winner.hard = true;
        }
        PathPlace::Rejected
        | PathPlace::NoWorkspace
        | PathPlace::GitMetadata
        | PathPlace::Outside
        | PathPlace::Inside
        | PathPlace::Unchecked => {}
    }
    winner
}

/// Strictest place among the action's paths.
///
/// Rejected beats no workspace, which beats a `.git` write, which beats
/// outside. A read of a `.git` path counts as inside.
fn fs_place(request: &Request<'_>) -> PathPlace {
    let Some(paths) = fs_fact_paths(request) else {
        return PathPlace::Unchecked;
    };
    let write = !is_readonly(&request.action);
    let mut place = PathPlace::Unchecked;
    for path in paths {
        match path.place {
            PathPlace::Rejected => return PathPlace::Rejected,
            PathPlace::NoWorkspace => place = PathPlace::NoWorkspace,
            PathPlace::GitMetadata if write && place != PathPlace::NoWorkspace => {
                place = PathPlace::GitMetadata;
            }
            PathPlace::Outside
                if !matches!(place, PathPlace::NoWorkspace | PathPlace::GitMetadata) =>
            {
                place = PathPlace::Outside;
            }
            PathPlace::GitMetadata
            | PathPlace::Outside
            | PathPlace::Inside
            | PathPlace::Unchecked => {}
        }
    }
    place
}

fn fs_fact_paths<'a>(request: &'a Request<'a>) -> Option<&'a [crate::action::ResolvedPath]> {
    match &request.action {
        Action::Tool {
            facts: ToolFacts::Fs { paths, .. } | ToolFacts::Patch { paths },
            ..
        } => Some(paths),
        _ => None,
    }
}

fn apply_grants(policy: &CompiledPolicy, request: &Request<'_>, mut winner: Winner) -> Winner {
    if winner.hard || winner.effect != Effect::Ask {
        return winner;
    }
    if best_match(&policy.grants, request).is_some() {
        winner.effect = Effect::Allow;
        winner.reason = Reason::session_grant();
    }
    winner
}

fn apply_overlay(request: &Request<'_>, mut winner: Winner) -> Winner {
    match request.session.overlay {
        ModeOverlay::Default => {}
        ModeOverlay::AcceptAll => {
            if winner.effect == Effect::Ask && !winner.hard {
                winner.effect = Effect::Allow;
                winner.reason = Reason::accept_all();
            }
        }
        ModeOverlay::AcceptEdits => {
            if winner.effect == Effect::Ask && !winner.hard && is_edit(&request.action) {
                winner.effect = Effect::Allow;
                winner.reason = Reason::new(
                    crate::reason::ReasonCode::AcceptEdits,
                    "accept-edits lowered a non-hard edit ask",
                );
            }
        }
        ModeOverlay::DenyAll => {
            if !is_readonly(&request.action) {
                winner.effect = Effect::Deny;
                winner.hard = true;
                winner.reason = Reason::new(
                    crate::reason::ReasonCode::DenyAll,
                    "deny-all blocks non-read actions",
                );
            }
        }
        ModeOverlay::Interactive => {
            if winner.effect == Effect::Allow && !is_readonly(&request.action) {
                winner.effect = Effect::Ask;
                winner.reason = Reason::new(
                    crate::reason::ReasonCode::Interactive,
                    "interactive overlay asks for side effects",
                );
            }
        }
    }
    winner
}

fn apply_capability_join(
    policy: &CompiledPolicy,
    request: &Request<'_>,
    mut winner: Winner,
) -> Winner {
    let Some(preset) = policy.preset else {
        return winner;
    };
    let gap = crate::capability::obligation_gap(policy.obligations, request.session.host);
    if !gap.limits_shell() || !is_shell_or_code_exec(&request.action) {
        return winner;
    }
    if winner.effect == Effect::Deny {
        return winner;
    }
    match preset {
        Preset::Locked | Preset::Review => {
            winner.effect = Effect::Deny;
            winner.hard = true;
            winner.reason = Reason::unmet_sandbox(gap.network);
            winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.unmet_sandbox"));
        }
        Preset::Standard | Preset::Trusted if winner.effect == Effect::Allow => {
            winner.effect = Effect::Ask;
            winner.hard = true;
            winner.reason = Reason::unmet_sandbox(gap.network);
            winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.unmet_sandbox"));
        }
        Preset::Standard | Preset::Trusted | Preset::Danger => {}
    }
    winner
}

fn is_shell_or_code_exec(action: &Action) -> bool {
    matches!(
        action.tool_class(),
        Some(ToolClass::Shell | ToolClass::CodeExec)
    )
}

fn apply_waiter(request: &Request<'_>, mut winner: Winner) -> Winner {
    if winner.effect == Effect::Ask && matches!(request.waiter, ApprovalWaiter::Absent) {
        winner.effect = Effect::Deny;
        winner.hard = true;
        winner.reason = Reason::no_approver();
        winner.rule = Some(RuleRef::builtin(Layer::Floor, "floor.no_approver"));
    }
    winner
}

fn binding_for(
    policy: &CompiledPolicy,
    request: &Request<'_>,
    winner: &Winner,
) -> Option<ActionBinding> {
    let audit = policy.facets.contains(ActiveFacets::AUDIT_BINDING);
    if winner.effect != Effect::Ask && !winner.content && !audit {
        return None;
    }
    let dialect = shell_dialect(request);
    Some(ActionBinding::compute(
        request.action.id().name(),
        request.raw_args,
        &request.session.cwd,
        dialect.as_deref(),
        request.session.policy_version,
    ))
}

fn shell_dialect(request: &Request<'_>) -> Option<String> {
    match &request.action {
        Action::Tool {
            facts: ToolFacts::Shell { dialect, .. },
            ..
        } => Some(
            match dialect {
                crate::posture::ShellDialect::Posix => "posix",
                crate::posture::ShellDialect::PowerShell => "powershell",
                crate::posture::ShellDialect::Cmd => "cmd",
            }
            .to_owned(),
        ),
        _ => None,
    }
}

const fn is_edit(action: &Action) -> bool {
    match action {
        Action::Tool {
            class: ToolClass::Patch,
            ..
        } => true,
        Action::Tool {
            facts: ToolFacts::Fs { op, .. },
            ..
        } => matches!(op, FsOp::Write | FsOp::Delete),
        _ => false,
    }
}

const fn is_readonly(action: &Action) -> bool {
    match action {
        Action::Tool {
            facts: ToolFacts::Fs { op, .. },
            ..
        } => matches!(op, FsOp::Read | FsOp::List),
        _ => false,
    }
}
