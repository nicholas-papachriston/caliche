//! Build harness config and apply profile settings for @mention and agent override.
//!
//! Turns an `AgentProfile` into a `HarnessConfig` (and applies profile bash rules
//! to a harness's service after creation) so the REPL can run one-off turns with
//! a named profile.

use caliche::AgentService;
use caliche::agent::profile::AgentProfile;
use caliche::agent::subagent::SubagentModelSelection;
use caliche::harness::config::HarnessConfig;
use caliche::harness::formatter::FormatterRule;
use caliche::llm::ProviderType;
use std::path::PathBuf;

/// Build a `HarnessConfig` from an agent profile for use in REPL @mention or override.
///
/// Uses the subagent selection when set. Profile `model:` is next. Parent identity is last.
pub fn harness_config_from_profile(
    profile: &AgentProfile,
    main_provider: ProviderType,
    workspace: PathBuf,
    formatter_rules: Vec<FormatterRule>,
    notify_command: Option<String>,
    fallback_model: String,
    selection: &SubagentModelSelection,
) -> HarnessConfig {
    let (provider, model) =
        selection.resolve(profile.model.as_deref(), main_provider, &fallback_model);
    let system_prompt = profile
        .system_prompt()
        .map(String::from)
        .unwrap_or_default();
    let mut config = HarnessConfig::default()
        .with_provider(provider)
        .with_model(model)
        .with_workspace(workspace)
        .with_auto_checkpoint(false)
        .with_formatters(formatter_rules)
        .with_system_prompt(if system_prompt.is_empty() {
            "You are a helpful assistant.".to_string()
        } else {
            system_prompt
        });
    if let Some(ref cmd) = notify_command {
        config = config.with_notify_command(cmd.clone());
    }
    if let Some(tools) = profile.enabled_tools() {
        if !tools.is_empty() {
            config = config.with_tools(tools);
        }
    }
    if let Some(s) = profile.steps {
        config = config.with_max_turns(s as usize);
    }
    config
}

/// Apply a profile's temperature to an agent service (e.g. after creating a harness).
///
/// When the profile has `temperature` set, the agent's generation temperature is updated.
pub fn apply_profile_temperature_to_service(profile: &AgentProfile, service: &mut AgentService) {
    if let Some(t) = profile.temperature {
        service.agent_mut().set_temperature(Some(t));
    }
}

/// Apply a profile's bash permission rules to an agent service (e.g. after creating a harness).
///
/// Registers `ExecPolicyHook` for the bash tool when the profile has `permission.bash` rules.
pub fn apply_profile_bash_rules_to_service(
    profile: &AgentProfile,
    service: &mut AgentService,
) -> anyhow::Result<()> {
    let Some(ref perm) = profile.permission else {
        return Ok(());
    };
    let Some(ref rules) = perm.bash else {
        return Ok(());
    };
    if rules.is_empty() {
        return Ok(());
    }
    for (i, rule) in rules.iter().enumerate() {
        if let Err(e) = rule.validate_examples() {
            anyhow::bail!(
                "profile '{}' permissions.bash rule {}: {}",
                profile.description.as_deref().unwrap_or("unknown"),
                i + 1,
                e
            );
        }
    }
    use caliche::agent::hooks::{ExecPolicy, ExecPolicyHook, HookMatcher, LifecycleHook};
    use std::sync::Arc;
    let policy = ExecPolicy::new(rules.clone());
    let hook = Arc::new(ExecPolicyHook::new(Arc::new(policy))) as Arc<dyn LifecycleHook>;
    service
        .agent_mut()
        .register_hook(HookMatcher::tool("bash"), hook);
    // REPL is interactive; wire approval resolver so Prompt decisions can be approved
    service.agent_mut().set_approval_resolver(Some(Arc::new(
        crate::approval_resolver::StdioApprovalResolver::new(),
    )));
    service.agent_mut().set_amendment_persister(Some(Arc::new(
        crate::approval_resolver::CliAmendmentPersister::new(None),
    )));
    Ok(())
}

/// Build the persisted subagent model selection from REPL state.
#[must_use]
pub fn selection_from_state(provider: Option<&str>, model: Option<&str>) -> SubagentModelSelection {
    SubagentModelSelection::from_parts(provider, model)
}
