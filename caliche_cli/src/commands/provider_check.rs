//! Early provider-name checks for CLI flags, env, and config selectors.
//!
//! These run before credential lookup so an unknown provider cannot fall
//! through to a default and exit 0.

use anyhow::{Result, bail};

use super::Args;
use super::config::ConfigAction;
use super::subcommands::{Commands, SessionAction};
use caliche::llm::ProviderType;

/// True when this invocation resolves a runtime provider.
///
/// Config inspection does not. A bad `DEFAULT_PROVIDER` must not fail
/// `config path`.
#[must_use]
pub fn invocation_uses_provider(command: Option<&Commands>) -> bool {
    match command {
        None => true,
        Some(
            Commands::Code { .. }
            | Commands::Evaluate { .. }
            | Commands::Init { .. }
            | Commands::Test { .. }
            | Commands::Doctor
            | Commands::Session {
                action: SessionAction::Resume { .. },
            },
        ) => true,
        #[cfg(feature = "scheduling")]
        Some(Commands::Cron { .. }) => true,
        Some(
            Commands::Agent { .. }
            | Commands::Config { .. }
            | Commands::Session { .. }
            | Commands::Share { .. }
            | Commands::Unshare { .. }
            | Commands::Daemon { .. }
            | Commands::Server { .. }
            | Commands::Lsp { .. }
            | Commands::Template { .. }
            | Commands::Prompt { .. }
            | Commands::Completions { .. }
            | Commands::Man { .. }
            | Commands::Skills { .. }
            | Commands::Policy { .. },
        ) => false,
    }
}

/// Reject unknown `--provider` and sibling selectors before credential lookup.
pub fn validate_explicit_providers(args: &Args) -> Result<()> {
    reject_unknown("--provider", args.provider.as_deref())?;
    reject_unknown_allow_clear("--subagent-provider", args.subagent_provider.as_deref())?;
    reject_unknown_allow_clear("--image-provider", args.image_provider.as_deref())?;
    reject_unknown("--list-models", args.list_models.as_deref())?;
    for spec in &args.failover_fallback {
        reject_failover(spec)?;
    }
    if let Some(command) = args.command.as_ref() {
        validate_command_providers(command)?;
    }
    Ok(())
}

/// Reject a set-but-invalid `AGENT_PROVIDER` or `DEFAULT_PROVIDER`.
pub fn validate_env_providers() -> Result<()> {
    reject_unknown("AGENT_PROVIDER", nonempty_env("AGENT_PROVIDER").as_deref())?;
    reject_unknown(
        "DEFAULT_PROVIDER",
        nonempty_env("DEFAULT_PROVIDER").as_deref(),
    )?;
    Ok(())
}

fn validate_command_providers(command: &Commands) -> Result<()> {
    match command {
        Commands::Evaluate { provider, .. } => {
            reject_unknown("evaluate --provider", provider.as_deref())
        }
        Commands::Config { action } => validate_config_action(action),
        _ => Ok(()),
    }
}

fn validate_config_action(action: &ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Default { provider } => reject_unknown("config default", provider.as_deref()),
        ConfigAction::Model { provider, .. } => reject_unknown("config model", Some(provider)),
        _ => Ok(()),
    }
}

fn reject_failover(spec: &str) -> Result<()> {
    let Some((provider, model)) = spec.split_once(':') else {
        bail!("--failover-fallback must be provider:model, got '{spec}'");
    };
    if model.trim().is_empty() {
        bail!("--failover-fallback model must be non-empty");
    }
    reject_unknown("--failover-fallback", Some(provider))
}

fn reject_unknown_allow_clear(label: &str, value: Option<&str>) -> Result<()> {
    if value.is_some_and(|raw| raw.trim().eq_ignore_ascii_case("clear")) {
        return Ok(());
    }
    reject_unknown(label, value)
}

fn reject_unknown(label: &str, value: Option<&str>) -> Result<()> {
    let Some(value) = value.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(());
    };
    if let Err(err) = ProviderType::parse_in_build(value) {
        bail!("{label}: {err}");
    }
    Ok(())
}

fn nonempty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        invocation_uses_provider, reject_unknown, validate_env_providers,
        validate_explicit_providers,
    };
    use crate::commands::{Args, Commands, ConfigAction};
    use clap::Parser;
    use serial_test::serial;

    #[test]
    fn fake_provider_flag_is_rejected_before_use() {
        let args = Args::parse_from(["caliche", "--provider", "not-a-provider", "-q"]);
        let err = validate_explicit_providers(&args).expect_err("fake provider");
        let message = err.to_string();
        assert!(message.contains("not-a-provider"), "{message}");
        assert!(message.contains("--provider"), "{message}");
    }

    #[test]
    #[cfg(not(feature = "local-inference-base"))]
    fn a_provider_flag_left_out_of_the_build_says_how_to_rebuild() {
        let args = Args::parse_from(["caliche", "--provider", "local-inference", "-q"]);

        let err = validate_explicit_providers(&args).expect_err("left-out provider");

        assert_eq!(
            err.to_string(),
            "--provider: local-inference is not in this build. \
             Rebuild with --features local-inference."
        );
    }

    /// Cloud providers left out of the default build name their feature.
    #[test]
    #[cfg(not(any(feature = "bedrock", feature = "azure-openai")))]
    fn cloud_provider_flags_left_out_of_the_build_say_how_to_rebuild() {
        let messages: Vec<String> = ["bedrock", "azure-openai"]
            .into_iter()
            .map(|name| {
                let args = Args::parse_from(["caliche", "--provider", name, "-q"]);
                validate_explicit_providers(&args)
                    .expect_err("left-out provider")
                    .to_string()
            })
            .collect();

        assert_eq!(
            messages,
            vec![
                "--provider: bedrock is not in this build. Rebuild with --features bedrock.",
                "--provider: azure-openai is not in this build. \
                 Rebuild with --features azure-openai.",
            ]
        );
    }

    #[test]
    fn known_provider_flags_are_accepted() {
        let args = Args::parse_from([
            "caliche",
            "--provider",
            "openai",
            "--subagent-provider",
            "anthropic",
            "--image-provider",
            "clear",
            "--failover-fallback",
            "google:gemini-2.5-flash",
        ]);
        validate_explicit_providers(&args).expect("known providers");
    }

    #[test]
    fn config_default_fake_provider_is_rejected() {
        let args = Args::parse_from(["caliche", "config", "default", "not-a-provider"]);
        let err = validate_explicit_providers(&args).expect_err("config default");
        assert!(err.to_string().contains("not-a-provider"));
        assert!(matches!(
            args.command,
            Some(Commands::Config {
                action: ConfigAction::Default { .. }
            })
        ));
    }

    #[test]
    fn config_path_does_not_resolve_a_runtime_provider() {
        let args = Args::parse_from(["caliche", "config", "path"]);
        assert!(!invocation_uses_provider(args.command.as_ref()));
        validate_explicit_providers(&args).expect("config path has no provider flag");
    }

    #[test]
    fn doctor_and_code_resolve_a_provider() {
        let doctor = Args::parse_from(["caliche", "doctor"]);
        assert!(invocation_uses_provider(doctor.command.as_ref()));
        let code = Args::parse_from(["caliche", "code", "fix tests"]);
        assert!(invocation_uses_provider(code.command.as_ref()));
        assert!(invocation_uses_provider(None));
    }

    #[test]
    fn reject_unknown_accepts_gemini_alias() {
        reject_unknown("--provider", Some("gemini")).expect("gemini alias");
        reject_unknown("--provider", None).expect("absent");
        reject_unknown("--provider", Some("  ")).expect("blank");
    }

    #[test]
    #[serial]
    fn fake_env_provider_is_rejected() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("AGENT_PROVIDER", "not-a-provider");
        env.remove("DEFAULT_PROVIDER");
        let err = validate_env_providers().expect_err("env provider");
        assert!(err.to_string().contains("AGENT_PROVIDER"), "{err}");
    }

    #[test]
    #[serial]
    fn default_provider_env_alias_is_validated() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("AGENT_PROVIDER");
        env.set("DEFAULT_PROVIDER", "not-a-provider");
        let err = validate_env_providers().expect_err("default provider env");
        assert!(err.to_string().contains("DEFAULT_PROVIDER"), "{err}");
    }
}
