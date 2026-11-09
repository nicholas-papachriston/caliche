//! Slash command parsing and catalog tests.

use caliche_contracts::slash::SlashSurface;
use caliche_web::{
    Command, InteractionMode, SLASH_COMMANDS, generate_help_text, is_command, parse_command,
};

#[test]
fn is_command_detects_slash_prefix() {
    assert!(is_command("/help"));
    assert!(is_command("  /help  "));
    assert!(!is_command("help"));
    assert!(!is_command("hello /help"));
}

#[test]
fn parse_help_command() {
    assert_eq!(parse_command("/help"), Ok(Command::Help));
    assert_eq!(parse_command("/?"), Ok(Command::Help));
}

#[test]
fn parse_mode_commands() {
    assert_eq!(
        parse_command("/plan"),
        Ok(Command::SetMode(InteractionMode::Plan))
    );
    assert_eq!(
        parse_command("/ask"),
        Ok(Command::SetMode(InteractionMode::Ask))
    );
    assert_eq!(
        parse_command("/agent"),
        Ok(Command::SetMode(InteractionMode::Agent))
    );
    assert_eq!(parse_command("/mode"), Ok(Command::ShowMode));
    assert_eq!(parse_command("/showplan"), Ok(Command::ShowPlan));
    assert_eq!(parse_command("/show-plan"), Ok(Command::ShowPlan));
}

#[test]
fn parse_checkpoint_requires_label() {
    assert_eq!(
        parse_command("/checkpoint my checkpoint"),
        Ok(Command::Checkpoint("my checkpoint".to_string()))
    );
    assert!(parse_command("/checkpoint").is_err());
}

#[test]
fn parse_unknown_suggests_near_match() {
    assert!(parse_command("/unknown").is_err());
    let Err(err) = parse_command("/showpln") else {
        panic!("misspelled command should fail");
    };
    assert!(err.contains("Did you mean /show-plan?"));
}

#[test]
fn parse_history_and_cli_hints() {
    assert_eq!(parse_command("/sessions"), Ok(Command::ShowHistory));
    assert_eq!(parse_command("/history"), Ok(Command::ShowHistory));
    assert_eq!(parse_command("/doctor"), Ok(Command::DoctorHelp));
    assert_eq!(parse_command("/init"), Ok(Command::InitHelp));
    assert_eq!(parse_command("/resume"), Ok(Command::ResumeHelp));
    assert_eq!(parse_command("/resume last"), Ok(Command::ResumeHelp));
    assert_eq!(parse_command("/export"), Ok(Command::ExportHelp));
    assert_eq!(parse_command("/export out.md"), Ok(Command::ExportHelp));
    assert_eq!(parse_command("/settings"), Ok(Command::ShowSettings));
    assert_eq!(parse_command("/chat"), Ok(Command::ShowChat));
    assert_eq!(parse_command("/documents"), Ok(Command::ShowDocuments));
    assert_eq!(parse_command("/docs"), Ok(Command::ShowDocuments));
    assert_eq!(parse_command("/plugins"), Ok(Command::ShowPlugins));
    assert_eq!(parse_command("/reasoning"), Ok(Command::ShowReasoning));
}

#[test]
fn slash_commands_catalog_includes_key_entries() {
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/help"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/plan"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/providers"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/show-plan"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/sessions"));
    assert!(!SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/doctor"));
    assert!(!SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/export"));
    assert!(!SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/init"));
    assert!(!SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/share"));
    assert!(!SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/unshare"));
    assert!(!SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/suspend"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/settings"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/chat"));
    assert!(SLASH_COMMANDS.iter().any(|(cmd, _)| *cmd == "/documents"));
}

#[test]
fn help_includes_show_plan_and_showplan_parses() {
    let help = generate_help_text();
    assert!(help.contains("`/show-plan`"));
    assert!(help.contains("`/sessions`"));
    assert!(!help.contains("`/export`"));
    assert!(!help.contains("`/doctor`"));
    assert!(!help.contains("`/init`"));
    assert!(!help.contains("`/share`"));
    assert!(!help.contains("`/unshare`"));
    assert!(!help.contains("`/suspend`"));
    assert_eq!(parse_command("/showplan"), Ok(Command::ShowPlan));
}

#[test]
fn shared_ui_slash_specs_resolve() {
    for spec in caliche_contracts::slash::all_slash_commands(SlashSurface::Ui) {
        let command = match spec.argument_kind {
            caliche_contracts::slash::SlashArgumentKind::Required
            | caliche_contracts::slash::SlashArgumentKind::Freeform => {
                format!("{} value", spec.primary_name)
            }
            _ => spec.primary_name.to_string(),
        };
        assert!(
            parse_command(&command).is_ok(),
            "{} should parse",
            spec.primary_name
        );
        for alias in spec.aliases {
            let alias_command = match spec.argument_kind {
                caliche_contracts::slash::SlashArgumentKind::Required
                | caliche_contracts::slash::SlashArgumentKind::Freeform => {
                    format!("{alias} value")
                }
                _ => (*alias).to_string(),
            };
            assert!(
                parse_command(&alias_command).is_ok(),
                "{alias} should parse"
            );
        }
    }
}
