//! Slash-command catalog tests.

use caliche_web::harness::{
    HARNESS_PRESETS, HarnessDraft, PromptPlacementKind, command_is_forbidden_shell,
    parse_arg_lines, parse_csv_list, preset_by_id, toggle_csv_tag, unique_capabilities,
};

#[test]
fn presets_include_known_harnesses_and_custom() {
    let ids: Vec<&str> = HARNESS_PRESETS.iter().map(|preset| preset.id).collect();
    assert!(ids.contains(&"claude-code"));
    assert!(ids.contains(&"codex"));
    assert!(ids.contains(&"cursor"));
    assert!(ids.contains(&"opencode"));
    assert!(ids.contains(&"pi"));
    assert!(ids.contains(&"hermes"));
    assert_eq!(ids.last().copied(), Some("custom"));
}

#[test]
fn unknown_preset_id_maps_to_custom() {
    let preset = preset_by_id("not-a-real-harness");
    assert_eq!(preset.id, "custom");
    assert!(preset.command.is_empty());
}

#[test]
fn applying_pi_preset_fills_command_and_keeps_name() {
    let mut draft = HarnessDraft::default();
    draft.name = "reviewer".to_string();
    draft.apply_preset(preset_by_id("pi"));
    assert_eq!(draft.name, "reviewer");
    assert_eq!(draft.command, "pi");
    assert_eq!(draft.args_text, "-p");
    assert_eq!(draft.prompt_placement, PromptPlacementKind::LastArg);
    assert!(draft.capabilities_text.contains("pi"));
}

#[test]
fn applying_preset_to_blank_name_uses_preset_label() {
    let draft = HarnessDraft::from_preset(preset_by_id("codex"));
    assert_eq!(draft.name, "Codex");
    assert_eq!(draft.command, "codex");
}

#[test]
fn create_input_rejects_empty_command_and_shells() {
    let mut draft = HarnessDraft::from_preset(preset_by_id("custom"));
    draft.name = "bad".to_string();
    assert!(draft.to_create_input().is_err());

    draft.command = "bash".to_string();
    let err = draft.to_create_input().expect_err("shell");
    assert!(err.contains("shell"));

    draft.command = "/bin/zsh".to_string();
    let err = draft.to_create_input().expect_err("zsh path");
    assert!(err.contains("shell"));
}

#[test]
fn create_input_accepts_echo_custom() {
    let mut draft = HarnessDraft::from_preset(preset_by_id("custom"));
    draft.name = "echo worker".to_string();
    draft.command = "echo".to_string();
    draft.args_text = "-n\n".to_string();
    draft.prompt_placement = PromptPlacementKind::LastArg;
    draft.capabilities_text = "coding, echo".to_string();
    let input = draft.to_create_input().expect("valid");
    assert_eq!(input.command, "echo");
    assert_eq!(input.args, vec!["-n"]);
    assert_eq!(input.capabilities, vec!["coding", "echo"]);
    assert_eq!(input.cwd, None);
}

#[test]
fn parse_arg_lines_skips_blanks_and_rejects_nul() {
    let args = parse_arg_lines("  -p  \n\n--quiet\n").expect("args");
    assert_eq!(args, vec!["-p", "--quiet"]);
    assert!(parse_arg_lines("ok\narg\0bad").is_err());
}

#[test]
fn parse_csv_list_trims() {
    assert_eq!(parse_csv_list(" coding, pi , "), vec!["coding", "pi"]);
}

#[test]
fn unique_capabilities_sorts_and_dedupes() {
    let a = vec!["coding".to_string(), "pi".to_string()];
    let b = vec!["coding".to_string(), "review".to_string()];
    assert_eq!(
        unique_capabilities([a.as_slice(), b.as_slice()]),
        vec!["coding", "pi", "review"]
    );
}

#[test]
fn toggle_csv_tag_adds_and_removes() {
    assert_eq!(toggle_csv_tag("", "coding"), "coding");
    assert_eq!(toggle_csv_tag("coding, pi", "coding"), "pi");
    assert_eq!(toggle_csv_tag("pi", "coding"), "pi, coding");
}

#[test]
fn forbidden_shell_detects_basename() {
    assert!(command_is_forbidden_shell("powershell.exe"));
    assert!(!command_is_forbidden_shell("claude"));
    assert!(!command_is_forbidden_shell(""));
}
