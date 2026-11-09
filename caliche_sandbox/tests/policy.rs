use caliche_sandbox::{ToolPolicy, evaluate_tool_policy};

#[test]
fn deny_list_wins_over_allow() {
    let policy = ToolPolicy {
        allow: vec!["bash".into(), "http".into()],
        deny: vec!["http".into()],
    };
    assert!(!evaluate_tool_policy(&policy, "http"));
    assert!(evaluate_tool_policy(&policy, "bash"));
}

#[test]
fn empty_allow_list_permits_any_non_denied_tool() {
    let policy = ToolPolicy {
        allow: vec![],
        deny: vec![],
    };
    assert!(evaluate_tool_policy(&policy, "custom_tool"));
}
