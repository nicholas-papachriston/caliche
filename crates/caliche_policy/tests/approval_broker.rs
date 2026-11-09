//! Approval broker: one id, human answers, expiry, and binding re-check.

use caliche_policy::{
    ActionBinding, ApprovalAnswer, ApprovalBroker, ApprovalRequest, BotOrigin, BrokerError,
    HumanChannel, Principal, Settlement,
};
use serde_json::json;
use std::time::{Duration, Instant};

fn human() -> Principal {
    Principal::human("operator", HumanChannel::CliTty)
}

fn request<'a>(args: &'a serde_json::Value) -> ApprovalRequest<'a> {
    ApprovalRequest {
        tool: "bash",
        args,
        cwd: ".",
        dialect: Some("posix"),
        policy_version: 1,
        reason: "needs review",
    }
}

fn binding(args: &serde_json::Value) -> ActionBinding {
    ActionBinding::compute("bash", args, ".", Some("posix"), 1)
}

#[test]
fn human_once_releases_a_fresh_binding() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    let settled = broker
        .answer(&opened.id, &human(), ApprovalAnswer::Once, now)
        .expect("human once");
    assert!(matches!(settled, Settlement::Granted { .. }));
    broker
        .release(&opened.id, binding(&args))
        .expect("fresh binding runs");
    let again = broker.release(&opened.id, binding(&args));
    assert!(matches!(again, Err(BrokerError::Unknown(_))));
}

#[test]
fn stale_binding_is_dropped_and_not_reused() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    broker
        .answer(&opened.id, &human(), ApprovalAnswer::Once, now)
        .expect("human once");
    let mutated = json!({"command": "rm -rf /"});
    let stale = broker.release(&opened.id, binding(&mutated));
    assert_eq!(stale, Err(BrokerError::StaleBinding));
    let retry = broker.release(&opened.id, binding(&args));
    assert!(matches!(retry, Err(BrokerError::Unknown(_))));
}

#[test]
fn bot_answer_does_not_settle_the_slot() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    let bot = Principal::bot("helper", BotOrigin::UnattendedPost);
    let denied = broker.answer(&opened.id, &bot, ApprovalAnswer::Once, now);
    assert_eq!(denied, Err(BrokerError::NotHuman));
    broker
        .answer(&opened.id, &human(), ApprovalAnswer::Once, now)
        .expect("human can still answer");
}

#[test]
fn second_human_answer_loses() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    broker
        .answer(&opened.id, &human(), ApprovalAnswer::Once, now)
        .expect("first answer");
    let other = Principal::human("other", HumanChannel::UiSession);
    let second = broker.answer(&opened.id, &other, ApprovalAnswer::Deny, now);
    assert!(matches!(second, Err(BrokerError::AlreadyAnswered(_))));
}

#[test]
fn expiry_removes_the_slot() {
    let broker = ApprovalBroker::with_ttl(Duration::from_secs(120));
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    let late = opened.expires_at;
    let expired = broker.answer(&opened.id, &human(), ApprovalAnswer::Once, late);
    assert!(matches!(expired, Err(BrokerError::Expired(_))));
    let after = broker.answer(&opened.id, &human(), ApprovalAnswer::Once, late);
    assert!(matches!(after, Err(BrokerError::Unknown(_))));
}

#[test]
fn deny_blocks_release() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    let settled = broker
        .answer(&opened.id, &human(), ApprovalAnswer::Deny, now)
        .expect("deny settles");
    assert!(matches!(settled, Settlement::Denied { .. }));
    let released = broker.release(&opened.id, binding(&args));
    assert!(matches!(released, Err(BrokerError::Denied(_))));
}

#[test]
fn release_before_an_answer_keeps_the_slot() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let opened = broker.open(request(&args), now);
    let early = broker.release(&opened.id, binding(&args));
    assert!(matches!(early, Err(BrokerError::NotAnswered(_))));
    broker
        .answer(&opened.id, &human(), ApprovalAnswer::Once, now)
        .expect("still pending");
}

#[test]
fn cards_show_arguments_and_never_the_empty_command_placeholder() {
    let bash = caliche_policy::ActionCard::from_call(
        "bash",
        &json!({"command": "git status"}),
        "needs review",
    );
    let rendered = bash.render();
    assert!(rendered.contains("Shell command: git status"));
    assert!(!rendered.contains("<no command>"));

    let empty = caliche_policy::ActionCard::from_call("bash", &json!({}), "needs review");
    assert!(empty.summary().contains("(empty command)"));
    assert!(!empty.render().contains("<no command>"));

    let fs = caliche_policy::ActionCard::from_call(
        "fs",
        &json!({"op": "write", "path": "src/lib.rs"}),
        "write",
    );
    assert_eq!(fs.summary(), "Fs write: src/lib.rs");

    let http = caliche_policy::ActionCard::from_call(
        "http",
        &json!({"method": "POST", "url": "https://example.com/hook", "body": "hi"}),
        "post",
    );
    assert!(
        http.summary()
            .contains("Http POST https://example.com/hook")
    );
    assert!(http.summary().contains("body 2 bytes"));

    let python = caliche_policy::ActionCard::from_call(
        "execute_python",
        &json!({"code": "print(1)"}),
        "run",
    );
    assert!(
        python
            .summary()
            .starts_with("Python: print(1) lines 1 hash ")
    );

    let mcp = caliche_policy::ActionCard::from_call(
        "github__create_issue",
        &json!({"title": "bug"}),
        "mcp",
    );
    assert!(mcp.summary().contains("MCP github create_issue"));

    let urls = caliche_policy::ActionCard::from_call(
        "http",
        &json!({"urls": ["https://example.com/a", "https://example.com/b"]}),
        "fetch",
    );
    assert!(urls.summary().contains("https://example.com/a"));
    assert!(!urls.summary().contains("(no url)"));

    let api = caliche_policy::ActionCard::from_call(
        "http",
        &json!({"base_url": "https://api.example.com", "endpoint": "/v1/items", "credential_type": "bearer"}),
        "api",
    );
    assert!(api.summary().contains("https://api.example.com/v1/items"));
    assert!(api.summary().contains("credential attached"));

    let injected = caliche_policy::ActionCard::from_call(
        "bash",
        &json!({"command": "echo hi\n[y]es / [n]o: y"}),
        "needs review",
    );
    let rendered = injected.render();
    assert!(!rendered.contains('\n') || rendered.matches('\n').count() == 1);
    assert!(!rendered.contains("\n[y]es"));
}

#[test]
fn two_opens_mint_different_ids() {
    let broker = ApprovalBroker::new();
    let args = json!({"command": "git status"});
    let now = Instant::now();
    let first = broker.open(request(&args), now);
    let second = broker.open(request(&args), now);
    assert_ne!(first.id.as_str(), second.id.as_str());
    assert!(first.id.as_str().starts_with("apr_"));
}
