use super::*;

fn bash_args(command: &str) -> Value {
    serde_json::json!({"command": command, "working_directory": "/workspace"})
}

fn bash_command(command: &str) -> Value {
    serde_json::json!({"command": command})
}

fn bash_in(command: &str, cwd: &str) -> Value {
    serde_json::json!({"command": command, "working_directory": cwd})
}

fn failed_python() -> ToolResult {
    let mut result =
        ToolResult::error("Command failed with exit code 1:\nNameError: name 'x' is not defined");
    result.typed_payload = Some(serde_json::json!({"exit_code": 1}));
    result
}

fn failed_typecheck() -> ToolResult {
    let mut result = ToolResult::error(
        "Command failed with exit code 2:\nsrc/main.ts(1,1): Cannot find name 'x'.",
    );
    result.typed_payload = Some(serde_json::json!({"exit_code": 2}));
    result.content = result.error.clone().unwrap_or_default();
    result
}

#[test]
fn second_identical_bash_failure_is_blocked_without_repeating_stderr() {
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    let args = bash_args("bun run typecheck");
    assert!(breaker.blocked("bash", &args).is_none());
    breaker.observe("bash", &args, &failed_typecheck());

    let blocked = breaker
        .blocked("bash", &args)
        .expect("second identical failure is blocked");
    let message = blocked.error.clone().unwrap_or_default();
    assert!(message.contains(RETRY_BLOCKED_CODE), "{message}");
    assert!(message.contains("stderr_hash="), "{message}");
    assert!(message.contains("exit_code=2"), "{message}");
    assert!(
        !message.contains("Cannot find name"),
        "blocked result repeated stderr: {message}"
    );
    assert!(is_retry_blocked(&blocked));
}

#[test]
fn command_change_and_relevant_edit_clear_the_block() {
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    let args = bash_args("bun run typecheck");
    breaker.observe("bash", &args, &failed_typecheck());
    assert!(breaker.blocked("bash", &args).is_some());

    let other = bash_args("bun run lint");
    assert!(
        breaker.blocked("bash", &other).is_none(),
        "a different command string must run"
    );

    breaker.observe(
        "fs",
        &serde_json::json!({"op": "write", "path": "/tmp/unrelated.txt"}),
        &ToolResult::success("wrote"),
    );
    assert!(
        breaker.blocked("bash", &args).is_some(),
        "an unrelated write must not reset the breaker"
    );

    breaker.observe(
        "fs",
        &serde_json::json!({"op": "edit", "path": "src/main.ts"}),
        &ToolResult::success("edited"),
    );
    assert!(
        breaker.blocked("bash", &args).is_none(),
        "an edit of a path named by the failure must reset the breaker"
    );
}

#[test]
fn same_non_bash_arguments_are_blocked_until_a_write() {
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    let args = serde_json::json!({"path": "src/a.ts", "pattern": "strict"});
    let failure = ToolResult::error("pattern not found");
    breaker.observe("fs", &args, &failure);
    assert!(breaker.blocked("fs", &args).is_some());
    let reordered = serde_json::json!({"pattern": "strict", "path": "src/a.ts"});
    assert!(breaker.blocked("fs", &reordered).is_some());

    breaker.observe(
        "write_file",
        &serde_json::json!({"path": "src/a.ts", "content": "ok"}),
        &ToolResult::success("wrote"),
    );
    assert!(breaker.blocked("fs", &args).is_none());
}

#[test]
fn stderr_hash_is_stable_and_outcome_json_keeps_the_action_id() {
    let body = "src/main.ts(1,1): Cannot find name 'x'.";
    assert_eq!(stderr_hash(body), stderr_hash(body));
    assert_eq!(stderr_hash(body).len(), 12);
    let line = OutcomeLine {
        event: "tool.end",
        action_id: "01936b8e-7c3a-7000-8000-000000000001".to_string(),
        tool: "bash".to_string(),
        ok: Some(false),
        exit_code: Some(2),
        cwd: Some("/workspace".to_string()),
        stderr_hash: Some(stderr_hash(body)),
        stderr: Some(truncate_text(body, STDERR_LOG_CHARS)),
        awaiting_confirmation: false,
        retry_blocked: false,
    };
    let json = serde_json::to_string(&line).expect("outcome json");
    assert!(json.contains("\"ok\":false"), "{json}");
    assert!(
        json.contains("01936b8e-7c3a-7000-8000-000000000001"),
        "{json}"
    );
    assert!(json.contains("\"awaiting_confirmation\":false"), "{json}");
}

#[test]
fn mutating_bash_in_the_same_cwd_clears_the_block() {
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    let typecheck = bash_args("bun run typecheck");
    breaker.observe(TOOL_BASH, &typecheck, &failed_typecheck());
    assert!(breaker.blocked(TOOL_BASH, &typecheck).is_some());

    breaker.observe(
        TOOL_BASH,
        &bash_args("sed -i 's/x/y/' other.ts"),
        &ToolResult::success("edited"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &typecheck).is_none(),
        "sed -i in the failure cwd must allow the typecheck rerun"
    );

    breaker.observe(TOOL_BASH, &typecheck, &failed_typecheck());
    breaker.observe(
        TOOL_BASH,
        &serde_json::json!({
            "command": "sed -i 's/x/y/' other.ts",
            "working_directory": "/tmp/elsewhere"
        }),
        &ToolResult::success("edited"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &typecheck).is_some(),
        "a mutating bash command in another cwd must not clear the block"
    );

    breaker.observe(
        TOOL_BASH,
        &bash_args("cat > src/other.ts <<'EOF'\nlet y = 1;\nEOF"),
        &ToolResult::success("wrote"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &typecheck).is_none(),
        "a heredoc redirect in the failure cwd must allow the typecheck rerun"
    );
}

#[test]
fn content_change_on_a_named_path_clears_the_block() {
    let dir = std::env::temp_dir().join(format!(
        "caliche-breaker-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let file = dir.join("main.ts");
    std::fs::write(&file, "let x = 1;\n").expect("seed file");
    let args = serde_json::json!({
        "command": "bun run typecheck",
        "working_directory": dir.display().to_string()
    });
    let mut failure = ToolResult::error(format!(
        "Command failed with exit code 2:\n{}(1,1): Cannot find name 'x'.",
        file.display()
    ));
    failure.typed_payload = Some(serde_json::json!({"exit_code": 2}));
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    breaker.observe(TOOL_BASH, &args, &failure);
    assert!(breaker.blocked(TOOL_BASH, &args).is_some());
    std::fs::write(&file, "let y = 1;\n").expect("rewrite file");
    assert!(
        breaker.blocked(TOOL_BASH, &args).is_none(),
        "a content change on a path named by the failure must allow the rerun"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn omitted_cwd_and_bare_names_clear_a_rerun() {
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    let failed = bash_command("python3 main.py");
    let failure = failed_python();
    let elsewhere = format!("/tmp/caliche-breaker-elsewhere-{}", std::process::id());
    breaker.observe(TOOL_BASH, &failed, &failure);
    assert!(breaker.blocked(TOOL_BASH, &failed).is_some());

    breaker.observe(
        TOOL_BASH,
        &bash_in("sed -i 's/x/1/' other.py", &elsewhere),
        &ToolResult::success("edited"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &failed).is_some(),
        "a sed expression in another cwd must not clear python3 main.py"
    );

    breaker.observe(
        TOOL_BASH,
        &bash_in("cat > main.py <<EOF\nprint(1)\nEOF", &elsewhere),
        &ToolResult::success("wrote"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &failed).is_none(),
        "writing bare main.py must clear the block"
    );

    breaker.observe(TOOL_BASH, &failed, &failure);
    assert!(breaker.blocked(TOOL_BASH, &failed).is_some());
    breaker.observe(
        TOOL_BASH,
        &bash_command("python3 -c open('main.py','r')"),
        &ToolResult::success("read"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &failed).is_some(),
        "reading main.py must not clear the block"
    );

    breaker.observe(
        TOOL_BASH,
        &bash_command("cat > other.py <<EOF\nprint(1)\nEOF"),
        &ToolResult::success("wrote"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &failed).is_none(),
        "a mutating command with no working_directory must clear a no-cwd failure"
    );

    breaker.observe(TOOL_BASH, &failed, &failure);
    assert!(breaker.blocked(TOOL_BASH, &failed).is_some());
    breaker.observe(
        TOOL_BASH,
        &bash_command("python3 -c open('main.py','w')"),
        &ToolResult::success("wrote"),
    );
    assert!(
        breaker.blocked(TOOL_BASH, &failed).is_none(),
        "python open(..., 'w') with no cwd must clear the named file"
    );
}

#[test]
fn content_change_on_a_bare_filename_clears_the_block() {
    let dir = std::env::temp_dir().join(format!(
        "caliche-breaker-bare-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let file = dir.join("main.py");
    std::fs::write(&file, "x = 1\n").expect("seed file");
    let args = serde_json::json!({
        "command": "python3 main.py",
        "working_directory": dir.display().to_string()
    });
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    breaker.observe(TOOL_BASH, &args, &failed_python());
    assert!(breaker.blocked(TOOL_BASH, &args).is_some());
    std::fs::write(&file, "x = 2\n").expect("rewrite file");
    assert!(
        breaker.blocked(TOOL_BASH, &args).is_none(),
        "a content change on bare main.py must allow the rerun"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn transient_network_failures_are_not_blocked() {
    let mut breaker = IdenticalFailureBreaker::with_enabled(true);
    let fetch = serde_json::json!({"url": "https://example.test/doc"});
    let unavailable = ToolResult::error("web_fetch failed: 503 Service Unavailable");
    breaker.observe(TOOL_WEB_FETCH, &fetch, &unavailable);
    assert!(
        breaker.blocked(TOOL_WEB_FETCH, &fetch).is_none(),
        "HTTP 503 must stay retryable"
    );

    let timed_out = ToolResult::error("request timed out after 30s");
    breaker.observe(TOOL_HTTP, &fetch, &timed_out);
    assert!(
        breaker.blocked(TOOL_HTTP, &fetch).is_none(),
        "timeouts must stay retryable"
    );

    let missing = ToolResult::error("web_fetch failed: 404 Not Found");
    breaker.observe(TOOL_WEB_FETCH, &fetch, &missing);
    assert!(
        breaker.blocked(TOOL_WEB_FETCH, &fetch).is_some(),
        "a stable 404 is not a transient failure"
    );

    let request_timeout = ToolResult::error("web_fetch failed: HTTP 408");
    let mut fetch_breaker = IdenticalFailureBreaker::with_enabled(true);
    fetch_breaker.observe(TOOL_WEB_FETCH, &fetch, &request_timeout);
    assert!(
        fetch_breaker.blocked(TOOL_WEB_FETCH, &fetch).is_none(),
        "HTTP 408 must stay retryable"
    );

    let too_many = ToolResult::error("http failed: HTTP 429");
    let mut http_breaker = IdenticalFailureBreaker::with_enabled(true);
    http_breaker.observe(TOOL_HTTP, &fetch, &too_many);
    assert!(
        http_breaker.blocked(TOOL_HTTP, &fetch).is_none(),
        "HTTP 429 must stay retryable"
    );
}

#[test]
fn off_switch_disables_the_breaker() {
    assert!(!env_enables_breaker("0"));
    assert!(!env_enables_breaker(" off "));
    assert!(!env_enables_breaker("false"));
    assert!(!env_enables_breaker("no"));
    assert!(!env_enables_breaker("disabled"));
    assert!(env_enables_breaker("1"));
    assert!(env_enables_breaker(""));

    let args = bash_args("bun run typecheck");
    for off in ["0", "false", "off", "no", "disabled"] {
        let mut breaker = IdenticalFailureBreaker::from_env_value(Some(off));
        breaker.observe(TOOL_BASH, &args, &failed_typecheck());
        assert!(
            breaker.blocked(TOOL_BASH, &args).is_none(),
            "injected {off} must disable the breaker"
        );
    }

    let mut enabled = IdenticalFailureBreaker::from_env_value(Some("1"));
    enabled.observe(TOOL_BASH, &args, &failed_typecheck());
    assert!(
        enabled.blocked(TOOL_BASH, &args).is_some(),
        "injected 1 must keep the breaker on"
    );

    let mut unset = IdenticalFailureBreaker::from_env_value(None);
    unset.observe(TOOL_BASH, &args, &failed_typecheck());
    assert!(
        unset.blocked(TOOL_BASH, &args).is_some(),
        "an unset value must keep the breaker on"
    );
}

#[tokio::test]
async fn new_reads_the_injected_env_value() {
    let off =
        super::with_injected_breaker_env(Some("0"), async { IdenticalFailureBreaker::new() }).await;
    assert!(
        !off.enabled,
        "IdenticalFailureBreaker::new must honor an injected off switch"
    );
    let on =
        super::with_injected_breaker_env(Some("1"), async { IdenticalFailureBreaker::new() }).await;
    assert!(
        on.enabled,
        "IdenticalFailureBreaker::new must honor an injected on switch"
    );
}
