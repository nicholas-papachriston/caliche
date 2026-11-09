//! The interactive CLI in a pseudo-terminal: it reaches its first prompt in
//! a workspace whose layout used to stop startup (a copied submodule gitfile
//! whose git dir does not exist, under an ignored build directory, and a
//! directory nobody can list), and Ctrl+C in a setup prompt returns to the
//! REPL.
//!
//! The binary runs with a temp HOME, the OS keychain off, and ollama pointed
//! at an in-test server. Each test sends input when the output shows a
//! given text, and ends with `/exit`.
//!
//! `CALICHE_PTY_BIN`, when set, runs that binary (an installed one) instead
//! of the one this package builds.

#![cfg(target_os = "macos")]

use std::io::{Read, Write};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The REPL prompt for the provider and model the test sets.
const PROMPT: &str = "[ollama/fake-model]";
/// A terminal's cursor-position query; reedline waits for the answer.
const CURSOR_QUERY: &[u8] = b"\x1b[6n";

/// A pseudo-terminal pair: (master, slave), 60 rows by 160 columns, so
/// pickers lay out every item the same way on each run.
fn open_pty() -> (OwnedFd, OwnedFd) {
    let mut master = 0;
    let mut slave = 0;
    let mut size = libc::winsize {
        ws_row: 60,
        ws_col: 160,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: openpty writes two new descriptors into the out parameters and
    // reads `size`; null name and termios are allowed.
    #[expect(unsafe_code, reason = "openpty has no safe wrapper in std or libc")]
    let rc = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut size,
        )
    };
    assert_eq!(rc, 0, "openpty: {}", std::io::Error::last_os_error());
    // SAFETY: openpty succeeded, so both are open descriptors this process
    // owns and nothing else closes.
    #[expect(unsafe_code, reason = "taking ownership of descriptors from openpty")]
    unsafe {
        (OwnedFd::from_raw_fd(master), OwnedFd::from_raw_fd(slave))
    }
}

/// Whether the terminal behind `fd` is in raw mode (no line editing), as it
/// is while the REPL or a picker waits for a key.
fn raw_mode(fd: &OwnedFd) -> bool {
    use std::os::fd::AsRawFd;
    // SAFETY: an all-zero termios is a valid out parameter for tcgetattr.
    #[expect(unsafe_code, reason = "termios has no safe wrapper in std or libc")]
    let mut termios: libc::termios = unsafe { std::mem::zeroed() };
    // SAFETY: `fd` is an open terminal descriptor and `termios` is writable.
    #[expect(unsafe_code, reason = "termios has no safe wrapper in std or libc")]
    let rc = unsafe { libc::tcgetattr(fd.as_raw_fd(), &mut termios) };
    rc == 0 && termios.c_lflag & libc::ICANON == 0
}

/// An OpenAI-compatible model server that answers every request with `done`.
async fn fake_model() -> String {
    async fn chat(
        axum::Json(_request): axum::Json<serde_json::Value>,
    ) -> axum::Json<serde_json::Value> {
        axum::Json(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "done" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
        }))
    }
    let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    tokio::spawn(async move { axum::serve(listener, app).await });
    base_url
}

/// The binary under test.
fn caliche_bin() -> std::path::PathBuf {
    std::env::var_os("CALICHE_PTY_BIN").map_or_else(
        || std::path::PathBuf::from(env!("CARGO_BIN_EXE_caliche")),
        std::path::PathBuf::from,
    )
}

/// Run the interactive CLI in `workspace`. For each `(text, input)` step in
/// order, wait until `text` appears after the previous step's match and the
/// terminal is in raw mode (something is reading keys), then send `input`.
/// Return the exit code and everything it printed.
fn run_script(
    home: &Path,
    workspace: &Path,
    base_url: &str,
    env: &[(&str, &str)],
    steps: &[(&str, &[u8])],
) -> (Option<i32>, String) {
    run_script_with_ctrl_c(home, workspace, base_url, env, steps, CtrlC::AsAKey)
}

/// The byte a terminal sends for Ctrl+C.
const CTRL_C: &[u8] = b"\x03";

/// When a Ctrl+C step is sent.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CtrlC {
    /// Like any key: once something reads keys in raw mode.
    AsAKey,
    /// At once, in cooked mode: during a turn the terminal turns it into
    /// SIGINT, as it does for a user.
    DuringATurn,
}

/// [`run_script`], with Ctrl+C steps sent as `ctrl_c` says.
fn run_script_with_ctrl_c(
    home: &Path,
    workspace: &Path,
    base_url: &str,
    env: &[(&str, &str)],
    steps: &[(&str, &[u8])],
    ctrl_c: CtrlC,
) -> (Option<i32>, String) {
    let (master, slave) = open_pty();
    let terminal = slave.try_clone().expect("slave");
    let mut command = Command::new(caliche_bin());
    command
        .current_dir(workspace)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("TERM", "xterm-256color")
        .env("CALICHE_NO_KEYRING", "1")
        .env("AGENT_PROVIDER", "ollama")
        .env("OLLAMA_MODEL", "fake-model")
        .env("OLLAMA_BASE_URL", base_url)
        .envs(env.iter().copied())
        .stdin(Stdio::from(slave.try_clone().expect("slave")))
        .stdout(Stdio::from(slave.try_clone().expect("slave")))
        .stderr(Stdio::from(slave));
    // SAFETY: only async-signal-safe calls run between fork and exec.
    #[expect(unsafe_code, reason = "make the pty the child's controlling terminal")]
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 || libc::ioctl(0, libc::TIOCSCTTY.into(), 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn().expect("spawn caliche");
    drop(command);

    let mut reader = std::fs::File::from(master.try_clone().expect("master"));
    let mut writer = std::fs::File::from(master);
    let (chunks, received) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        while let Ok(n) = reader.read(&mut buf) {
            if n == 0 || chunks.send(buf[..n].to_vec()).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + WAIT;
    let mut output = Vec::new();
    let mut step = 0;
    let mut matched_to = 0;
    let code = loop {
        if let Some(status) = child.try_wait().expect("wait") {
            break status.code();
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            break None;
        }
        // An idle tick still checks the steps: raw mode can start after the
        // last output.
        if let Ok(chunk) = received.recv_timeout(Duration::from_millis(50)) {
            for _ in 0..chunk
                .windows(CURSOR_QUERY.len())
                .filter(|w| *w == CURSOR_QUERY)
                .count()
            {
                writer.write_all(b"\x1b[1;1R").expect("answer cursor query");
            }
            output.extend_from_slice(&chunk);
        }
        while let Some((text, input)) = steps.get(step) {
            let seen = String::from_utf8_lossy(&output[matched_to..]).into_owned();
            let Some(at) = seen.find(text) else {
                break;
            };
            let signal = ctrl_c == CtrlC::DuringATurn && *input == CTRL_C;
            if !input.is_empty() && !signal && !raw_mode(&terminal) {
                break;
            }
            matched_to += seen[..at + text.len()].len();
            writer.write_all(input).expect("send step input");
            step += 1;
        }
    };
    while let Ok(chunk) = received.recv_timeout(Duration::from_millis(200)) {
        output.extend_from_slice(&chunk);
    }
    (code, String::from_utf8_lossy(&output).into_owned())
}

#[tokio::test(flavor = "multi_thread")]
async fn repl_reaches_the_prompt_with_a_dangling_gitfile_and_a_locked_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    for sub in ["objects", "refs/heads"] {
        std::fs::create_dir_all(workspace.join(".git").join(sub)).expect("git dir");
    }
    std::fs::write(
        workspace.join(".git").join("HEAD"),
        "ref: refs/heads/main\n",
    )
    .expect("HEAD");
    std::fs::write(workspace.join(".gitignore"), "node_modules/\n").expect(".gitignore");
    let pkg = workspace.join("node_modules").join("pkg");
    std::fs::create_dir_all(&pkg).expect("pkg");
    std::fs::write(pkg.join(".git"), "gitdir: ../.git/modules/pkg\n").expect("gitfile");
    let locked = workspace.join("locked");
    std::fs::create_dir(&locked).expect("locked");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");
    let base_url = fake_model().await;

    let (home_c, workspace_c) = (home.clone(), workspace.clone());
    let (code, output) = tokio::task::spawn_blocking(move || {
        run_script(
            &home_c,
            &workspace_c,
            &base_url,
            &[],
            &[(PROMPT, b"/exit\r")],
        )
    })
    .await
    .expect("pty run");

    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod back");
    assert!(output.contains(PROMPT), "{output}");
    assert_eq!(code, Some(0), "{output}");
}

/// Ctrl+C in the setup picker prints one cancel line and returns to the
/// REPL, which still takes commands.
#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_in_setup_returns_to_the_repl_with_the_cancel_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    std::fs::create_dir_all(&workspace).expect("workspace");
    let base_url = fake_model().await;

    let (code, output) = tokio::task::spawn_blocking(move || {
        run_script(
            &home,
            &workspace,
            &base_url,
            &[],
            &[
                (PROMPT, b"/creds setup\r"),
                // The menu's last line: the picker has the terminal in raw mode.
                ("Done - finish provider setup", b"\x03"),
                ("Setup cancelled.", b""),
                (PROMPT, b"/exit\r"),
            ],
        )
    })
    .await
    .expect("pty run");

    let after_cancel = output
        .split("Done - finish provider setup")
        .last()
        .unwrap_or("");
    assert!(after_cancel.contains("Setup cancelled."), "{output}");
    assert!(!output.contains("Setup failed"), "{output}");
    assert_eq!(code, Some(0), "{output}");
}

/// A provider without a key: the REPL starts, prints one line naming the
/// provider, the missing key, and the fix, and takes commands.
#[tokio::test(flavor = "multi_thread")]
async fn repl_starts_without_a_model_key_and_names_the_fix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    std::fs::create_dir_all(&workspace).expect("workspace");
    let base_url = fake_model().await;

    let (code, output) = tokio::task::spawn_blocking(move || {
        run_script(
            &home,
            &workspace,
            &base_url,
            &[("AGENT_PROVIDER", "openrouter")],
            &[("[openrouter/", b"/exit\r")],
        )
    })
    .await
    .expect("pty run");

    assert!(
        output.contains(
            "openrouter has no API key. Run /creds setup providers or set OPENROUTER_API_KEY."
        ),
        "{output}"
    );
    assert!(output.contains("[openrouter/"), "{output}");
    assert_eq!(code, Some(0), "{output}");
}

/// The REPL prompt with the in-test LM Studio.
const LM_STUDIO_PROMPT: &str = "[lmstudio/fake-model]";

/// An in-test LM Studio that has the chat model but not the embedding model.
struct LmStudio {
    base_url: String,
    /// `POST /v1/embeddings` requests; each gets 400 "model not found".
    embedding_requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// The body of each `POST /v1/chat/completions`.
    chat_requests: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

/// Start the in-test LM Studio with `models`, an array of model states.
/// `GET /api/v0/models/<id>` answers the state whose `id` matches.
async fn lm_studio_without_the_embedding_model(models: serde_json::Value) -> LmStudio {
    use axum::extract::{Path, State};
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Shared {
        models: serde_json::Value,
        embedding_requests: Arc<AtomicUsize>,
        chat_requests: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    async fn chat(
        State(shared): State<Shared>,
        axum::Json(request): axum::Json<serde_json::Value>,
    ) -> axum::response::Response {
        let stream = request["stream"].as_bool() == Some(true);
        shared.chat_requests.lock().expect("lock").push(request);
        if !stream {
            return axum::Json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "fake-model",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "stub answer" },
                    "finish_reason": "stop",
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
            }))
            .into_response();
        }
        let chunk = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "stub answer" },
                "finish_reason": "stop",
            }],
        });
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
        )
            .into_response()
    }
    async fn embeddings(State(shared): State<Shared>) -> impl IntoResponse {
        shared.embedding_requests.fetch_add(1, Ordering::SeqCst);
        (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": { "message": "Model not found: text-embedding-qwen3-embedding-4b" },
            })),
        )
    }
    async fn model_list(State(shared): State<Shared>) -> axum::Json<serde_json::Value> {
        let data: Vec<serde_json::Value> = shared
            .models
            .as_array()
            .into_iter()
            .flatten()
            .map(|model| serde_json::json!({ "id": model["id"], "object": "model" }))
            .collect();
        axum::Json(serde_json::json!({ "object": "list", "data": data }))
    }
    async fn model_state(
        State(shared): State<Shared>,
        Path(id): Path<String>,
    ) -> axum::response::Response {
        shared
            .models
            .as_array()
            .into_iter()
            .flatten()
            .find(|model| model["id"] == id.as_str())
            .map_or_else(
                || StatusCode::NOT_FOUND.into_response(),
                |model| axum::Json(model.clone()).into_response(),
            )
    }

    let shared = Shared {
        models,
        embedding_requests: Arc::new(AtomicUsize::new(0)),
        chat_requests: Arc::new(Mutex::new(Vec::new())),
    };
    let stub = LmStudio {
        base_url: String::new(),
        embedding_requests: Arc::clone(&shared.embedding_requests),
        chat_requests: Arc::clone(&shared.chat_requests),
    };
    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(chat))
        .route("/v1/embeddings", axum::routing::post(embeddings))
        .route("/v1/models", axum::routing::get(model_list))
        .route("/api/v0/models/{id}", axum::routing::get(model_state))
        .with_state(shared);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    tokio::spawn(async move { axum::serve(listener, app).await });
    LmStudio { base_url, ..stub }
}

/// Run the REPL against the in-test LM Studio: say `hey`, wait for the
/// answer, exit.
async fn hey_to_lm_studio(
    home: std::path::PathBuf,
    workspace: std::path::PathBuf,
    base_url: String,
) -> (Option<i32>, String) {
    tokio::task::spawn_blocking(move || {
        run_script(
            &home,
            &workspace,
            &base_url,
            &[
                ("AGENT_PROVIDER", "lmstudio"),
                ("LMSTUDIO_BASE_URL", &base_url),
                ("LMSTUDIO_MODEL", "fake-model"),
            ],
            &[
                (LM_STUDIO_PROMPT, b"hey\r"),
                ("stub answer", b""),
                (LM_STUDIO_PROMPT, b"/exit\r"),
            ],
        )
    })
    .await
    .expect("pty run")
}

/// LM Studio has no embedding model: one embedding request, one line that
/// says workspace search is off and how to fix it, and the REPL still
/// answers.
#[tokio::test(flavor = "multi_thread")]
async fn a_missing_embedding_model_costs_one_request_and_one_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    std::fs::create_dir_all(&workspace).expect("workspace");
    for i in 0..40 {
        std::fs::write(
            workspace.join(format!("f{i}.rs")),
            format!("fn f{i}() {{}}\n"),
        )
        .expect("file");
    }
    let lm_studio = lm_studio_without_the_embedding_model(
        serde_json::json!([{ "id": "fake-model", "state": "loaded" }]),
    )
    .await;

    let (code, output) = hey_to_lm_studio(home, workspace, lm_studio.base_url.clone()).await;

    let warning = "Workspace search is off for this session: lmstudio API error (400 Bad Request): Model not found: text-embedding-qwen3-embedding-4b. Load or download the embedding model, or start with --no-workspace-embed.";
    assert_eq!(output.matches(warning).count(), 1, "{output}");
    assert!(output.contains("stub answer"), "{output}");
    assert_eq!(
        lm_studio
            .embedding_requests
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "{output}"
    );
    assert_eq!(code, Some(0), "{output}");
}

/// The system prompt of the first chat request.
fn first_system_prompt(lm_studio: &LmStudio) -> String {
    let requests = lm_studio.chat_requests.lock().expect("lock");
    requests
        .first()
        .and_then(|request| request["messages"][0]["content"].as_str())
        .expect("a chat request with a system message")
        .to_string()
}

/// A workspace with an AGENTS.md of 2000 lines (52,000 characters).
fn workspace_with_a_large_agents_md(base: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    std::fs::create_dir_all(&workspace).expect("workspace");
    let agents_md: String = (0..2000)
        .map(|i| format!("rule {i:04}: keep this line\n"))
        .collect();
    std::fs::write(workspace.join("AGENTS.md"), agents_md).expect("AGENTS.md");
    (home, workspace)
}

/// LM Studio reports a 32K loaded context: the first prompt carries the
/// head of AGENTS.md, within 25% of the window, and the cut note.
#[tokio::test(flavor = "multi_thread")]
async fn a_32k_lm_studio_window_gets_a_cut_agents_md_with_a_note() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let agents_md = workspace.join("AGENTS.md");
    let lm_studio = lm_studio_without_the_embedding_model(serde_json::json!([{
        "id": "fake-model",
        "state": "loaded",
        "max_context_length": 262_144,
        "loaded_context_length": 32_768,
    }]))
    .await;

    let (code, output) = hey_to_lm_studio(home, workspace, lm_studio.base_url.clone()).await;

    let system = first_system_prompt(&lm_studio);
    cut_section_len(&system, &agents_md, 32_768);
    assert_eq!(code, Some(0), "{output}");
}

/// A server that does not report the window: the whole AGENTS.md, as before.
#[tokio::test(flavor = "multi_thread")]
async fn an_unreported_window_gets_the_whole_agents_md() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let lm_studio = lm_studio_without_the_embedding_model(
        serde_json::json!([{ "id": "fake-model", "state": "loaded" }]),
    )
    .await;

    let (code, output) = hey_to_lm_studio(home, workspace, lm_studio.base_url.clone()).await;

    let system = first_system_prompt(&lm_studio);
    assert!(system.contains("rule 1999: keep this line"), "{system}");
    assert!(!system.contains("is cut to fit"), "{system}");
    assert_eq!(code, Some(0), "{output}");
}

/// Check that `system` carries the head of the test AGENTS.md cut for
/// `window`, with its note, in 25% of the window used to within a line.
/// Returns the section's length.
fn cut_section_len(system: &str, agents_md: &Path, window: usize) -> usize {
    let header = "# Project Context (from AGENTS.md)";
    let section = system
        .find(header)
        .map(|start| &system[start..])
        .unwrap_or_else(|| panic!("no project context section: {system}"));
    // The note's length depends on the temp path, so the kept count is
    // read from it and checked against the lines and the budget.
    let kept: usize = section
        .split_once("these are its first ")
        .and_then(|(_, rest)| rest.split_once(' '))
        .and_then(|(count, _)| count.parse().ok())
        .unwrap_or_else(|| panic!("no cut note: {section}"));
    let note = format!(
        "rule {:04}: keep this line\n\n[AGENTS.md is cut to fit this model's {window}-token context: these are its first {kept} of 2000 lines. Read {} for the rest.]",
        kept - 1,
        agents_md.display()
    );
    let section_len = section.find(&note).expect("cut note") + note.len();
    assert!(!system.contains(&format!("rule {kept:04}")), "{system}");
    // 25% of the window at 3 characters per token.
    let budget = window / 4 * 3;
    assert!(
        (budget - 100..=budget).contains(&section_len),
        "{section_len} characters"
    );
    section_len
}

/// Check that `system` carries the whole test AGENTS.md and no cut note.
fn assert_whole_agents_md(system: &str) {
    assert!(system.contains("rule 0000: keep this line\n"), "{system}");
    assert!(system.contains("rule 1999: keep this line"), "{system}");
    assert!(!system.contains("is cut to fit"), "{system}");
}

/// LM Studio with a 200K and a 32K model, both loaded.
fn big_and_small_models() -> serde_json::Value {
    serde_json::json!([
        {
            "id": "big-model",
            "state": "loaded",
            "max_context_length": 262_144,
            "loaded_context_length": 200_000,
        },
        {
            "id": "small-model",
            "state": "loaded",
            "max_context_length": 262_144,
            "loaded_context_length": 32_768,
        },
    ])
}

/// The model and system prompt of each chat request that carries `hey`.
fn hey_turns(lm_studio: &LmStudio) -> Vec<(String, String)> {
    let requests = lm_studio.chat_requests.lock().expect("lock");
    requests
        .iter()
        .filter(|request| request["messages"].to_string().contains("hey"))
        .map(|request| {
            (
                request["model"].as_str().unwrap_or_default().to_string(),
                request["messages"][0]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect()
}

/// Run the REPL on `big-model`: say `hey`, then `/model` to each of
/// `switches` and say `hey` again after each, then exit.
async fn hey_across_models(
    home: std::path::PathBuf,
    workspace: std::path::PathBuf,
    base_url: String,
    extra_env: Vec<(&'static str, String)>,
    switches: &'static [&'static str],
) -> (Option<i32>, String) {
    tokio::task::spawn_blocking(move || {
        let mut steps: Vec<(String, Vec<u8>)> = vec![
            ("[lmstudio/big-model]".to_string(), b"hey\r".to_vec()),
            ("stub answer".to_string(), Vec::new()),
        ];
        let mut current = "big-model";
        for model in switches {
            steps.push((
                format!("[lmstudio/{current}]"),
                format!("/model {model}\r").into_bytes(),
            ));
            steps.push((format!("Switched to model: {model}"), Vec::new()));
            steps.push((format!("[lmstudio/{model}]"), b"hey\r".to_vec()));
            steps.push(("stub answer".to_string(), Vec::new()));
            current = model;
        }
        steps.push((format!("[lmstudio/{current}]"), b"/exit\r".to_vec()));
        let steps: Vec<(&str, &[u8])> = steps
            .iter()
            .map(|(text, input)| (text.as_str(), input.as_slice()))
            .collect();
        let mut env: Vec<(&str, &str)> = vec![
            ("AGENT_PROVIDER", "lmstudio"),
            ("LMSTUDIO_BASE_URL", &base_url),
            ("LMSTUDIO_MODEL", "big-model"),
        ];
        env.extend(
            extra_env
                .iter()
                .map(|(name, value)| (*name, value.as_str())),
        );
        run_script(&home, &workspace, &base_url, &env, &steps)
    })
    .await
    .expect("pty run")
}

/// `/model` from a 200K model to a 32K one cuts AGENTS.md for the next
/// turn, with the note; back to 200K restores the whole file.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_switch_sizes_agents_md_for_the_new_window() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let agents_md = workspace.join("AGENTS.md");
    let lm_studio = lm_studio_without_the_embedding_model(big_and_small_models()).await;

    let (code, output) = hey_across_models(
        home,
        workspace,
        lm_studio.base_url.clone(),
        Vec::new(),
        &["small-model", "big-model"],
    )
    .await;

    let turns = hey_turns(&lm_studio);
    let models: Vec<&str> = turns.iter().map(|(model, _)| model.as_str()).collect();
    assert_eq!(
        models,
        ["big-model", "small-model", "big-model"],
        "{output}"
    );
    assert_whole_agents_md(&turns[0].1);
    cut_section_len(&turns[1].1, &agents_md, 32_768);
    assert_whole_agents_md(&turns[2].1);
    assert_eq!(code, Some(0), "{output}");
}

/// A config `context_window` of 32K wins over the 200K the server reports,
/// at startup and after a switch to another 200K-reported model.
#[tokio::test(flavor = "multi_thread")]
async fn an_explicit_window_wins_over_the_reported_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let agents_md = workspace.join("AGENTS.md");
    let config_dir = base.join("config");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::write(config_dir.join("config.toml"), "context_window = 32768\n").expect("config");
    let lm_studio = lm_studio_without_the_embedding_model(serde_json::json!([
        { "id": "big-model", "state": "loaded", "loaded_context_length": 200_000 },
        { "id": "other-big-model", "state": "loaded", "loaded_context_length": 200_000 },
    ]))
    .await;

    let (code, output) = hey_across_models(
        home,
        workspace,
        lm_studio.base_url.clone(),
        vec![("CALICHE_CONFIG_DIR", config_dir.display().to_string())],
        &["other-big-model"],
    )
    .await;

    let turns = hey_turns(&lm_studio);
    let models: Vec<&str> = turns.iter().map(|(model, _)| model.as_str()).collect();
    assert_eq!(models, ["big-model", "other-big-model"], "{output}");
    cut_section_len(&turns[0].1, &agents_md, 32_768);
    cut_section_len(&turns[1].1, &agents_md, 32_768);
    assert_eq!(code, Some(0), "{output}");
}

/// On a 200K model, `/context-window` shows the reported 200K, not the fixed
/// 32K fallback. `/context-window 16384` then reaches the embedded server
/// without a model switch: the next turn's prompt has the 16K cut.
#[tokio::test(flavor = "multi_thread")]
async fn context_window_reaches_the_server_for_the_next_turn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let agents_md = workspace.join("AGENTS.md");
    let lm_studio = lm_studio_without_the_embedding_model(big_and_small_models()).await;
    let base_url = lm_studio.base_url.clone();

    let (code, output) = tokio::task::spawn_blocking(move || {
        run_script(
            &home,
            &workspace,
            &base_url,
            &[
                ("AGENT_PROVIDER", "lmstudio"),
                ("LMSTUDIO_BASE_URL", &base_url),
                ("LMSTUDIO_MODEL", "big-model"),
            ],
            &[
                ("[lmstudio/big-model]", b"/context-window\r"),
                ("Current context window: 200.0K", b""),
                ("[lmstudio/big-model]", b"/context-window 16384\r"),
                ("Context window set to 16.4K.", b""),
                ("[lmstudio/big-model]", b"hey\r"),
                ("stub answer", b""),
                ("[lmstudio/big-model]", b"/exit\r"),
            ],
        )
    })
    .await
    .expect("pty run");

    let turns = hey_turns(&lm_studio);
    let models: Vec<&str> = turns.iter().map(|(model, _)| model.as_str()).collect();
    assert_eq!(models, ["big-model"], "{output}");
    cut_section_len(&turns[0].1, &agents_md, 16_384);
    assert_eq!(code, Some(0), "{output}");
}

/// Run `caliche <top_level> code --no-verify <task>` against the in-test
/// LM Studio on `model`, with stdin closed.
fn run_code(
    home: &Path,
    workspace: &Path,
    base_url: &str,
    model: &str,
    top_level: &[&str],
) -> std::process::Output {
    Command::new(caliche_bin())
        .current_dir(workspace)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", home)
        .env("CALICHE_NO_KEYRING", "1")
        .env("AGENT_PROVIDER", "lmstudio")
        .env("LMSTUDIO_BASE_URL", base_url)
        .env("LMSTUDIO_MODEL", model)
        .args(top_level)
        .args([
            "--no-workspace-embed",
            "code",
            "--no-verify",
            "--no-repomap",
            "say hey",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("run caliche code")
}

/// `code` against a 32K model: the head of AGENTS.md and the note, like the
/// REPL. Before, `code` always sent the whole file.
#[tokio::test(flavor = "multi_thread")]
async fn code_on_a_32k_model_gets_a_cut_agents_md_with_a_note() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let agents_md = workspace.join("AGENTS.md");
    let lm_studio = lm_studio_without_the_embedding_model(big_and_small_models()).await;
    let base_url = lm_studio.base_url.clone();

    let output = tokio::task::spawn_blocking(move || {
        run_code(&home, &workspace, &base_url, "small-model", &[])
    })
    .await
    .expect("code run");

    let log = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    let turns = hey_turns(&lm_studio);
    assert_eq!(
        turns.first().map(|(model, _)| model.as_str()),
        Some("small-model"),
        "{log}"
    );
    cut_section_len(&turns[0].1, &agents_md, 32_768);
    assert_eq!(output.status.code(), Some(0), "{log}");
}

/// `code` on a 200K model gets the whole file; with `--context-window
/// 32768` the explicit window wins and the file is cut.
#[tokio::test(flavor = "multi_thread")]
async fn code_uses_the_explicit_window_before_the_reported_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let (home, workspace) = workspace_with_a_large_agents_md(&base);
    let agents_md = workspace.join("AGENTS.md");
    let lm_studio = lm_studio_without_the_embedding_model(big_and_small_models()).await;
    let base_url = lm_studio.base_url.clone();

    let (whole, explicit) = tokio::task::spawn_blocking(move || {
        (
            run_code(&home, &workspace, &base_url, "big-model", &[]),
            run_code(
                &home,
                &workspace,
                &base_url,
                "big-model",
                &["--context-window", "32768"],
            ),
        )
    })
    .await
    .expect("code runs");

    let log = |output: &std::process::Output| {
        String::from_utf8_lossy(&output.stdout).into_owned()
            + &String::from_utf8_lossy(&output.stderr)
    };
    let turns = hey_turns(&lm_studio);
    assert_eq!(turns.len(), 2, "{}{}", log(&whole), log(&explicit));
    assert_whole_agents_md(&turns[0].1);
    cut_section_len(&turns[1].1, &agents_md, 32_768);
    assert_eq!(whole.status.code(), Some(0), "{}", log(&whole));
    assert_eq!(explicit.status.code(), Some(0), "{}", log(&explicit));
}

/// LM Studio's reply to a request for a model it has not loaded, with
/// just-in-time loading off.
const NO_MODELS_LOADED: &str =
    "No models loaded. Please load a model in the developer page or use the 'lms load' command.";

/// The embedding model the REPL asks LM Studio for by default.
const LM_STUDIO_EMBEDDING_MODEL: &str = "text-embedding-qwen3-embedding-4b";

/// How the in-test LM Studio answers a load request for one model.
#[derive(Clone, Copy)]
enum StubLoad {
    /// Accept it; the model reads `loading` this many times, then `loaded`.
    Finishes(usize),
    /// Accept it; the model never finishes loading.
    Never,
    /// Refuse it with LM Studio's out-of-memory reason.
    Fails,
}

/// An in-test LM Studio with nothing loaded: `fake-model` and the embedding
/// model are on disk, and each loads as its `StubLoad` says.
struct LoadingLmStudio {
    base_url: String,
    /// The `model` of each `POST /api/v1/models/load`.
    loads: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// The body of each `POST /v1/chat/completions`.
    chat_requests: std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

async fn lm_studio_with_nothing_loaded(
    chat_load: StubLoad,
    embedding_load: StubLoad,
) -> LoadingLmStudio {
    use axum::extract::{Path, State};
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Shared {
        how: Arc<HashMap<String, StubLoad>>,
        /// Reads left before each model is loaded; absent until its load.
        loading: Arc<Mutex<HashMap<String, usize>>>,
        loads: Arc<Mutex<Vec<String>>>,
        chat_requests: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl Shared {
        fn state(&self, id: &str, count_down: bool) -> &'static str {
            let mut loading = self.loading.lock().expect("lock");
            match (loading.get_mut(id), self.how.get(id)) {
                (None, _) | (_, Some(StubLoad::Fails) | None) => "not-loaded",
                (Some(_), Some(StubLoad::Never)) => "loading",
                (Some(0), Some(StubLoad::Finishes(_))) => "loaded",
                (Some(left), Some(StubLoad::Finishes(_))) => {
                    if count_down {
                        *left -= 1;
                    }
                    "loading"
                }
            }
        }
    }

    fn not_loaded() -> axum::response::Response {
        (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({ "error": { "message": NO_MODELS_LOADED } })),
        )
            .into_response()
    }

    async fn model_state(
        State(shared): State<Shared>,
        Path(id): Path<String>,
    ) -> axum::response::Response {
        if !shared.how.contains_key(&id) {
            return StatusCode::NOT_FOUND.into_response();
        }
        let state = shared.state(&id, true);
        axum::Json(serde_json::json!({ "id": id, "state": state, "max_context_length": 32_768 }))
            .into_response()
    }
    async fn load(
        State(shared): State<Shared>,
        axum::Json(request): axum::Json<serde_json::Value>,
    ) -> axum::response::Response {
        let id = request["model"].as_str().unwrap_or_default().to_string();
        shared.loads.lock().expect("lock").push(id.clone());
        let reads = match shared.how.get(&id) {
            Some(StubLoad::Finishes(reads)) => *reads,
            Some(StubLoad::Never) => usize::MAX,
            Some(StubLoad::Fails) | None => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": { "message": "Failed to load model: not enough memory." },
                    })),
                )
                    .into_response();
            }
        };
        shared
            .loading
            .lock()
            .expect("lock")
            .insert(id.clone(), reads);
        axum::Json(serde_json::json!({
            "type": "llm", "instance_id": id, "load_time_seconds": 1.5, "status": "loaded",
        }))
        .into_response()
    }
    async fn chat(
        State(shared): State<Shared>,
        axum::Json(request): axum::Json<serde_json::Value>,
    ) -> axum::response::Response {
        let model = request["model"].as_str().unwrap_or_default().to_string();
        let streams = request["stream"] == serde_json::Value::Bool(true);
        shared.chat_requests.lock().expect("lock").push(request);
        if shared.state(&model, false) != "loaded" {
            return not_loaded();
        }
        if !streams {
            return axum::Json(serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "fake-model",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "stub answer" },
                    "finish_reason": "stop",
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 },
            }))
            .into_response();
        }
        let chunk = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "stub answer" },
                "finish_reason": "stop",
            }],
        });
        (
            [(header::CONTENT_TYPE, "text/event-stream")],
            format!("data: {chunk}\n\ndata: [DONE]\n\n"),
        )
            .into_response()
    }
    async fn embeddings(
        State(shared): State<Shared>,
        axum::Json(request): axum::Json<serde_json::Value>,
    ) -> axum::response::Response {
        let model = request["model"].as_str().unwrap_or_default();
        if shared.state(model, false) != "loaded" {
            return not_loaded();
        }
        let count = request["input"].as_array().map_or(1, Vec::len);
        let data: Vec<serde_json::Value> = (0..count)
            .map(|index| serde_json::json!({ "object": "embedding", "embedding": [0.5, 0.5], "index": index }))
            .collect();
        axum::Json(serde_json::json!({
            "object": "list", "data": data, "model": model,
            "usage": { "prompt_tokens": 1, "total_tokens": 1 },
        }))
        .into_response()
    }
    async fn model_list() -> axum::Json<serde_json::Value> {
        axum::Json(serde_json::json!({
            "object": "list",
            "data": [
                { "id": "fake-model", "object": "model" },
                { "id": LM_STUDIO_EMBEDDING_MODEL, "object": "model" },
            ],
        }))
    }

    let shared = Shared {
        how: Arc::new(HashMap::from([
            ("fake-model".to_string(), chat_load),
            (LM_STUDIO_EMBEDDING_MODEL.to_string(), embedding_load),
        ])),
        loading: Arc::default(),
        loads: Arc::default(),
        chat_requests: Arc::default(),
    };
    let loads = Arc::clone(&shared.loads);
    let chat_requests = Arc::clone(&shared.chat_requests);
    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(chat))
        .route("/v1/embeddings", axum::routing::post(embeddings))
        .route("/v1/models", axum::routing::get(model_list))
        .route("/api/v0/models/{id}", axum::routing::get(model_state))
        .route("/api/v1/models/load", axum::routing::post(load))
        .with_state(shared);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
    tokio::spawn(async move { axum::serve(listener, app).await });
    LoadingLmStudio {
        base_url,
        loads,
        chat_requests,
    }
}

/// Run the thin-client REPL against `lm_studio` with `steps`.
async fn run_on_lm_studio(
    lm_studio: &LoadingLmStudio,
    steps: Vec<(&'static str, &'static [u8])>,
    ctrl_c: CtrlC,
) -> (Option<i32>, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(
        workspace.join("notes.md"),
        "# Notes\n\nOne file to index.\n",
    )
    .expect("file");
    let base_url = lm_studio.base_url.clone();
    tokio::task::spawn_blocking(move || {
        let _dir = dir;
        run_script_with_ctrl_c(
            &home,
            &workspace,
            &base_url,
            &[
                ("AGENT_PROVIDER", "lmstudio"),
                ("LMSTUDIO_BASE_URL", &base_url),
                ("LMSTUDIO_MODEL", "fake-model"),
            ],
            &steps,
            ctrl_c,
        )
    })
    .await
    .expect("pty run")
}

/// Nothing loaded: the REPL asks LM Studio to load the chat model, shows
/// one status line, and sends the turn once after the load. The embedding
/// model is loaded too, so workspace search starts with no warning.
#[tokio::test(flavor = "multi_thread")]
async fn a_turn_loads_the_lm_studio_model_and_then_answers() {
    let lm_studio =
        lm_studio_with_nothing_loaded(StubLoad::Finishes(3), StubLoad::Finishes(1)).await;

    let (code, output) = run_on_lm_studio(
        &lm_studio,
        vec![
            (LM_STUDIO_PROMPT, b"hey\r"),
            ("stub answer", b""),
            (LM_STUDIO_PROMPT, b"/exit\r"),
        ],
        CtrlC::AsAKey,
    )
    .await;

    assert_eq!(
        output.matches("Loading fake-model in LM Studio...").count(),
        1,
        "{output}"
    );
    let status_at = output
        .find("Loading fake-model in LM Studio...")
        .expect("status");
    let answer_at = output.find("stub answer").expect("answer");
    assert!(status_at < answer_at, "{output}");
    assert!(!output.contains("Workspace search is off"), "{output}");
    assert!(!output.contains(NO_MODELS_LOADED), "{output}");
    let mut loads = lm_studio.loads.lock().expect("lock").clone();
    loads.sort();
    assert_eq!(loads, ["fake-model", LM_STUDIO_EMBEDDING_MODEL], "{output}");
    let hey_turns = lm_studio
        .chat_requests
        .lock()
        .expect("lock")
        .iter()
        .filter(|request| request["messages"].to_string().contains("hey"))
        .count();
    assert_eq!(hey_turns, 1, "{output}");
    assert_eq!(code, Some(0), "{output}");
}

/// LM Studio refuses to load the embedding model: one warning line that
/// quotes its reason once, with no doubled period, and the turn still
/// answers.
#[tokio::test(flavor = "multi_thread")]
async fn a_refused_embedding_load_turns_workspace_search_off_with_one_line() {
    let lm_studio = lm_studio_with_nothing_loaded(StubLoad::Finishes(0), StubLoad::Fails).await;

    let (code, output) = run_on_lm_studio(
        &lm_studio,
        vec![
            (LM_STUDIO_PROMPT, b"hey\r"),
            ("stub answer", b""),
            (LM_STUDIO_PROMPT, b"/exit\r"),
        ],
        CtrlC::AsAKey,
    )
    .await;

    let warning = "Workspace search is off for this session: LM Studio could not load text-embedding-qwen3-embedding-4b: lmstudio API error (500 Internal Server Error): Failed to load model: not enough memory. Load or download the embedding model, or start with --no-workspace-embed.";
    assert_eq!(output.matches(warning).count(), 1, "{output}");
    assert_eq!(output.matches("not enough memory").count(), 1, "{output}");
    assert!(!output.contains(".. Load or download"), "{output}");
    assert!(output.contains("stub answer"), "{output}");
    assert_eq!(code, Some(0), "{output}");
}

/// Ctrl+C while LM Studio loads the model cancels the wait and the turn:
/// the cancel line shows, the prompt returns, and no chat request is sent.
#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_while_lm_studio_loads_returns_to_the_prompt() {
    let lm_studio = lm_studio_with_nothing_loaded(StubLoad::Never, StubLoad::Finishes(0)).await;

    let (code, output) = run_on_lm_studio(
        &lm_studio,
        vec![
            (LM_STUDIO_PROMPT, b"hey\r"),
            ("Loading fake-model in LM Studio...", b"\x03"),
            ("Cancelled.", b""),
            (LM_STUDIO_PROMPT, b"/exit\r"),
        ],
        CtrlC::DuringATurn,
    )
    .await;

    let after_status = output
        .split("Loading fake-model in LM Studio...")
        .last()
        .unwrap_or("");
    assert!(after_status.contains("Cancelled."), "{output}");
    assert!(after_status.contains(LM_STUDIO_PROMPT), "{output}");
    let chat_loads = lm_studio
        .loads
        .lock()
        .expect("lock")
        .iter()
        .filter(|model| *model == "fake-model")
        .count();
    // The status line shows before the load request goes out, so Ctrl+C can
    // cancel the turn before or after that request: never more than one.
    assert!(chat_loads <= 1, "{chat_loads} load requests\n{output}");
    let hey_turns = lm_studio
        .chat_requests
        .lock()
        .expect("lock")
        .iter()
        .filter(|request| request["messages"].to_string().contains("hey"))
        .count();
    assert_eq!(hey_turns, 0, "{output}");
    assert_eq!(code, Some(0), "{output}");
}

/// `code` with nothing loaded: one load request for the chat model, then the
/// task is sent once and the run succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn code_loads_the_lm_studio_model_before_the_task() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().canonicalize().expect("canonical");
    let home = base.join("home");
    let workspace = base.join("workspace");
    std::fs::create_dir_all(&home).expect("home");
    std::fs::create_dir_all(&workspace).expect("workspace");
    let lm_studio =
        lm_studio_with_nothing_loaded(StubLoad::Finishes(2), StubLoad::Finishes(0)).await;
    let base_url = lm_studio.base_url.clone();

    let output = tokio::task::spawn_blocking(move || {
        run_code(&home, &workspace, &base_url, "fake-model", &[])
    })
    .await
    .expect("code run");

    let log = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        *lm_studio.loads.lock().expect("lock"),
        ["fake-model"],
        "{log}"
    );
    let hey_turns = lm_studio
        .chat_requests
        .lock()
        .expect("lock")
        .iter()
        .filter(|request| request["messages"].to_string().contains("say hey"))
        .count();
    assert_eq!(hey_turns, 1, "{log}");
    assert!(!log.contains(NO_MODELS_LOADED), "{log}");
    assert_eq!(output.status.code(), Some(0), "{log}");
}
