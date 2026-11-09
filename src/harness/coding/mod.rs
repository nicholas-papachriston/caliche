//! Shared coding-task workflow built on top of [`AgentHarness`].
//!
//! This module is intentionally UI/API/CLI-neutral. Frontends should build a
//! [`CodingTaskRequest`], execute it through [`CodingTaskExecutor`], and render
//! the returned [`CodingTaskResult`] plus regular harness events.

use crate::harness::{
    AgentEvent, AgentHarness, HarnessConfig, SessionDiffSummary, TaskResult,
    config::PermissionModeConfig,
};
use crate::llm::ProviderType;
use crate::messages::TokenUsage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;

mod panic_scan;

/// High-level coding behavior requested by a frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CodingMode {
    /// Answer or inspect only; edits should be denied by the harness config.
    Ask,
    /// Produce a concrete implementation plan without editing.
    Plan,
    /// Apply edits and return a diff/verification report.
    #[default]
    Edit,
    /// Apply edits, verify, and allow future retry policies to repair failures.
    FullAuto,
}

/// Verification behavior after a coding task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VerificationPolicy {
    /// Do not run verification.
    None,
    /// Infer a lightweight verification command from the workspace.
    #[default]
    Auto,
    /// Run the provided commands in order.
    Commands(Vec<String>),
}

/// Context construction behavior for a coding task.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodingContextPolicy {
    /// Optional context block prepended by the caller, e.g. a repo map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_prefix: Option<String>,
}

/// Request for a coding task. This is the shared contract used by CLI/API/UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTaskRequest {
    pub task: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_root: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub mode: CodingMode,
    #[serde(default)]
    pub permission_mode: PermissionModeConfig,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    #[serde(default)]
    pub verify: VerificationPolicy,
    #[serde(default)]
    pub context: CodingContextPolicy,
}

fn default_max_turns() -> usize {
    50
}

/// Contract prepended to edit/full-auto coding tasks so the model finishes in one loop.
const CODING_TASK_CONTRACT: &str = "Complete this coding task in this conversation. Call tools immediately; do not announce upcoming actions. Do not re-read files already in context. Match the surrounding file's indentation, quoting, naming, and imports; do not restyle unrelated code. Apply large edits yourself in this turn with multi-hunk fs edit or apply_patch — do not ask the user to paste patches or split the work. After edits, run the project's real test command (cargo test, bun test, pytest, etc.) rather than compile-only checks when tests exist, and fix failures before finishing. Do not use unwrap, expect, panic, todo, unimplemented, or unreachable in library code.";

const REUSED_VERIFICATION: &str = "reused in-session result after last edit";

impl CodingTaskRequest {
    /// Convert request-level execution options into a harness configuration.
    pub fn to_harness_config(&self) -> HarnessConfig {
        HarnessConfig {
            provider: self.provider,
            model: self.model.clone(),
            tools: self.tools.clone(),
            workspace_root: self.workspace_root.clone(),
            max_turns: self.max_turns,
            permission_mode: self.permission_mode,
            ..Default::default()
        }
    }
}

/// Lightweight workspace facts gathered before a coding task starts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceProfile {
    pub root: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_root: Option<PathBuf>,
    #[serde(default)]
    pub project_kinds: Vec<ProjectKind>,
    #[serde(default)]
    pub default_check_commands: Vec<String>,
    #[serde(default)]
    pub dirty_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectKind {
    Rust,
    Node,
    Python,
    Unknown,
}

/// One verification command execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationRun {
    pub command: String,
    pub exit_code: Option<i32>,
    pub stdout_tail: String,
    pub stderr_tail: String,
    pub duration_ms: u64,
}

/// Verification summary for a coding task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationReport {
    pub commands: Vec<VerificationRun>,
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skipped_reason: Option<String>,
}

/// Shared coding-task result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTaskResult {
    pub session_id: String,
    pub status: CodingTaskStatus,
    pub final_response: String,
    pub changed_files: Vec<PathBuf>,
    pub diff_summary: SessionDiffSummary,
    pub workspace: WorkspaceProfile,
    pub verification: VerificationReport,
    pub usage: TokenUsage,
    pub turns: usize,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingTaskStatus {
    Planned,
    Completed,
    VerificationFailed,
}

/// Executes shared coding workflows using an [`AgentHarness`].
pub struct CodingTaskExecutor {
    harness: AgentHarness,
}

impl CodingTaskExecutor {
    pub async fn new(config: HarnessConfig) -> Result<Self> {
        Ok(Self {
            harness: AgentHarness::new(config).await?,
        })
    }

    pub fn from_harness(harness: AgentHarness) -> Self {
        Self { harness }
    }

    pub fn harness(&self) -> &AgentHarness {
        &self.harness
    }

    pub fn harness_mut(&mut self) -> &mut AgentHarness {
        &mut self.harness
    }

    pub async fn run(&mut self, request: &CodingTaskRequest) -> Result<CodingTaskResult> {
        run_with_harness(&mut self.harness, request).await
    }
}

/// Execute a coding task against an existing harness.
pub async fn run_with_harness(
    harness: &mut AgentHarness,
    request: &CodingTaskRequest,
) -> Result<CodingTaskResult> {
    let start = Instant::now();
    let workspace_root = request
        .workspace_root
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let profile = profile_workspace(&workspace_root).await;
    let verify_commands = verification_command_list(&profile, &request.verify);

    harness.event_stream().emit(AgentEvent::Info {
        message: format!("coding task started in {}", profile.root.to_string_lossy()),
    });

    let effective_task = build_effective_task(request, &verify_commands);
    let mut task_result = match request.mode {
        CodingMode::Plan => run_plan_only(harness, &effective_task).await?,
        CodingMode::Ask | CodingMode::Edit | CodingMode::FullAuto => {
            harness.run_task(&effective_task).await?
        }
    };

    let ledger = harness.command_ledger().read().await.clone();
    let mut diff_summary = harness.session_diff_summary().await;
    let mut verification = if matches!(request.mode, CodingMode::Plan | CodingMode::Ask) {
        VerificationReport {
            commands: Vec::new(),
            passed: true,
            skipped_reason: Some("verification skipped for non-edit mode".to_string()),
        }
    } else {
        run_verification(
            &profile,
            &request.verify,
            harness.event_stream(),
            Some(&ledger),
            &diff_summary.files_changed,
        )
        .await
    };

    if !verification.passed {
        apply_verification_repairs(
            harness,
            request,
            &mut task_result,
            &mut verification,
            &profile,
            &mut diff_summary,
        )
        .await?;
    }

    if !diff_summary.files_changed.is_empty() {
        task_result
            .files_modified
            .clone_from(&diff_summary.files_changed);
    }

    let status = if matches!(request.mode, CodingMode::Plan) {
        CodingTaskStatus::Planned
    } else if verification.passed {
        CodingTaskStatus::Completed
    } else {
        CodingTaskStatus::VerificationFailed
    };

    Ok(CodingTaskResult {
        session_id: harness.session_id().to_string(),
        status,
        final_response: task_result.response,
        changed_files: task_result.files_modified,
        diff_summary,
        workspace: profile,
        verification,
        usage: task_result.total_usage,
        turns: task_result.turns,
        duration_ms: start.elapsed().as_millis() as u64,
        checkpoint_id: task_result.checkpoint_id,
    })
}

async fn run_plan_only(harness: &mut AgentHarness, task: &str) -> Result<TaskResult> {
    let prompt = format!(
        "Create a concrete implementation plan for this coding task. Do not edit files.\n\n{task}"
    );
    let turn = harness.run_turn(&prompt).await?;
    Ok(TaskResult {
        response: turn.response,
        total_usage: *harness.total_usage(),
        turns: 1,
        duration_ms: turn.duration_ms,
        files_modified: Vec::new(),
        checkpoint_id: None,
        discipline: turn.discipline,
    })
}

fn build_effective_task(request: &CodingTaskRequest, verify_commands: &[String]) -> String {
    let task = match request.context.context_prefix.as_deref() {
        Some(prefix) if !prefix.trim().is_empty() => {
            format!("{}\n\n[Task]\n{}", prefix.trim(), request.task)
        }
        _ => request.task.clone(),
    };
    if matches!(request.mode, CodingMode::Edit | CodingMode::FullAuto) {
        let mut contract = CODING_TASK_CONTRACT.to_string();
        if !verify_commands.is_empty() {
            let _ = write!(
                contract,
                " Run exactly these verification commands after the last edit: {}.",
                verify_commands.join("; ")
            );
        }
        format!("{contract}\n\n{task}")
    } else {
        task
    }
}

fn verification_repair_prompt(report: &VerificationReport) -> String {
    let mut body = String::from(
        "Verification failed after your edits. Do not restart the original task. Fix the cause and re-run the failing command.\n",
    );
    for run in &report.commands {
        if run.exit_code == Some(0) {
            continue;
        }
        let _ = writeln!(body, "\nCommand: {}", run.command);
        if let Some(code) = run.exit_code {
            let _ = writeln!(body, "Exit code: {code}");
        }
        let stderr = run.stderr_tail.trim();
        if !stderr.is_empty() {
            let _ = writeln!(body, "stderr:\n{stderr}");
        }
        let stdout = run.stdout_tail.trim();
        if !stdout.is_empty() {
            let _ = writeln!(body, "stdout:\n{stdout}");
        }
    }
    body
}

const fn max_verification_repairs(mode: CodingMode, remaining_turns: usize) -> usize {
    match mode {
        CodingMode::Edit => {
            if remaining_turns == 0 {
                0
            } else {
                1
            }
        }
        CodingMode::FullAuto => remaining_turns,
        CodingMode::Ask | CodingMode::Plan => 0,
    }
}

async fn apply_verification_repairs(
    harness: &mut AgentHarness,
    request: &CodingTaskRequest,
    task_result: &mut TaskResult,
    verification: &mut VerificationReport,
    profile: &WorkspaceProfile,
    diff_summary: &mut SessionDiffSummary,
) -> Result<()> {
    let remaining = request.max_turns.saturating_sub(harness.turn_count());
    let max_repairs = max_verification_repairs(request.mode, remaining);
    let mut repairs = 0usize;
    while !verification.passed && repairs < max_repairs {
        let prompt = verification_repair_prompt(verification);
        harness.event_stream().emit(AgentEvent::Info {
            message: format!(
                "verification repair {}/{max_repairs}",
                repairs.saturating_add(1)
            ),
        });
        let turn = harness.run_turn(&prompt).await?;
        task_result.response = turn.response;
        task_result.turns = task_result.turns.saturating_add(1);
        task_result.total_usage = (task_result.total_usage + turn.usage).ensure_total_tokens();
        *diff_summary = harness.session_diff_summary().await;
        let ledger = harness.command_ledger().read().await.clone();
        *verification = run_verification(
            profile,
            &request.verify,
            harness.event_stream(),
            Some(&ledger),
            &diff_summary.files_changed,
        )
        .await;
        repairs = repairs.saturating_add(1);
    }
    Ok(())
}

pub async fn profile_workspace(root: &Path) -> WorkspaceProfile {
    let root = root.to_path_buf();
    let mut project_kinds = Vec::new();
    let mut default_check_commands = Vec::new();

    if root.join("Cargo.toml").exists() {
        project_kinds.push(ProjectKind::Rust);
        default_check_commands.push("cargo test".to_string());
    }
    if root.join("package.json").exists() {
        project_kinds.push(ProjectKind::Node);
        default_check_commands.extend(node_verify_commands(&root));
    }
    if root.join("pyproject.toml").exists() {
        project_kinds.push(ProjectKind::Python);
        default_check_commands.extend(python_verify_commands(&root));
    }
    if project_kinds.is_empty() {
        project_kinds.push(ProjectKind::Unknown);
    }

    let git_root = git_root(&root).await;
    let dirty_files = git_dirty_files(&root).await.unwrap_or_default();

    WorkspaceProfile {
        root,
        git_root,
        project_kinds,
        default_check_commands,
        dirty_files,
    }
}

fn node_verify_commands(root: &Path) -> Vec<String> {
    if root.join("bun.lock").exists() || root.join("bun.lockb").exists() {
        return vec!["bun test".to_string()];
    }
    if !package_json_has_test_script(root) {
        return Vec::new();
    }
    if root.join("pnpm-lock.yaml").exists() {
        return vec!["pnpm test".to_string()];
    }
    if root.join("yarn.lock").exists() {
        return vec!["yarn test".to_string()];
    }
    vec!["npm test".to_string()]
}

fn python_verify_commands(root: &Path) -> Vec<String> {
    if root.join("tests").is_dir() || root.join("test").is_dir() || pyproject_mentions_pytest(root)
    {
        return vec!["uv run pytest".to_string()];
    }
    Vec::new()
}

fn package_json_has_test_script(root: &Path) -> bool {
    let path = root.join("package.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value
        .get("scripts")
        .and_then(|scripts| scripts.get("test"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|script| !script.trim().is_empty())
}

fn pyproject_mentions_pytest(root: &Path) -> bool {
    let path = root.join("pyproject.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return false;
    };
    text.contains("pytest")
}

/// The repository work tree containing `root`, read with libgit2. It runs no
/// command that repository config names (filters, hooks, fsmonitor), so a
/// repository the model wrote cannot make the host run one.
#[cfg(feature = "tools-git")]
async fn git_root(root: &Path) -> Option<PathBuf> {
    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let repo = git2::Repository::discover(&root).ok()?;
        repo.workdir()
            .map(|dir| dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()))
    })
    .await
    .ok()
    .flatten()
}

/// Without libgit2 the agent does not run host `git`: no repository root.
#[cfg(not(feature = "tools-git"))]
#[expect(clippy::unused_async, reason = "same signature as the libgit2 build")]
async fn git_root(_root: &Path) -> Option<PathBuf> {
    None
}

/// Changed and untracked paths, relative to the repository root, read with
/// libgit2 like [`git_root`].
#[cfg(feature = "tools-git")]
async fn git_dirty_files(root: &Path) -> Result<Vec<PathBuf>> {
    let root = root.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let Ok(repo) = git2::Repository::discover(&root) else {
            return Ok(Vec::new());
        };
        let mut options = git2::StatusOptions::new();
        options
            .include_untracked(true)
            .recurse_untracked_dirs(false)
            .include_ignored(false);
        let statuses = repo
            .statuses(Some(&mut options))
            .context("failed to read git status")?;
        Ok(statuses
            .iter()
            .filter(|entry| entry.status() != git2::Status::CURRENT)
            .filter_map(|entry| entry.path().map(PathBuf::from))
            .collect())
    })
    .await
    .context("git status task failed")?
}

/// Without libgit2 the agent does not run host `git`: no dirty files.
#[cfg(not(feature = "tools-git"))]
#[expect(clippy::unused_async, reason = "same signature as the libgit2 build")]
async fn git_dirty_files(_root: &Path) -> Result<Vec<PathBuf>> {
    Ok(Vec::new())
}

fn verification_command_list(
    profile: &WorkspaceProfile,
    policy: &VerificationPolicy,
) -> Vec<String> {
    match policy {
        VerificationPolicy::None => Vec::new(),
        VerificationPolicy::Auto => profile.default_check_commands.clone(),
        VerificationPolicy::Commands(commands) => commands.clone(),
    }
}

fn reused_verification_run(command: String) -> VerificationRun {
    VerificationRun {
        command,
        exit_code: Some(0),
        stdout_tail: REUSED_VERIFICATION.to_string(),
        stderr_tail: String::new(),
        duration_ms: 0,
    }
}

async fn run_verification(
    profile: &WorkspaceProfile,
    policy: &VerificationPolicy,
    events: &crate::harness::EventStream,
    ledger: Option<&crate::harness::CommandLedger>,
    changed_files: &[PathBuf],
) -> VerificationReport {
    if matches!(policy, VerificationPolicy::None) {
        return VerificationReport {
            commands: Vec::new(),
            passed: true,
            skipped_reason: Some("verification disabled".to_string()),
        };
    }

    let commands = verification_command_list(profile, policy);
    let mut runs = Vec::new();
    let mut passed = true;
    let mut reused = 0usize;
    for command in &commands {
        if ledger.is_some_and(|ledger| ledger.covers(command)) {
            events.emit(AgentEvent::Info {
                message: format!("verification reused: {command}"),
            });
            reused = reused.saturating_add(1);
            runs.push(reused_verification_run(command.clone()));
            continue;
        }
        events.emit(AgentEvent::Info {
            message: format!("verification started: {command}"),
        });
        let run = run_shell_command(&profile.root, command).await;
        if run.exit_code != Some(0) {
            passed = false;
        }
        events.emit(AgentEvent::Info {
            message: format!("verification finished: {} ({:?})", command, run.exit_code),
        });
        runs.push(run);
    }

    if let Some(scan) = panic_scan::panic_path_verification(&profile.root, changed_files) {
        events.emit(AgentEvent::Info {
            message: "verification started: panic-path scan".to_string(),
        });
        events.emit(AgentEvent::Info {
            message: "verification finished: panic-path scan (Some(1))".to_string(),
        });
        passed = false;
        runs.push(scan);
    }

    if runs.is_empty() {
        return VerificationReport {
            commands: Vec::new(),
            passed: true,
            skipped_reason: Some("no verification command detected".to_string()),
        };
    }

    let skipped_reason =
        if !commands.is_empty() && reused == commands.len() && runs.len() == commands.len() {
            Some(REUSED_VERIFICATION.to_string())
        } else {
            None
        };

    VerificationReport {
        commands: runs,
        passed,
        skipped_reason,
    }
}

async fn run_shell_command(root: &Path, command: &str) -> VerificationRun {
    let start = Instant::now();
    let output = Command::new("sh")
        .arg("-lc")
        .arg(command)
        .current_dir(root)
        .env_clear()
        .envs(caliche_contracts::child_env::host_child_env())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;

    match output {
        Ok(output) => VerificationRun {
            command: command.to_string(),
            exit_code: output.status.code(),
            stdout_tail: tail_lossy(&output.stdout, 4096),
            stderr_tail: tail_lossy(&output.stderr, 4096),
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Err(err) => VerificationRun {
            command: command.to_string(),
            exit_code: None,
            stdout_tail: String::new(),
            stderr_tail: err.to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        },
    }
}

fn tail_lossy(bytes: &[u8], max: usize) -> String {
    let start = bytes.len().saturating_sub(max);
    String::from_utf8_lossy(&bytes[start..]).to_string()
}

#[cfg(test)]
mod tests {
    /// Every repository config key that names a program, set to a command
    /// that leaves a marker file named after the key.
    #[cfg(all(unix, feature = "tools-git"))]
    const PROGRAM_KEYS: &[&str] = &[
        "filter.probe.clean",
        "filter.probe.smudge",
        "filter.probe.process",
        "diff.probe.textconv",
        "diff.external",
        "gpg.program",
        "gpg.ssh.program",
        "gpg.x509.program",
        crate::policy::FSMONITOR_KEY,
        "core.sshCommand",
        "core.pager",
        "core.editor",
        "sequence.editor",
        "credential.helper",
    ];

    /// Hooks planted both in `.git/hooks` and in a `core.hooksPath` dir.
    #[cfg(all(unix, feature = "tools-git"))]
    const HOOKS: &[&str] = &[
        "pre-commit",
        "post-checkout",
        "post-index-change",
        "reference-transaction",
        "fsmonitor-watchman",
    ];

    /// A repository with a committed `a.txt`, then a same-size edit (so a
    /// reader must hash the file through its filters), and every program key
    /// and hook planted. Returns the repository root and the marker dir.
    #[cfg(all(unix, feature = "tools-git"))]
    fn planted_repo() -> (tempfile::TempDir, PathBuf, tempfile::TempDir) {
        use std::os::unix::fs::PermissionsExt;
        let markers = tempfile::tempdir().expect("markers");
        let repo = tempfile::tempdir().expect("repo");
        let root = repo.path().canonicalize().expect("canonical");
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(&root)
                .output()
                .expect("git");
            assert!(out.status.success(), "{args:?}: {out:?}");
        };
        git(&["init", "-q"]);
        std::fs::write(root.join("a.txt"), "aaaa\n").expect("write");
        git(&["add", "a.txt"]);
        git(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "m",
        ]);
        std::fs::write(root.join(".gitattributes"), "* filter=probe diff=probe\n").expect("attrs");
        for key in PROGRAM_KEYS {
            let command = format!("sh -c 'touch {}/{key}; cat'", markers.path().display());
            git(&["config", key, &command]);
        }
        git(&["config", "commit.gpgsign", "true"]);
        git(&["config", "filter.probe.required", "true"]);
        let hooks_path = root.join(".git/probe-hooks");
        std::fs::create_dir_all(&hooks_path).expect("hooks dir");
        git(&[
            "config",
            "core.hooksPath",
            hooks_path.to_str().expect("utf-8"),
        ]);
        for dir in [root.join(".git/hooks"), hooks_path] {
            for hook in HOOKS {
                let script = dir.join(hook);
                let body = format!(
                    "#!/bin/sh\ntouch {}/hook-{hook}\n",
                    markers.path().display()
                );
                std::fs::write(&script, body).expect("hook");
                std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                    .expect("chmod");
            }
        }
        std::fs::write(root.join("a.txt"), "bbbb\n").expect("same-size edit");
        (repo, root, markers)
    }

    #[cfg(all(unix, feature = "tools-git"))]
    fn markers_left(markers: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(markers)
            .expect("markers")
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    }

    /// No agent-initiated git read (the verification root and dirty files,
    /// the harness git context) runs a program that repository config or a
    /// hook names, and each still reports the change.
    #[cfg(all(unix, feature = "tools-git"))]
    #[tokio::test]
    async fn agent_git_reads_run_no_repo_configured_program() {
        let (_repo, root, markers) = planted_repo();

        let top = git_root(&root).await;
        let dirty = git_dirty_files(&root).await.expect("dirty files");
        let mut context = crate::harness::context::ContextManager::new(root.clone());
        let git_context = context.refresh_git_context().await.expect("git context");

        assert_eq!(markers_left(markers.path()), Vec::<String>::new());
        assert_eq!(top, Some(root.clone()));
        assert!(dirty.contains(&PathBuf::from("a.txt")), "{dirty:?}");
        assert_eq!(git_context.map(|c| c.modified_count), Some(1));
    }

    /// A submodule gitfile pointing at a git dir with a planted fsmonitor
    /// (the sandbox-side attack) runs nothing through the agent's reads.
    #[cfg(all(unix, feature = "tools-git"))]
    #[tokio::test]
    async fn agent_git_reads_ignore_a_planted_submodule_fsmonitor() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path().canonicalize().expect("canonical");
        let marker = base.join("PWNED");
        let git = |dir: &Path, args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
                .args(["-c", "protocol.file.allow=always"])
                .args(args)
                .current_dir(dir)
                .output()
                .expect("git");
            assert!(out.status.success(), "{args:?}: {out:?}");
        };
        let lib = base.join("lib");
        let root = base.join("top");
        std::fs::create_dir(&lib).expect("lib");
        std::fs::create_dir(&root).expect("top");
        git(&lib, &["init", "-q"]);
        git(&lib, &["commit", "-q", "--allow-empty", "-m", "init"]);
        git(&root, &["init", "-q"]);
        git(&root, &["submodule", "add", "-q", "../lib", "sub"]);
        git(&root, &["commit", "-q", "-m", "add"]);
        let evil = root.join("sub/.evil");
        let copied = std::process::Command::new("cp")
            .arg("-R")
            .arg(root.join(".git/modules/sub"))
            .arg(&evil)
            .status()
            .expect("cp");
        assert!(copied.success());
        let evil_config = evil.join("config");
        let evil_config = evil_config.to_str().expect("utf-8");
        git(
            &base,
            &["config", "--file", evil_config, "--unset", "core.worktree"],
        );
        let fsmonitor = format!("touch {}; false", marker.display());
        git(
            &base,
            &[
                "config",
                "--file",
                evil_config,
                crate::policy::FSMONITOR_KEY,
                &fsmonitor,
            ],
        );
        std::fs::write(
            root.join("sub/.git"),
            format!("{} .evil\n", crate::policy::GITFILE_PREFIX),
        )
        .expect("gitfile");
        std::fs::write(root.join("sub/new.txt"), "x").expect("dirty submodule");

        let dirty = git_dirty_files(&root).await.expect("dirty files");
        let mut context = crate::harness::context::ContextManager::new(root.clone());
        let git_context = context.refresh_git_context().await.expect("git context");

        assert!(!marker.exists(), "planted fsmonitor ran");
        assert!(dirty.contains(&PathBuf::from("sub")), "{dirty:?}");
        assert!(git_context.is_some());
    }

    /// libgit2 itself runs no external filter or diff driver: hashing the
    /// work tree through the attribute filters, staging into an in-memory
    /// index, and rendering a patch leave no marker.
    #[cfg(all(unix, feature = "tools-git"))]
    #[test]
    fn libgit2_runs_no_external_filter_or_diff_driver() {
        let (_repo, root, markers) = planted_repo();
        let repo = git2::Repository::open(&root).expect("open");

        let status = repo.status_file(Path::new("a.txt")).expect("status");
        let diff = repo.diff_index_to_workdir(None, None).expect("diff");
        let mut patch = String::new();
        diff.print(git2::DiffFormat::Patch, |_, _, line| {
            patch.push_str(&String::from_utf8_lossy(line.content()));
            true
        })
        .expect("print");
        repo.index()
            .expect("index")
            .add_path(Path::new("a.txt"))
            .expect("in-memory add");

        assert_eq!(markers_left(markers.path()), Vec::<String>::new());
        assert_eq!(status, git2::Status::WT_MODIFIED);
        assert!(patch.contains("bbbb"), "{patch}");
    }

    use super::*;
    use crate::agent::service::AgentService;
    use crate::agent::tools::ToolRegistry;
    use crate::harness::config::{HarnessConfig, PermissionModeConfig};
    use crate::llm::ProviderType;
    use crate::messages::MessageRole;
    use crate::testing::MockProvider;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn mock_harness(responses: Vec<String>, max_turns: usize) -> AgentHarness {
        let provider = Arc::new(MockProvider::new(responses));
        let tools = Arc::new(ToolRegistry::new());
        let agent = crate::agent::Agent::new(provider, tools);
        let service =
            AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service");
        AgentHarness::from_service(
            service,
            HarnessConfig {
                max_turns,
                auto_checkpoint: false,
                permission_mode: PermissionModeConfig::AcceptAll,
                ..HarnessConfig::default()
            },
        )
        .expect("harness from mock service")
    }

    fn user_texts(harness: &AgentHarness) -> Vec<String> {
        harness
            .history()
            .iter()
            .filter(|message| matches!(message.role, MessageRole::User))
            .filter_map(|message| message.as_text().map(ToOwned::to_owned))
            .collect()
    }

    fn edit_request(
        task: &str,
        root: PathBuf,
        max_turns: usize,
        command: &str,
    ) -> CodingTaskRequest {
        CodingTaskRequest {
            task: task.to_string(),
            workspace_root: Some(root),
            provider: None,
            model: None,
            mode: CodingMode::Edit,
            permission_mode: PermissionModeConfig::AcceptAll,
            max_turns,
            tools: None,
            verify: VerificationPolicy::Commands(vec![command.to_string()]),
            context: CodingContextPolicy::default(),
        }
    }

    #[test]
    fn verification_repair_prompt_includes_failing_command_output() {
        let report = VerificationReport {
            commands: vec![VerificationRun {
                command: "cargo check".to_string(),
                exit_code: Some(101),
                stdout_tail: String::new(),
                stderr_tail: "error[E0425]: cannot find value `x`".to_string(),
                duration_ms: 12,
            }],
            passed: false,
            skipped_reason: None,
        };
        let prompt = verification_repair_prompt(&report);
        assert!(prompt.contains("Verification failed"));
        assert!(prompt.contains("cargo check"));
        assert!(prompt.contains("Exit code: 101"));
        assert!(prompt.contains("cannot find value `x`"));
        assert!(prompt.contains("Do not restart the original task"));
    }

    #[test]
    fn edit_mode_allows_one_repair_full_auto_uses_remaining_turns() {
        assert_eq!(max_verification_repairs(CodingMode::Edit, 4), 1);
        assert_eq!(max_verification_repairs(CodingMode::FullAuto, 4), 4);
        assert_eq!(max_verification_repairs(CodingMode::Ask, 4), 0);
        assert_eq!(max_verification_repairs(CodingMode::Edit, 0), 0);
    }

    #[tokio::test]
    async fn edit_mode_repairs_once_after_verification_failure() {
        let tmp = tempdir().expect("tempdir");
        let mut harness = mock_harness(
            vec![
                "I added the helper.".into(),
                "I fixed the compile error.".into(),
                "should not be requested".into(),
            ],
            4,
        );
        let request = edit_request(
            "add a helper",
            tmp.path().to_path_buf(),
            4,
            "sh -c 'echo compile error >&2; exit 1'",
        );
        let result = run_with_harness(&mut harness, &request)
            .await
            .expect("coding task");

        assert_eq!(result.turns, 2);
        assert_eq!(result.status, CodingTaskStatus::VerificationFailed);
        assert!(!result.verification.passed);
        let users = user_texts(&harness);
        assert_eq!(users.len(), 2);
        assert!(users[0].contains("add a helper"));
        assert!(users[0].contains("Complete this coding task"));
        assert!(users[1].contains("Verification failed"));
        assert!(users[1].contains("compile error"));
        assert_eq!(
            users
                .iter()
                .filter(|text| text.contains("add a helper"))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn full_auto_keeps_repairing_until_turn_budget() {
        let tmp = tempdir().expect("tempdir");
        let mut harness = mock_harness(
            vec![
                "first attempt".into(),
                "second attempt".into(),
                "third attempt".into(),
            ],
            3,
        );
        let mut request = edit_request(
            "add a helper",
            tmp.path().to_path_buf(),
            3,
            "sh -c 'echo still broken >&2; exit 1'",
        );
        request.mode = CodingMode::FullAuto;
        let result = run_with_harness(&mut harness, &request)
            .await
            .expect("coding task");

        assert_eq!(result.turns, 3);
        assert_eq!(result.status, CodingTaskStatus::VerificationFailed);
        let users = user_texts(&harness);
        assert_eq!(users.len(), 3);
        assert_eq!(
            users
                .iter()
                .filter(|text| text.contains("Verification failed"))
                .count(),
            2
        );
    }

    #[tokio::test]
    async fn rust_workspace_auto_verification_runs_cargo_test() {
        let tmp = tempdir().expect("tempdir");
        std::fs::write(
            tmp.path().join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("write Cargo.toml");
        let profile = profile_workspace(tmp.path()).await;
        assert!(profile.project_kinds.contains(&ProjectKind::Rust));
        assert_eq!(
            profile.default_check_commands,
            vec!["cargo test".to_string()]
        );
    }

    #[test]
    fn node_verify_commands_prefer_bun_then_package_scripts() {
        let tmp = tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("package.json"), "{}").expect("write package.json");
        assert!(node_verify_commands(tmp.path()).is_empty());

        std::fs::write(
            tmp.path().join("package.json"),
            r#"{"scripts":{"test":"node --test"}}"#,
        )
        .expect("write package.json");
        assert_eq!(
            node_verify_commands(tmp.path()),
            vec!["npm test".to_string()]
        );

        std::fs::write(tmp.path().join("bun.lock"), "[]").expect("write bun.lock");
        assert_eq!(
            node_verify_commands(tmp.path()),
            vec!["bun test".to_string()]
        );
    }

    #[test]
    fn build_effective_task_names_exact_verify_commands() {
        let request = CodingTaskRequest {
            task: "add a helper".to_string(),
            workspace_root: None,
            provider: None,
            model: None,
            mode: CodingMode::Edit,
            permission_mode: PermissionModeConfig::AcceptAll,
            max_turns: 4,
            tools: None,
            verify: VerificationPolicy::Auto,
            context: CodingContextPolicy::default(),
        };
        let prompt = build_effective_task(&request, &["cargo test".to_string()]);
        assert!(prompt.contains("add a helper"));
        assert!(prompt.contains("Complete this coding task"));
        assert!(prompt.contains("Match the surrounding file's indentation"));
        assert!(prompt.contains("Apply large edits yourself"));
        assert!(
            prompt.contains(
                "Run exactly these verification commands after the last edit: cargo test."
            )
        );
    }

    #[tokio::test]
    async fn verification_reuses_last_successful_matching_command() {
        let tmp = tempdir().expect("tempdir");
        let profile = WorkspaceProfile {
            root: tmp.path().to_path_buf(),
            git_root: None,
            project_kinds: vec![ProjectKind::Unknown],
            default_check_commands: vec!["sh -c 'echo should-not-run; exit 1'".to_string()],
            dirty_files: Vec::new(),
        };
        let mut ledger = crate::harness::CommandLedger::default();
        ledger.record_command("sh -c 'echo should-not-run; exit 1'", true);
        let events = crate::harness::EventStream::new();
        let report = run_verification(
            &profile,
            &VerificationPolicy::Auto,
            &events,
            Some(&ledger),
            &[],
        )
        .await;
        assert!(report.passed);
        assert_eq!(report.commands.len(), 1);
        assert_eq!(report.commands[0].duration_ms, 0);
        assert_eq!(report.skipped_reason.as_deref(), Some(REUSED_VERIFICATION));
    }

    #[tokio::test]
    async fn verification_runs_again_after_a_later_mutation() {
        let tmp = tempdir().expect("tempdir");
        let profile = WorkspaceProfile {
            root: tmp.path().to_path_buf(),
            git_root: None,
            project_kinds: vec![ProjectKind::Unknown],
            default_check_commands: vec!["sh -c 'echo ran-again; exit 0'".to_string()],
            dirty_files: Vec::new(),
        };
        let mut ledger = crate::harness::CommandLedger::default();
        ledger.record_command("sh -c 'echo ran-again; exit 0'", true);
        ledger.record_mutation();
        let events = crate::harness::EventStream::new();
        let report = run_verification(
            &profile,
            &VerificationPolicy::Auto,
            &events,
            Some(&ledger),
            &[],
        )
        .await;
        assert!(report.passed);
        assert!(report.commands[0].duration_ms > 0 || report.commands[0].exit_code == Some(0));
        assert_ne!(report.commands[0].stdout_tail.trim(), REUSED_VERIFICATION);
        assert!(report.skipped_reason.is_none());
    }

    #[tokio::test]
    async fn verification_fails_when_library_code_uses_unwrap() {
        let tmp = tempdir().expect("tempdir");
        let src = tmp.path().join("src");
        std::fs::create_dir(&src).expect("mkdir");
        std::fs::write(src.join("lib.rs"), "pub fn f() -> u8 { None.unwrap() }\n").expect("write");
        let profile = WorkspaceProfile {
            root: tmp.path().to_path_buf(),
            git_root: None,
            project_kinds: vec![ProjectKind::Unknown],
            default_check_commands: vec!["true".to_string()],
            dirty_files: Vec::new(),
        };
        let events = crate::harness::EventStream::new();
        let report = run_verification(
            &profile,
            &VerificationPolicy::Auto,
            &events,
            None,
            &[PathBuf::from("src/lib.rs")],
        )
        .await;
        assert!(!report.passed);
        assert!(
            report
                .commands
                .iter()
                .any(|run| run.command == "panic-path scan" && run.exit_code == Some(1))
        );
        assert!(
            report
                .commands
                .iter()
                .any(|run| run.stderr_tail.contains("unwrap"))
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn verification_command_does_not_receive_host_credentials() {
        use crate::testing::planted_credential::{PRINT_PLANTED, rerun_with_planted_credential};
        if rerun_with_planted_credential(
            "harness::coding::tests::verification_command_does_not_receive_host_credentials",
        ) {
            return;
        }
        let dir = tempdir().expect("temp dir");
        let run = run_shell_command(dir.path(), PRINT_PLANTED).await;
        assert_eq!(
            (run.exit_code, run.stdout_tail.as_str()),
            (Some(0), "unset")
        );
    }
}
