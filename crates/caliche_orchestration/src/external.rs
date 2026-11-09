//! Subprocess adapter for external agent harnesses (Pi, Hermes, custom CLIs).
//!
//! The prompt is always a separate argv element or stdin payload. It is never
//! interpolated into a shell string. The executable is validated by
//! [`crate::CommandPolicy`] before spawn.

use crate::command::{CommandPolicy, is_forbidden_shell};
use crate::harness::{HarnessRequest, HarnessResponse, HarnessRuntime, HarnessUsage};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

/// Default bound when a request does not set a timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
/// Prompts larger than this are refused (argv / pipe limits).
const DEFAULT_MAX_PROMPT_BYTES: usize = 256 * 1024;
/// Combined stdout cap; excess is a memory-bomb refusal.
const DEFAULT_MAX_STDOUT_BYTES: usize = 1024 * 1024;
/// Stderr is capped separately so a noisy CLI cannot fill memory.
const DEFAULT_MAX_STDERR_BYTES: usize = 256 * 1024;

/// Where the composed prompt is delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptPlacement {
    /// Prompt is the last argv element (value of `-p` / `-q`, or a positional).
    ///
    /// Set `end_of_options` when the prompt is positional so a payload that
    /// starts with `-` cannot be parsed as a flag. Leave it false when the
    /// prompt is the value of a flag such as `-p` or `-q`.
    LastArg {
        /// Insert `--` immediately before the prompt.
        end_of_options: bool,
    },
    /// Prompt is written to stdin; stdin is then closed.
    Stdin,
}

impl Default for PromptPlacement {
    fn default() -> Self {
        Self::LastArg {
            end_of_options: false,
        }
    }
}

/// Operator-supplied CLI invocation. The prompt is never joined into `command`.
#[derive(Debug, Clone)]
pub struct ExternalCliSpec {
    /// Executable. Must be a static path or PATH name, never a shell.
    pub command: String,
    /// Arguments placed before the prompt (for `LastArg`) or as the full argv
    /// (for `Stdin`).
    pub args_prefix: Vec<String>,
    /// How the prompt is delivered.
    pub prompt_placement: PromptPlacement,
    /// Optional working directory. Inherited from the operator process when unset.
    pub cwd: Option<PathBuf>,
    /// Label returned by [`HarnessRuntime::kind`].
    pub kind: &'static str,
    /// Prompt size cap in bytes.
    pub max_prompt_bytes: usize,
    /// Stdout size cap in bytes.
    pub max_stdout_bytes: usize,
    /// Variables set on the child on top of the scrubbed host environment.
    /// The harness never inherits this process's provider keys or tokens,
    /// so a key it needs goes here.
    pub env: Vec<(String, String)>,
}

impl ExternalCliSpec {
    /// Pi print mode: `pi -p <prompt>`.
    #[must_use]
    pub fn pi() -> Self {
        Self {
            command: "pi".to_string(),
            args_prefix: vec!["-p".to_string()],
            prompt_placement: PromptPlacement::LastArg {
                end_of_options: false,
            },
            cwd: None,
            kind: "pi",
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            max_stdout_bytes: DEFAULT_MAX_STDOUT_BYTES,
            env: Vec::new(),
        }
    }

    /// Hermes one-shot chat: `hermes chat -Q -q <prompt>`.
    #[must_use]
    pub fn hermes() -> Self {
        Self {
            command: "hermes".to_string(),
            args_prefix: vec!["chat".to_string(), "-Q".to_string(), "-q".to_string()],
            prompt_placement: PromptPlacement::LastArg {
                end_of_options: false,
            },
            cwd: None,
            kind: "hermes",
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            max_stdout_bytes: DEFAULT_MAX_STDOUT_BYTES,
            env: Vec::new(),
        }
    }

    /// Custom executable and argv prefix.
    #[must_use]
    pub fn custom(command: impl Into<String>, args_prefix: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args_prefix,
            prompt_placement: PromptPlacement::default(),
            cwd: None,
            kind: "external_cli",
            max_prompt_bytes: DEFAULT_MAX_PROMPT_BYTES,
            max_stdout_bytes: DEFAULT_MAX_STDOUT_BYTES,
            env: Vec::new(),
        }
    }
}

/// Runs an external harness as a subprocess.
#[derive(Debug, Clone)]
pub struct ExternalCliRuntime {
    spec: ExternalCliSpec,
}

impl ExternalCliRuntime {
    /// Validate `spec.command` with `policy` and build a runtime.
    ///
    /// # Errors
    ///
    /// Returns an error when the command is a shell, fails static checks, or is
    /// not on the allowlist.
    pub fn new(spec: ExternalCliSpec, policy: &CommandPolicy) -> Result<Self> {
        policy.validate(&spec.command)?;
        for arg in &spec.args_prefix {
            if arg.contains('\0') {
                bail!("harness argument contains interior NUL");
            }
            if is_forbidden_shell(arg) {
                bail!(
                    "harness argument '{arg}' is a shell interpreter; pass the harness executable directly and keep the prompt as argv"
                );
            }
        }
        Ok(Self { spec })
    }

    /// Pi preset with the process environment allowlist.
    ///
    /// # Errors
    ///
    /// Returns an error when the command policy refuses `pi`.
    pub fn pi() -> Result<Self> {
        Self::new(ExternalCliSpec::pi(), &CommandPolicy::from_env()?)
    }

    /// Hermes preset with the process environment allowlist.
    ///
    /// # Errors
    ///
    /// Returns an error when the command policy refuses `hermes`.
    pub fn hermes() -> Result<Self> {
        Self::new(ExternalCliSpec::hermes(), &CommandPolicy::from_env()?)
    }

    /// Pi preset that skips the env allowlist (still rejects shells).
    ///
    /// # Errors
    ///
    /// Returns an error if the Pi command fails static/shell checks.
    pub fn pi_open() -> Result<Self> {
        Self::new(ExternalCliSpec::pi(), &CommandPolicy::open())
    }

    /// Hermes preset that skips the env allowlist (still rejects shells).
    ///
    /// # Errors
    ///
    /// Returns an error if the Hermes command fails static/shell checks.
    pub fn hermes_open() -> Result<Self> {
        Self::new(ExternalCliSpec::hermes(), &CommandPolicy::open())
    }
}

#[async_trait]
impl HarnessRuntime for ExternalCliRuntime {
    fn kind(&self) -> &'static str {
        self.spec.kind
    }

    async fn execute(&self, request: HarnessRequest) -> Result<HarnessResponse> {
        let prompt = compose_prompt(&request);
        if prompt.contains('\0') {
            bail!("harness prompt contains interior NUL");
        }
        if prompt.len() > self.spec.max_prompt_bytes {
            bail!(
                "harness prompt is {} bytes; max is {}",
                prompt.len(),
                self.spec.max_prompt_bytes
            );
        }

        let mut command = Command::new(&self.spec.command);
        command
            .args(&self.spec.args_prefix)
            .env_clear()
            .envs(caliche_contracts::child_env::host_child_env_with(
                self.spec.env.iter().map(|(key, value)| (key, value)),
            ))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &self.spec.cwd {
            command.current_dir(cwd);
        }

        match self.spec.prompt_placement {
            PromptPlacement::LastArg { end_of_options } => {
                command.stdin(Stdio::null());
                if end_of_options {
                    command.arg("--");
                }
                command.arg(&prompt);
            }
            PromptPlacement::Stdin => {
                command.stdin(Stdio::piped());
            }
        }

        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to spawn harness '{}' ({})",
                self.spec.command, self.spec.kind
            )
        })?;

        if matches!(self.spec.prompt_placement, PromptPlacement::Stdin)
            && let Some(mut stdin) = child.stdin.take()
        {
            stdin
                .write_all(prompt.as_bytes())
                .await
                .context("write harness stdin")?;
            stdin.shutdown().await.context("close harness stdin")?;
        }

        let mut stdout = child.stdout.take().context("harness stdout")?;
        let mut stderr = child.stderr.take().context("harness stderr")?;
        let timeout = request.timeout.unwrap_or(DEFAULT_TIMEOUT);

        let outcome = Box::pin(tokio::time::timeout(timeout, async {
            tokio::try_join!(
                async { child.wait().await.context("wait for harness process") },
                read_capped(&mut stdout, self.spec.max_stdout_bytes),
                read_capped(&mut stderr, DEFAULT_MAX_STDERR_BYTES),
            )
        }))
        .await;

        match outcome {
            Ok(Ok((status, stdout, stderr))) => {
                if !status.success() {
                    let err = String::from_utf8_lossy(&stderr);
                    bail!(
                        "harness '{}' exited with {}: {}",
                        self.spec.kind,
                        status
                            .code()
                            .map_or_else(|| "signal".to_string(), |code| code.to_string()),
                        err.trim()
                    );
                }
                let text = String::from_utf8_lossy(&stdout).trim_end().to_string();
                Ok(HarnessResponse {
                    text,
                    usage: HarnessUsage::default(),
                })
            }
            Ok(Err(error)) => Err(error),
            Err(_) => {
                let _ = child.kill().await;
                bail!(
                    "harness '{}' timed out after {}ms",
                    self.spec.kind,
                    timeout.as_millis()
                )
            }
        }
    }
}

fn compose_prompt(request: &HarnessRequest) -> String {
    let mut parts = Vec::new();
    if let Some(persona) = request.persona.as_ref().map(|value| value.trim())
        && !persona.is_empty()
    {
        parts.push(format!("Persona:\n{persona}"));
    }
    if let Some(tools) = &request.tools
        && !tools.is_empty()
    {
        parts.push(format!("Allowed tools:\n{}", tools.join(", ")));
    }
    if let Some(instructions) = request.instructions.as_ref().map(|value| value.trim())
        && !instructions.is_empty()
    {
        parts.push(format!("Instructions:\n{instructions}"));
    }
    if parts.is_empty() {
        return request.message.clone();
    }
    format!("{}\n\nTask:\n{}", parts.join("\n\n"), request.message)
}

async fn read_capped<R: AsyncReadExt + Unpin>(reader: &mut R, max_bytes: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        let n = reader
            .read(&mut chunk)
            .await
            .context("read harness output")?;
        if n == 0 {
            break;
        }
        let next = buf.len().saturating_add(n);
        if next > max_bytes {
            bail!("harness output exceeded {max_bytes} bytes");
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandPolicy;
    use std::collections::HashSet;

    #[test]
    fn shells_cannot_be_the_executable() {
        let spec = ExternalCliSpec::custom("sh", vec!["-c".to_string()]);
        ExternalCliRuntime::new(spec, &CommandPolicy::open()).expect_err("sh");
        let bash = ExternalCliSpec::custom("/bin/bash", vec!["-c".to_string()]);
        ExternalCliRuntime::new(bash, &CommandPolicy::open()).expect_err("bash path");
        let wrapped = ExternalCliSpec::custom("env", vec!["sh".to_string(), "-c".to_string()]);
        ExternalCliRuntime::new(wrapped, &CommandPolicy::open())
            .expect_err("env sh -c must not wrap a shell");
    }

    #[test]
    fn path_command_requires_exact_allowlist_entry() {
        let spec = ExternalCliSpec::custom("/tmp/pi", Vec::new());
        let policy = CommandPolicy::with_allowlist(HashSet::from(["pi".to_string()]));
        ExternalCliRuntime::new(spec, &policy).expect_err("basename must not authorize path");
    }

    #[test]
    fn presets_use_flag_values_not_end_of_options() {
        assert_eq!(
            ExternalCliSpec::pi().prompt_placement,
            PromptPlacement::LastArg {
                end_of_options: false
            }
        );
        assert_eq!(ExternalCliSpec::pi().args_prefix, vec!["-p".to_string()]);
        assert_eq!(
            ExternalCliSpec::hermes().args_prefix,
            vec!["chat".to_string(), "-Q".to_string(), "-q".to_string()]
        );
    }

    #[test]
    fn compose_keeps_metacharacters_in_the_message() {
        let request = HarnessRequest::new("'; rm -rf /");
        assert_eq!(compose_prompt(&request), "'; rm -rf /");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn echo_round_trips_metacharacters_as_argv() {
        let spec = ExternalCliSpec::custom("/bin/echo", Vec::new());
        let runtime = ExternalCliRuntime::new(spec, &CommandPolicy::open()).expect("echo");
        let response = runtime
            .execute(HarnessRequest::new("'; rm -rf / && true"))
            .await
            .expect("echo should run");
        assert_eq!(response.text, "'; rm -rf / && true");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn prompt_cap_is_enforced() {
        let mut spec = ExternalCliSpec::custom("/bin/echo", Vec::new());
        spec.max_prompt_bytes = 8;
        let runtime = ExternalCliRuntime::new(spec, &CommandPolicy::open()).expect("echo");
        runtime
            .execute(HarnessRequest::new("0123456789"))
            .await
            .expect_err("oversize prompt");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn harness_gets_its_configured_env_but_no_host_credentials() {
        // Tests may not mutate this process's environment: re-run this one
        // test in a child that holds the credential.
        const PLANTED: &str = "CALICHE_SCRUB_PROBE_TOKEN";
        if std::env::var_os(PLANTED).is_none() {
            let name = "external::tests::harness_gets_its_configured_env_but_no_host_credentials";
            let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", name, "--nocapture"])
                .env(PLANTED, "planted")
                .output()
                .expect("re-run test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let mut spec = ExternalCliSpec::custom("printenv", Vec::new());
        spec.prompt_placement = PromptPlacement::Stdin;
        spec.env = vec![
            ("CALICHE_HARNESS_KEY".to_string(), "configured".to_string()),
            (
                caliche_contracts::child_env::PATH_VAR.to_string(),
                ".:/usr/bin:/bin".to_string(),
            ),
        ];
        let runtime = ExternalCliRuntime::new(spec, &CommandPolicy::open()).expect("printenv");
        let response = runtime
            .execute(HarnessRequest::new("hi"))
            .await
            .expect("printenv should run");
        let names: Vec<&str> = response
            .text
            .lines()
            .filter_map(|line| line.split_once('=').map(|(name, _)| name))
            .collect();
        assert!(!names.contains(&PLANTED), "{names:?}");
        assert!(
            response
                .text
                .lines()
                .any(|line| line == "CALICHE_HARNESS_KEY=configured"),
            "{}",
            response.text
        );
        assert!(
            response
                .text
                .lines()
                .any(|line| line == "PATH=/usr/bin:/bin"),
            "{}",
            response.text
        );
    }
}
