//! JSONL transcript of ACP frames on the child stdio pipes.
//!
//! Each line is one object, `{"dir":"client","frame":...}` or
//! `{"dir":"peer","frame":...}`. A peer line that is not JSON is stored as
//! `{"dir":"peer","invalid":"..."}`. Values of environment variables whose
//! names look like credentials are replaced with `[redacted]` before the
//! line is written.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex as StdMutex, MutexGuard};

use serde_json::{Map, Value};

pub struct Transcript {
    file: StdMutex<std::fs::File>,
    redactor: Redactor,
}

impl Transcript {
    /// Create or truncate `path` and write frames there.
    ///
    /// # Errors
    ///
    /// Returns an IO error when the parent directory or the file cannot be created.
    pub fn create(path: &Path) -> Result<Self, std::io::Error> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            file: StdMutex::new(file),
            redactor: Redactor::from_env(),
        })
    }

    /// Append one client-to-peer frame.
    ///
    /// # Errors
    ///
    /// Returns an IO error when the transcript file cannot be written.
    pub fn record_client(&self, frame: &Value) -> Result<(), std::io::Error> {
        let row = serde_json::json!({
            "dir": "client",
            "frame": self.redactor.scrub_value(frame),
        });
        self.write_row(&row)
    }

    /// Append one peer stdout line.
    ///
    /// # Errors
    ///
    /// Returns an IO error when the transcript file cannot be written.
    pub fn record_peer_line(&self, line: &str) -> Result<(), std::io::Error> {
        let row = serde_json::from_str::<Value>(line).map_or_else(
            |_| {
                serde_json::json!({
                    "dir": "peer",
                    "invalid": self.redactor.scrub(line),
                })
            },
            |frame| {
                serde_json::json!({
                    "dir": "peer",
                    "frame": self.redactor.scrub_value(&frame),
                })
            },
        );
        self.write_row(&row)
    }

    fn write_row(&self, row: &Value) -> Result<(), std::io::Error> {
        let mut bytes = serde_json::to_vec(row).map_err(std::io::Error::other)?;
        bytes.push(b'\n');
        let mut file = lock_file(&self.file);
        file.write_all(&bytes)?;
        file.flush()
    }
}

/// Replaces credential-shaped environment values with `[redacted]`.
#[derive(Debug, Clone)]
pub struct Redactor {
    secrets: Vec<String>,
}

impl Redactor {
    /// Collect secret-shaped values from the process environment.
    ///
    /// A variable counts when
    /// [`caliche_contracts::child_env::is_credential_name`] matches its
    /// name and the value is at least 8 characters. Short values are left
    /// alone so ordinary words are not stripped from protocol text.
    #[must_use]
    pub fn from_env() -> Self {
        let mut secrets = Vec::new();
        for (key, value) in std::env::vars() {
            if value.len() >= 8 && caliche_contracts::child_env::is_credential_name(&key) {
                secrets.push(value);
            }
        }
        Self::from_secrets(secrets)
    }

    #[must_use]
    pub fn from_secrets(mut secrets: Vec<String>) -> Self {
        secrets.retain(|secret| secret.len() >= 8);
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        secrets.dedup();
        Self { secrets }
    }

    #[must_use]
    pub fn scrub(&self, text: &str) -> String {
        let mut scrubbed = text.to_owned();
        for secret in &self.secrets {
            if scrubbed.contains(secret.as_str()) {
                scrubbed = scrubbed.replace(secret.as_str(), "[redacted]");
            }
        }
        scrubbed
    }

    #[must_use]
    pub fn scrub_value(&self, value: &Value) -> Value {
        match value {
            Value::String(text) => Value::String(self.scrub(text)),
            Value::Array(items) => {
                Value::Array(items.iter().map(|item| self.scrub_value(item)).collect())
            }
            Value::Object(map) => Value::Object(scrub_map(self, map)),
            other => other.clone(),
        }
    }
}

fn scrub_map(redactor: &Redactor, map: &Map<String, Value>) -> Map<String, Value> {
    let mut scrubbed = Map::new();
    for (key, value) in map {
        scrubbed.insert(redactor.scrub(key), redactor.scrub_value(value));
    }
    scrubbed
}

fn lock_file(mutex: &StdMutex<std::fs::File>) -> MutexGuard<'_, std::fs::File> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::{Redactor, Transcript};
    use serde_json::{Value, json};

    /// The token must be in this process's environment, which tests may not
    /// mutate: the test re-runs itself in a child that has it.
    #[test]
    fn from_env_redacts_credential_values_and_keeps_others() {
        const NAME: &str = "CALICHE_REDACTOR_PROBE_TOKEN";
        const VALUE: &str = "probe-secret-value";
        if std::env::var_os(NAME).is_none() {
            let test = "transcript::tests::from_env_redacts_credential_values_and_keeps_others";
            let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", test, "--nocapture"])
                .env(NAME, VALUE)
                .env("CALICHE_REDACTOR_PROBE_SMTP_PASS", "probe-pass-value")
                .env("CALICHE_REDACTOR_PROBE_GITHUB_PAT", "probe-pat-value")
                .env("CALICHE_REDACTOR_PROBE_PLAIN", "plain-value-kept")
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
        let redactor = Redactor::from_env();
        assert_eq!(
            redactor.scrub(
                "a probe-secret-value b plain-value-kept c probe-pass-value d probe-pat-value"
            ),
            "a [redacted] b plain-value-kept c [redacted] d [redacted]"
        );
    }

    #[test]
    fn secret_values_are_removed() {
        let redactor = Redactor::from_secrets(vec!["sk-live-secret".to_owned()]);
        assert_eq!(
            redactor.scrub("token sk-live-secret end"),
            "token [redacted] end"
        );
        let scrubbed = redactor.scrub_value(&json!({"api": "sk-live-secret"}));
        assert_eq!(scrubbed["api"], "[redacted]");
    }

    #[test]
    fn short_values_are_kept() {
        let redactor = Redactor::from_secrets(vec!["short".to_owned()]);
        assert_eq!(redactor.scrub("short"), "short");
    }

    #[test]
    fn frames_are_one_json_object_per_line() {
        let path = std::env::temp_dir().join(format!(
            "caliche-acp-transcript-{}-frames.jsonl",
            std::process::id()
        ));
        let transcript = Transcript::create(&path).expect("create transcript");
        transcript
            .record_client(&json!({"method": "initialize"}))
            .expect("client frame");
        transcript
            .record_peer_line(r#"{"result":{"authMethods":[]}}"#)
            .expect("peer frame");
        transcript
            .record_peer_line("not-json")
            .expect("invalid line");
        drop(transcript);
        let text = std::fs::read_to_string(&path).expect("read transcript");
        let _ = std::fs::remove_file(&path);
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 3, "{text}");
        for line in &lines {
            let parsed: Value = serde_json::from_str(line).expect("json line");
            assert!(parsed.get("dir").is_some(), "{line}");
        }
        assert!(text.contains("not-json"), "{text}");
    }
}
