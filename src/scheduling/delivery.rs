//! Deliver job output to configured targets.

use crate::scheduling::job::{DeliveryTarget, RetryPolicy};
use std::path::Path;
use tokio::time::{Duration, sleep};

/// Result of delivering to one target.
#[derive(Debug, Clone)]
pub struct DeliveryResult {
    pub target: DeliveryTargetSummary,
    pub success: bool,
    pub error: Option<String>,
    pub attempts: u32,
}

/// Summary of a delivery target (for results; DeliveryTarget may hold paths/urls).
#[derive(Debug, Clone)]
#[allow(clippy::module_name_repetitions)]
pub enum DeliveryTargetSummary {
    File {
        path: std::path::PathBuf,
    },
    Webhook {
        url: String,
    },
    Channel {
        adapter_id: String,
        channel_id: String,
        user_id: Option<String>,
        thread_id: Option<String>,
    },
    EventStream,
    Callback,
}

impl From<&DeliveryTarget> for DeliveryTargetSummary {
    fn from(t: &DeliveryTarget) -> Self {
        match t {
            DeliveryTarget::File { path } => DeliveryTargetSummary::File { path: path.clone() },
            DeliveryTarget::Webhook { url, .. } => {
                DeliveryTargetSummary::Webhook { url: url.clone() }
            }
            DeliveryTarget::Channel {
                adapter_id,
                channel_id,
                user_id,
                thread_id,
                ..
            } => DeliveryTargetSummary::Channel {
                adapter_id: adapter_id.clone(),
                channel_id: channel_id.clone(),
                user_id: user_id.clone(),
                thread_id: thread_id.clone(),
            },
            DeliveryTarget::EventStream => DeliveryTargetSummary::EventStream,
            DeliveryTarget::Callback => DeliveryTargetSummary::Callback,
        }
    }
}

/// Deliver job output to all configured targets.
pub async fn deliver(output: &str, targets: &[DeliveryTarget]) -> Vec<DeliveryResult> {
    let mut results = Vec::with_capacity(targets.len());
    for target in targets {
        let summary = DeliveryTargetSummary::from(target);
        let (success, error, attempts) = match target {
            DeliveryTarget::File { path } => {
                let (ok, err) = deliver_file(output, path).await;
                (ok, err, 1)
            }
            DeliveryTarget::Webhook {
                url,
                headers,
                retry_policy,
            } => {
                deliver_with_retry(retry_policy.as_ref(), || async move {
                    deliver_webhook(output, url, headers).await
                })
                .await
            }
            DeliveryTarget::Channel {
                adapter_id,
                channel_id,
                user_id,
                thread_id,
                retry_policy,
            } => {
                deliver_with_retry(retry_policy.as_ref(), || async move {
                    deliver_channel(
                        output,
                        adapter_id,
                        channel_id,
                        user_id.as_deref(),
                        thread_id.as_deref(),
                    )
                    .await
                })
                .await
            }
            DeliveryTarget::EventStream | DeliveryTarget::Callback => (true, None, 1),
        };
        results.push(DeliveryResult {
            target: summary,
            success,
            error,
            attempts,
        });
    }
    results
}

async fn deliver_with_retry<F, Fut>(
    policy: Option<&RetryPolicy>,
    mut action: F,
) -> (bool, Option<String>, u32)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = (bool, Option<String>)>,
{
    let default_policy = RetryPolicy::default();
    let policy = policy.unwrap_or(&default_policy);

    let max_attempts = policy.max_attempts.max(1);
    let mut attempt = 0_u32;
    let mut backoff_ms = policy.initial_backoff_ms.max(1);
    let mut last_error = None;

    while attempt < max_attempts {
        attempt += 1;
        let (ok, err) = action().await;
        if ok {
            return (true, None, attempt);
        }
        last_error = err;
        if attempt >= max_attempts {
            break;
        }

        let jittered = if policy.jitter {
            let mut rng = fastrand::Rng::new();
            let jitter_cap = (backoff_ms / 5).max(1);
            let jitter = rng.u64(0..jitter_cap.saturating_add(1));
            backoff_ms.saturating_add(jitter)
        } else {
            backoff_ms
        };
        sleep(Duration::from_millis(jittered)).await;
        backoff_ms = (backoff_ms.saturating_mul(2)).min(policy.max_backoff_ms.max(1));
    }

    (false, last_error, attempt)
}

async fn deliver_file(output: &str, path: &Path) -> (bool, Option<String>) {
    let path = match crate::agent::tools::common::path::resolve_write_path_for_tool(
        &path.to_string_lossy(),
    ) {
        Ok(path) => path,
        Err(e) => return (false, Some(e.to_string())),
    };
    // The write follows no link below the workspace and creates missing
    // directories, so a link swapped in after the check cannot redirect it.
    let output = output.to_string();
    match crate::agent::tools::common::spawn_tool_blocking(move || {
        crate::agent::tools::common::path::write_tool_file(&path, output.as_bytes())
    })
    .await
    {
        Ok(Ok(())) => (true, None),
        Ok(Err(e)) => (false, Some(e.to_string())),
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn deliver_webhook(
    output: &str,
    url: &str,
    headers: &std::collections::HashMap<String, String>,
) -> (bool, Option<String>) {
    for k in headers.keys() {
        if crate::security::http_policy::is_forbidden_egress_header(k) {
            return (
                false,
                Some(format!("webhook delivery header not allowed: {k}")),
            );
        }
    }
    let validated = match crate::security::http_policy::HttpPolicy::from_env()
        .and_then(|policy| policy.validate_resolved_url(url))
    {
        Ok(validated) => validated,
        Err(e) => return (false, Some(e.to_string())),
    };
    let client = match validated.pinned_client() {
        Ok(client) => client,
        Err(e) => return (false, Some(e.to_string())),
    };
    let mut req = client
        .post(validated.url.as_str())
        .body(output.to_string())
        .header("Content-Type", "text/plain; charset=utf-8");
    for (k, v) in headers {
        req = req.header(k.as_str(), v.as_str());
    }
    match req.send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                (true, None)
            } else {
                (false, Some(format!("HTTP {}", resp.status())))
            }
        }
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn deliver_channel(
    output: &str,
    adapter_id: &str,
    channel_id: &str,
    user_id: Option<&str>,
    thread_id: Option<&str>,
) -> (bool, Option<String>) {
    // Fallback transport until dedicated gateway adapter delivery wiring is configured.
    // If channel_id is a URL, treat it like a generic webhook endpoint.
    if channel_id.starts_with("http://") || channel_id.starts_with("https://") {
        let mut headers = std::collections::HashMap::new();
        headers.insert("X-Caliche-Adapter".to_string(), adapter_id.to_string());
        if let Some(user_id) = user_id {
            headers.insert("X-Caliche-User".to_string(), user_id.to_string());
        }
        if let Some(thread_id) = thread_id {
            headers.insert("X-Caliche-Thread".to_string(), thread_id.to_string());
        }
        return deliver_webhook(output, channel_id, &headers).await;
    }

    (
        false,
        Some(format!(
            "channel delivery adapter not configured (adapter={}, channel={})",
            adapter_id, channel_id
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::job::DeliveryTarget;
    use std::collections::HashMap;

    /// A file delivery into a directory that is a link to an outside
    /// directory writes nothing there.
    #[cfg(unix)]
    #[tokio::test]
    async fn file_delivery_writes_nothing_through_a_linked_directory() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let out = tempfile::tempdir().expect("outside");
        let root = tmp.path().canonicalize().expect("canonical");
        let outside = out.path().canonicalize().expect("outside");
        std::os::unix::fs::symlink(&outside, root.join("reports")).expect("link");
        let target = root.join("reports/today.txt");
        crate::agent::tools::common::set_tool_workspace_root(Some(root.clone()));

        let results = deliver("payload", &[DeliveryTarget::File { path: target }]).await;

        crate::agent::tools::common::set_tool_workspace_root(None);
        let written: Vec<_> = std::fs::read_dir(&outside).expect("outside").collect();
        assert_eq!(
            (results.len(), results[0].success, written.len()),
            (1, false, 0)
        );
    }

    #[tokio::test]
    async fn file_delivery_refuses_git_metadata() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().canonicalize().expect("canonical");
        std::fs::create_dir_all(root.join(".git/hooks")).expect("hooks");
        let hook = root.join(".git/hooks/post-commit");
        crate::agent::tools::common::set_tool_workspace_root(Some(root.clone()));

        let results = deliver("payload", &[DeliveryTarget::File { path: hook.clone() }]).await;

        crate::agent::tools::common::set_tool_workspace_root(None);
        assert_eq!(results.len(), 1);
        assert!(!results[0].success, "a hook was delivered");
        assert_eq!(
            results[0].error.as_deref(),
            Some(
                format!(
                    "refusing to write '{}': it is inside a git repository's metadata",
                    hook.display()
                )
                .as_str()
            )
        );
        assert!(!hook.exists(), "hook file written");
    }

    #[tokio::test]
    async fn webhook_delivery_rejects_loopback() {
        let results = deliver(
            "payload",
            &[DeliveryTarget::Webhook {
                url: "http://127.0.0.1/ssrf".to_string(),
                headers: HashMap::new(),
                retry_policy: None,
            }],
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success, "loopback webhook must fail closed");
        let err = results[0].error.as_deref().unwrap_or_default();
        assert!(
            err.contains("blocked") || err.contains("allowlist"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn channel_http_url_rejects_metadata() {
        let results = deliver(
            "payload",
            &[DeliveryTarget::Channel {
                adapter_id: "webhook".to_string(),
                channel_id: "http://169.254.169.254/latest/meta-data/".to_string(),
                user_id: None,
                thread_id: None,
                retry_policy: None,
            }],
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success, "metadata channel URL must fail closed");
        let err = results[0].error.as_deref().unwrap_or_default();
        assert!(
            err.contains("blocked") || err.contains("allowlist"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn file_delivery_rejects_parent_dir() {
        let results = deliver(
            "payload",
            &[DeliveryTarget::File {
                path: std::path::PathBuf::from("../etc/passwd"),
            }],
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success, "path traversal must fail closed");
        let err = results[0].error.as_deref().unwrap_or_default();
        assert!(
            err.to_lowercase().contains("traversal") || err.contains("NUL"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn webhook_delivery_rejects_host_header() {
        let mut headers = HashMap::new();
        headers.insert("Host".to_string(), "169.254.169.254".to_string());
        let results = deliver(
            "payload",
            &[DeliveryTarget::Webhook {
                url: "http://127.0.0.1/hook".to_string(),
                headers,
                retry_policy: None,
            }],
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success, "Host header must fail closed");
        let err = results[0].error.as_deref().unwrap_or_default();
        assert!(
            err.to_lowercase().contains("header"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn file_delivery_rejects_absolute_outside_workspace() {
        let results = deliver(
            "payload",
            &[DeliveryTarget::File {
                path: std::path::PathBuf::from("/etc/passwd"),
            }],
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(
            !results[0].success,
            "absolute path outside workspace must fail closed"
        );
        let err = results[0].error.as_deref().unwrap_or_default();
        assert!(
            err.to_lowercase().contains("outside workspace")
                || err.to_lowercase().contains("traversal"),
            "unexpected error: {err}"
        );
    }
}
