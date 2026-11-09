//! One-time-per-domain probe for `/llms.txt` and `/llm.txt` capability files.
//!
//! These files, when present, describe how LLM agents should interact with a domain.
//! The probe fires once per hostname, caches the result, and never blocks or fails the
//! primary tool action.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

const PROBE_TIMEOUT_SECS: u64 = 4;
const PROBE_MAX_BYTES: usize = 65_536; // 64 KiB
const PROBE_PATHS: &[&str] = &["/llms.txt", "/llm.txt"];

/// Outcome of a single domain probe.
#[derive(Clone, Debug)]
pub enum ProbeResult {
    /// A file was found; `url` is the URL it was fetched from, `content` the (possibly truncated) body.
    Found { url: String, content: String },
    /// Neither `/llms.txt` nor `/llm.txt` returned a 200.
    NotFound,
    /// Probe was skipped (no domain could be parsed from the URL) or network error occurred.
    Unavailable,
}

impl ProbeResult {
    /// Human-readable summary used as a tool warning.
    #[must_use]
    pub fn as_warning(&self) -> Option<String> {
        match self {
            Self::Found { url, content } => {
                let preview: String = content.chars().take(200).collect();
                Some(format!(
                    "llms.txt found at {} — {} char(s): {}{}",
                    url,
                    content.len(),
                    preview,
                    if content.len() > 200 { "…" } else { "" }
                ))
            }
            Self::NotFound | Self::Unavailable => None,
        }
    }
}

/// Thread-safe per-domain probe cache. Wrap in `Arc` and share across tool instances.
pub type DomainProbeCache = Arc<RwLock<HashMap<String, ProbeResult>>>;

/// Create a new, empty probe cache.
#[must_use]
pub fn new_probe_cache() -> DomainProbeCache {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Extract the normalized hostname (lowercased, no port) from a URL string.
///
/// Returns `None` if the string cannot be parsed as an HTTP(S) URL.
#[must_use]
pub fn extract_domain(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .filter(|u| u.scheme() == "http" || u.scheme() == "https")
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
}

/// Ensure the domain for `url` has been probed.
///
/// Returns the cached `ProbeResult` (either from a prior probe or a freshly executed one).
/// If the domain was already probed, returns immediately without any network activity.
pub async fn ensure_domain_probed(cache: &DomainProbeCache, url: &str) -> ProbeResult {
    let Some(domain) = extract_domain(url) else {
        return ProbeResult::Unavailable;
    };

    // Fast path: already probed.
    {
        let guard = cache.read().await;
        if let Some(cached) = guard.get(&domain) {
            return cached.clone();
        }
    }

    // Slow path: probe and store.
    let result = probe_domain(&domain).await;
    {
        let mut guard = cache.write().await;
        // Double-check to avoid racing two probes for the same domain.
        guard.entry(domain).or_insert_with(|| result.clone());
    }
    result
}

/// Perform the actual HTTP probes for a known domain.
async fn probe_domain(domain: &str) -> ProbeResult {
    let Ok(policy) = crate::gateway::egress::current_policy() else {
        return ProbeResult::Unavailable;
    };
    let policy = policy.http_policy(crate::gateway::egress::EgressClass::Tool);
    let client_timeout = Duration::from_secs(PROBE_TIMEOUT_SECS);

    for path in PROBE_PATHS {
        let probe_url = format!("https://{domain}{path}");
        let Ok(validated) = policy.validate_resolved_url(&probe_url) else {
            return ProbeResult::Unavailable;
        };
        let Ok(builder) = validated.pinned_client_builder() else {
            return ProbeResult::Unavailable;
        };
        let Ok(client) = builder
            .timeout(client_timeout)
            .user_agent("Mozilla/5.0 (compatible; Caliche/1.0; +llms-probe)")
            .redirect(reqwest::redirect::Policy::none())
            .build()
        else {
            return ProbeResult::Unavailable;
        };
        if let Ok(resp) = client.get(validated.url.as_str()).send().await
            && resp.status().is_success()
            && let Ok(text) = resp.text().await
        {
            let content = if text.len() > PROBE_MAX_BYTES {
                caliche_contracts::text::truncate_utf8(&text, PROBE_MAX_BYTES).to_string()
            } else {
                text
            };
            return ProbeResult::Found {
                url: probe_url,
                content,
            };
        }
    }

    ProbeResult::NotFound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn probe_rejects_loopback_and_metadata() {
        assert!(matches!(
            probe_domain("127.0.0.1").await,
            ProbeResult::Unavailable
        ));
        assert!(matches!(
            probe_domain("localhost").await,
            ProbeResult::Unavailable
        ));
        assert!(matches!(
            probe_domain("169.254.169.254").await,
            ProbeResult::Unavailable
        ));
    }
}
