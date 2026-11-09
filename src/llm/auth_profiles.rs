//! Auth profile management for credential rotation and cooldown.

use crate::traits::retry::RetryError;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// A credential source for a single provider.
#[derive(Debug, Clone)]
pub enum AuthProfile {
    /// Static API key
    ApiKey { name: String, key: String },
    /// Environment variable reference
    EnvVar { name: String, var_name: String },
    /// OAuth token (e.g., Anthropic Pro/Max subscription)
    OAuth {
        name: String,
        token: String,
        refresh_url: Option<String>,
    },
}

impl AuthProfile {
    /// Resolve the credential string (e.g. for API key). For `EnvVar`, reads from environment.
    /// Empty/whitespace credentials are treated as unresolved so rotation can skip them.
    #[must_use]
    pub fn resolve_key(&self) -> Option<String> {
        let raw = match self {
            Self::ApiKey { key, .. } => Some(key.clone()),
            Self::EnvVar { var_name, .. } => caliche_contracts::credential_env::var(var_name).ok(),
            Self::OAuth { token, .. } => Some(token.clone()),
        }?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        // Preserve original credential bytes aside from rejecting all-whitespace.
        Some(raw)
    }
}

/// Tracks health and cooldown state for a single auth profile.
#[derive(Debug)]
struct ProfileState {
    profile: AuthProfile,
    cooldown_until: Option<Instant>,
    consecutive_failures: u32,
    last_success: Option<Instant>,
    total_requests: AtomicU64,
    total_failures: AtomicU64,
}

/// Strategy for selecting which profile to use next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationStrategy {
    /// Use profiles in order, skip those in cooldown
    Priority,
    /// Round-robin across healthy profiles
    RoundRobin,
    /// Use the profile with the most recent success
    LeastRecentlyFailed,
}

/// Manages multiple auth profiles for a single provider with rotation and cooldown.
#[derive(Debug)]
pub struct AuthProfileManager {
    profiles: Vec<Arc<RwLock<ProfileState>>>,
    cooldown_base_seconds: u64,
    cooldown_max_seconds: u64,
    rotation_strategy: RotationStrategy,
    next_index: std::sync::atomic::AtomicUsize,
}

impl AuthProfileManager {
    /// Create a manager with the given profiles and defaults (cooldown 60s base, 3600s max).
    #[must_use]
    pub fn new(profiles: Vec<AuthProfile>, strategy: RotationStrategy) -> Self {
        let profiles = profiles
            .into_iter()
            .map(|profile| {
                Arc::new(RwLock::new(ProfileState {
                    profile,
                    cooldown_until: None,
                    consecutive_failures: 0,
                    last_success: None,
                    total_requests: AtomicU64::new(0),
                    total_failures: AtomicU64::new(0),
                }))
            })
            .collect();
        Self {
            profiles,
            cooldown_base_seconds: 60,
            cooldown_max_seconds: 3600,
            rotation_strategy: strategy,
            next_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Set cooldown base duration in seconds (default 60).
    #[must_use]
    pub const fn cooldown_base(mut self, seconds: u64) -> Self {
        self.cooldown_base_seconds = seconds;
        self
    }

    /// Set cooldown max duration in seconds (default 3600).
    #[must_use]
    pub const fn cooldown_max(mut self, seconds: u64) -> Self {
        self.cooldown_max_seconds = seconds;
        self
    }

    /// Get the next available credential, skipping profiles in cooldown.
    /// Returns (`profile_index`, `api_key_string`) or None if all are in cooldown.
    pub async fn next_credential(&self) -> Option<(usize, String)> {
        let now = Instant::now();
        let n = self.profiles.len();
        if n == 0 {
            return None;
        }

        match self.rotation_strategy {
            RotationStrategy::Priority => {
                for (i, state) in self.profiles.iter().enumerate() {
                    let guard = state.read().await;
                    if guard.cooldown_until.is_none_or(|t| now >= t)
                        && let Some(key) = guard.profile.resolve_key()
                    {
                        return Some((i, key));
                    }
                }
            }
            RotationStrategy::RoundRobin => {
                let start = self.next_index.fetch_add(1, Ordering::Relaxed) % n;
                for k in 0..n {
                    let i = (start + k) % n;
                    let state = &self.profiles[i];
                    let key = {
                        let guard = state.read().await;
                        let key = guard
                            .cooldown_until
                            .is_none_or(|t| now >= t)
                            .then(|| guard.profile.resolve_key())
                            .flatten();
                        drop(guard);
                        key
                    };
                    if let Some(key) = key {
                        return Some((i, key));
                    }
                }
            }
            RotationStrategy::LeastRecentlyFailed => {
                let mut best: Option<(usize, String, Option<Instant>)> = None;
                for (i, state) in self.profiles.iter().enumerate() {
                    let candidate = {
                        let guard = state.read().await;
                        let candidate = guard
                            .cooldown_until
                            .is_none_or(|t| now >= t)
                            .then(|| {
                                guard
                                    .profile
                                    .resolve_key()
                                    .map(|key| (key, guard.last_success))
                            })
                            .flatten();
                        drop(guard);
                        candidate
                    };
                    if let Some((key, cur)) = candidate {
                        let last = best.as_ref().and_then(|(_, _, t)| *t);
                        let better = match (last, cur) {
                            (Some(_), None) => false,
                            (Some(a), Some(b)) => b >= a,
                            (None, Some(_) | None) => true,
                        };
                        if better {
                            best = Some((i, key, cur));
                        }
                    }
                }
                if let Some((i, key, _)) = best {
                    return Some((i, key));
                }
            }
        }
        None
    }

    /// Mark a profile as failed (enters cooldown with exponential backoff).
    pub async fn mark_failure(&self, profile_index: usize, error: &RetryError) {
        if profile_index >= self.profiles.len() {
            return;
        }
        let state = &self.profiles[profile_index];
        let mut guard = state.write().await;
        guard.consecutive_failures = guard.consecutive_failures.saturating_add(1);
        guard.total_failures.fetch_add(1, Ordering::Relaxed);

        let cooldown_secs = match error {
            RetryError::RateLimited { retry_after, .. } => {
                retry_after.map(|d| d.as_secs()).unwrap_or_else(|| {
                    self.cooldown_base_seconds
                        .saturating_mul(1 << guard.consecutive_failures.min(6))
                })
            }
            RetryError::AuthError { .. } | RetryError::ModelUnavailable { .. } => self
                .cooldown_base_seconds
                .saturating_mul(1 << guard.consecutive_failures.min(6)),
            _ => self.cooldown_base_seconds,
        };
        let cooldown_secs = cooldown_secs.min(self.cooldown_max_seconds);
        guard.cooldown_until = Some(Instant::now() + Duration::from_secs(cooldown_secs));
    }

    /// Mark a profile as successful (resets cooldown).
    pub async fn mark_success(&self, profile_index: usize) {
        if profile_index >= self.profiles.len() {
            return;
        }
        let state = &self.profiles[profile_index];
        let mut guard = state.write().await;
        guard.cooldown_until = None;
        guard.consecutive_failures = 0;
        guard.last_success = Some(Instant::now());
        guard.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the soonest cooldown expiry across all profiles (for sleep/retry scheduling).
    pub async fn soonest_cooldown_expiry(&self) -> Option<Instant> {
        let now = Instant::now();
        let mut soonest: Option<Instant> = None;
        for state in &self.profiles {
            let guard = state.read().await;
            if let Some(t) = guard.cooldown_until
                && t > now
            {
                soonest = soonest.map(|s| s.min(t)).or(Some(t));
            }
        }
        soonest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_key_rejects_empty_and_whitespace() {
        assert!(
            AuthProfile::ApiKey {
                name: "a".into(),
                key: "   ".into(),
            }
            .resolve_key()
            .is_none()
        );
        assert!(
            AuthProfile::OAuth {
                name: "o".into(),
                token: String::new(),
                refresh_url: None,
            }
            .resolve_key()
            .is_none()
        );
        assert_eq!(
            AuthProfile::ApiKey {
                name: "a".into(),
                key: "sk-live".into(),
            }
            .resolve_key()
            .as_deref(),
            Some("sk-live")
        );
    }

    #[tokio::test]
    async fn empty_manager_returns_none() {
        let manager = AuthProfileManager::new(vec![], RotationStrategy::Priority);
        assert!(manager.next_credential().await.is_none());
    }

    #[tokio::test]
    async fn priority_skips_empty_key_and_cooldown() {
        let manager = AuthProfileManager::new(
            vec![
                AuthProfile::ApiKey {
                    name: "empty".into(),
                    key: " ".into(),
                },
                AuthProfile::ApiKey {
                    name: "primary".into(),
                    key: "key-1".into(),
                },
                AuthProfile::ApiKey {
                    name: "secondary".into(),
                    key: "key-2".into(),
                },
            ],
            RotationStrategy::Priority,
        )
        .cooldown_base(1)
        .cooldown_max(2);

        let (idx, key) = manager.next_credential().await.expect("credential");
        assert_eq!(idx, 1);
        assert_eq!(key, "key-1");

        manager
            .mark_failure(
                idx,
                &RetryError::AuthError {
                    message: "bad key".into(),
                },
            )
            .await;
        let (idx2, key2) = manager.next_credential().await.expect("fallback");
        assert_eq!(idx2, 2);
        assert_eq!(key2, "key-2");
    }

    #[tokio::test]
    async fn round_robin_rotates_healthy_profiles() {
        let manager = AuthProfileManager::new(
            vec![
                AuthProfile::ApiKey {
                    name: "a".into(),
                    key: "ka".into(),
                },
                AuthProfile::ApiKey {
                    name: "b".into(),
                    key: "kb".into(),
                },
            ],
            RotationStrategy::RoundRobin,
        );
        let first = manager.next_credential().await.expect("first");
        let second = manager.next_credential().await.expect("second");
        let third = manager.next_credential().await.expect("third");
        assert_ne!(first.0, second.0);
        assert_eq!(first.0, third.0);
    }

    #[tokio::test]
    async fn all_profiles_in_cooldown_returns_none() {
        let manager = AuthProfileManager::new(
            vec![AuthProfile::ApiKey {
                name: "only".into(),
                key: "k".into(),
            }],
            RotationStrategy::Priority,
        )
        .cooldown_base(60)
        .cooldown_max(60);
        let (idx, _) = manager.next_credential().await.expect("key");
        manager
            .mark_failure(
                idx,
                &RetryError::RateLimited {
                    message: "slow down".into(),
                    retry_after: Some(Duration::from_secs(30)),
                },
            )
            .await;
        assert!(manager.next_credential().await.is_none());
        assert!(manager.soonest_cooldown_expiry().await.is_some());
    }

    #[tokio::test]
    async fn mark_success_clears_cooldown() {
        let manager = AuthProfileManager::new(
            vec![AuthProfile::ApiKey {
                name: "only".into(),
                key: "k".into(),
            }],
            RotationStrategy::Priority,
        )
        .cooldown_base(60)
        .cooldown_max(60);
        manager
            .mark_failure(
                0,
                &RetryError::AuthError {
                    message: "nope".into(),
                },
            )
            .await;
        assert!(manager.next_credential().await.is_none());
        manager.mark_success(0).await;
        assert_eq!(
            manager.next_credential().await.map(|(_, k)| k).as_deref(),
            Some("k")
        );
    }

    #[tokio::test]
    async fn out_of_range_mark_is_noop() {
        let manager = AuthProfileManager::new(
            vec![AuthProfile::ApiKey {
                name: "only".into(),
                key: "k".into(),
            }],
            RotationStrategy::Priority,
        );
        manager
            .mark_failure(
                99,
                &RetryError::AuthError {
                    message: "x".into(),
                },
            )
            .await;
        manager.mark_success(99).await;
        assert!(manager.next_credential().await.is_some());
    }
}
