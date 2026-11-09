//! Loading a local model on demand.
//!
//! LM Studio answers `No models loaded` when its just-in-time loading is off
//! and the model is not in memory. Before a request, the provider asks LM
//! Studio for the model's state (`GET {root}/api/v0/models/{model}`). When
//! the model is not loaded, one `POST {root}/api/v1/models/load` asks LM
//! Studio to load it (LM Studio 0.4 and later), and the state is read again
//! until it is `loaded`. The whole wait has one deadline.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

/// Gap between state reads while a model loads.
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// How long one state read may take.
const STATE_TIMEOUT: Duration = Duration::from_secs(2);

/// Models with a load request in flight from this process, by server root
/// and model. A second request for one of them waits for that load instead
/// of sending another.
fn in_flight() -> &'static Mutex<HashSet<String>> {
    static IN_FLIGHT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    IN_FLIGHT.get_or_init(Mutex::default)
}

/// Removes its key from [`in_flight`] when the load ends or is cancelled.
struct InFlight(String);

impl InFlight {
    /// Claim `key`; `None` when a load for it is already in flight.
    fn claim(key: String) -> Option<Self> {
        let claimed = in_flight()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key.clone());
        // Built only when claimed: a guard dropped here would remove the
        // other load's key.
        claimed.then(|| Self(key))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        in_flight()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.0);
    }
}

/// A model's state as the server reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelState {
    Loaded,
    NotLoaded,
    /// Any other state, such as `loading`: a load is under way.
    Loading,
}

#[derive(serde::Deserialize)]
struct StateBody {
    state: String,
}

/// Loads a model on a local server that does not load it by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLoader {
    root: String,
}

impl ModelLoader {
    /// A loader for the LM Studio server at `base_url` (with or without its
    /// `/v1` suffix).
    #[must_use]
    pub fn lm_studio(base_url: &str) -> Self {
        Self {
            root: super::first_token::server_root(base_url).to_string(),
        }
    }

    /// The model's state; `None` when the server does not say (an older
    /// server, an unknown model, or no answer).
    async fn state(&self, client: &reqwest::Client, model: &str) -> Option<ModelState> {
        let body: StateBody = client
            .get(format!("{}/api/v0/models/{model}", self.root))
            .timeout(STATE_TIMEOUT)
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        Some(match body.state.as_str() {
            "loaded" => ModelState::Loaded,
            "not-loaded" => ModelState::NotLoaded,
            _ => ModelState::Loading,
        })
    }

    /// True when the server says `model` is not loaded yet, or is loading.
    pub async fn needs_load(&self, client: &reqwest::Client, model: &str) -> bool {
        matches!(
            self.state(client, model).await,
            Some(ModelState::NotLoaded | ModelState::Loading)
        )
    }

    /// Load `model` and wait until the server reports it loaded, for at
    /// most `limit`. Sends one load request, and none when a load is already
    /// under way. Dropping the future cancels the wait.
    ///
    /// # Errors
    ///
    /// The server refused the load (its reason is quoted), or the model was
    /// not loaded within `limit`. Both are a
    /// [`RetryError::ModelUnavailable`](crate::traits::retry::RetryError).
    pub async fn load(
        &self,
        client: &reqwest::Client,
        model: &str,
        limit: Duration,
    ) -> anyhow::Result<()> {
        let deadline = tokio::time::Instant::now() + limit;
        tokio::time::timeout_at(deadline, self.load_until_ready(client, model, deadline))
            .await
            .unwrap_or_else(|_| Err(unavailable(load_timeout_error(model, limit))))
    }

    async fn load_until_ready(
        &self,
        client: &reqwest::Client,
        model: &str,
        deadline: tokio::time::Instant,
    ) -> anyhow::Result<()> {
        let claim = InFlight::claim(format!("{} {model}", self.root));
        let mut state = self.state(client, model).await;
        if claim.is_some() && state == Some(ModelState::NotLoaded) {
            self.request_load(client, model, deadline).await?;
            state = self.state(client, model).await;
        }
        loop {
            match state {
                // A server that stops reporting the state after accepting the
                // load is taken at its word.
                Some(ModelState::Loaded) | None => return Ok(()),
                Some(ModelState::NotLoaded | ModelState::Loading) => {
                    tokio::time::sleep(POLL_INTERVAL).await;
                    state = self.state(client, model).await;
                }
            }
        }
    }

    async fn request_load(
        &self,
        client: &reqwest::Client,
        model: &str,
        deadline: tokio::time::Instant,
    ) -> anyhow::Result<()> {
        let response = client
            .post(format!("{}/api/v1/models/load", self.root))
            .json(&serde_json::json!({ "model": model }))
            // A second past the wait's deadline, so the wait reports the
            // timeout rather than the request.
            .timeout(
                deadline.saturating_duration_since(tokio::time::Instant::now())
                    + Duration::from_secs(1),
            )
            .send()
            .await
            .map_err(|error| unavailable(format!("LM Studio could not load {model}: {error}")))?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let body = response.text().await.unwrap_or_default();
        let reason = super::response_parser::api_error_message("lmstudio", status, &body);
        Err(unavailable(format!(
            "LM Studio could not load {model}: {reason}"
        )))
    }
}

/// A load that failed or timed out. A retry cannot fix it in this turn, so
/// it is not retried, and embeddings stop for the session.
fn unavailable(message: String) -> anyhow::Error {
    crate::traits::retry::RetryError::ModelUnavailable { message }.into()
}

/// The error when a load does not finish within `limit`.
#[must_use]
pub fn load_timeout_error(model: &str, limit: Duration) -> String {
    format!(
        "LM Studio did not finish loading {model} within {} seconds.",
        limit.as_secs()
    )
}
