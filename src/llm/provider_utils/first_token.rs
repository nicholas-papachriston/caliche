//! The wait for a local model's first token.
//!
//! A local server (LM Studio, Ollama) may load the model and read the whole
//! prompt before it sends a byte. That can take minutes, so local providers
//! get a long first-token limit, separate from the stall limit between
//! chunks. While the first token is awaited, the stream carries
//! [`StreamEvent::Status`] lines so a client does not look hung.

use crate::messages::StreamEvent;
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// First-token limit for local providers when nothing sets one.
pub const DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(600);

/// Shortest first-token limit a setting may choose.
pub const MIN_LOCAL_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(10);

/// Longest first-token limit a setting may choose.
pub const MAX_LOCAL_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(3600);

/// Environment variable read when no setting was applied, in seconds.
pub const LOCAL_FIRST_TOKEN_TIMEOUT_ENV: &str = "CALICHE_LOCAL_FIRST_TOKEN_SECS";

/// Status line while a local server loads the model.
#[must_use]
pub fn loading_status(model: &str, server: &str) -> String {
    format!("Loading {model} in {server}...")
}

/// Status line while the first token is awaited and the model is loaded,
/// or its state is unknown.
pub const WAITING_FOR_FIRST_TOKEN: &str = "Waiting for the first token...";

/// Delay before the first status line, so a fast reply shows none.
const STATUS_DELAY: Duration = Duration::from_secs(2);

/// Gap between repeated status lines. A client that fails after a silent
/// period sees one at least this often.
const STATUS_INTERVAL: Duration = Duration::from_secs(15);

/// How long the loaded-model probe may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

static LOCAL_FIRST_TOKEN_SECS: AtomicU64 = AtomicU64::new(0);

/// Set the first-token limit for local providers in this process, clamped
/// to [`MIN_LOCAL_FIRST_TOKEN_TIMEOUT`]..=[`MAX_LOCAL_FIRST_TOKEN_TIMEOUT`].
/// Returns the limit applied.
pub fn set_local_first_token_timeout(limit: Duration) -> Duration {
    let limit = limit.clamp(MIN_LOCAL_FIRST_TOKEN_TIMEOUT, MAX_LOCAL_FIRST_TOKEN_TIMEOUT);
    LOCAL_FIRST_TOKEN_SECS.store(limit.as_secs(), Ordering::Relaxed);
    limit
}

/// The first-token limit for local providers: the value set with
/// [`set_local_first_token_timeout`], else [`LOCAL_FIRST_TOKEN_TIMEOUT_ENV`]
/// (clamped), else [`DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT`].
#[must_use]
pub fn local_first_token_timeout() -> Duration {
    match LOCAL_FIRST_TOKEN_SECS.load(Ordering::Relaxed) {
        0 => std::env::var(LOCAL_FIRST_TOKEN_TIMEOUT_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map_or(DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT, |secs| {
                Duration::from_secs(secs)
                    .clamp(MIN_LOCAL_FIRST_TOKEN_TIMEOUT, MAX_LOCAL_FIRST_TOKEN_TIMEOUT)
            }),
        secs => Duration::from_secs(secs),
    }
}

/// How to ask a local server about a model: whether it is loaded and its
/// context window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelLoadProbe {
    /// LM Studio: `GET {root}/api/v0/models/{model}` reports `state`,
    /// `loaded_context_length` and `max_context_length`.
    LmStudio { url: String, model: String },
    /// Ollama: `GET {root}/api/ps` lists the loaded models, each with its
    /// `context_length`.
    Ollama { url: String, model: String },
}

#[derive(Deserialize)]
struct LmStudioModel {
    state: String,
    loaded_context_length: Option<usize>,
    max_context_length: Option<usize>,
}

#[derive(Deserialize)]
struct OllamaPs {
    models: Vec<OllamaRunning>,
}

#[derive(Deserialize)]
struct OllamaRunning {
    name: String,
    context_length: Option<usize>,
}

impl ModelLoadProbe {
    fn model(&self) -> &str {
        match self {
            Self::LmStudio { model, .. } | Self::Ollama { model, .. } => model,
        }
    }

    fn server(&self) -> &'static str {
        match self {
            Self::LmStudio { .. } => "LM Studio",
            Self::Ollama { .. } => "Ollama",
        }
    }

    async fn fetch(&self, client: &reqwest::Client) -> Option<reqwest::Response> {
        let url = match self {
            Self::LmStudio { url, .. } | Self::Ollama { url, .. } => url,
        };
        client
            .get(url)
            .timeout(PROBE_TIMEOUT)
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()
    }

    /// The model's entry in Ollama's list of loaded models.
    async fn ollama_running(&self, client: &reqwest::Client) -> Option<Option<OllamaRunning>> {
        let ps: OllamaPs = self.fetch(client).await?.json().await.ok()?;
        let model = self.model();
        Some(
            ps.models
                .into_iter()
                .find(|running| running.name == model || running.name == format!("{model}:latest")),
        )
    }

    /// `Some(false)` when the server says the model is not loaded,
    /// `Some(true)` when it is, `None` when the server cannot tell.
    async fn model_loaded(&self, client: &reqwest::Client) -> Option<bool> {
        match self {
            Self::LmStudio { .. } => {
                let model: LmStudioModel = self.fetch(client).await?.json().await.ok()?;
                Some(model.state == "loaded")
            }
            Self::Ollama { .. } => Some(self.ollama_running(client).await?.is_some()),
        }
    }

    /// The context window, in tokens, that the server reports for the
    /// model: LM Studio's loaded context length, else the model's maximum;
    /// Ollama's context length for the loaded model. `None` when the
    /// server does not say.
    pub async fn context_window(&self, client: &reqwest::Client) -> Option<usize> {
        match self {
            Self::LmStudio { .. } => {
                let model: LmStudioModel = self.fetch(client).await?.json().await.ok()?;
                model.loaded_context_length.or(model.max_context_length)
            }
            Self::Ollama { .. } => self.ollama_running(client).await??.context_length,
        }
        .filter(|&window| window > 0)
    }

    /// The status line for this wait.
    pub async fn status(&self, client: &reqwest::Client) -> String {
        match self.model_loaded(client).await {
            Some(false) => loading_status(self.model(), self.server()),
            Some(true) | None => WAITING_FOR_FIRST_TOKEN.to_string(),
        }
    }
}

/// Forward `inner`. Until it yields its first item, yield a
/// [`StreamEvent::Status`] after a short delay and then at a fixed
/// interval. `status` gives the line; it runs once, at the first one.
pub fn with_waiting_status<'a, S, F>(
    inner: S,
    status: F,
) -> impl Stream<Item = anyhow::Result<StreamEvent>> + Send + 'a
where
    S: Stream<Item = anyhow::Result<StreamEvent>> + Send + 'a,
    F: Future<Output = String> + Send + 'a,
{
    async_stream::stream! {
        let mut inner = Box::pin(inner);
        let mut status = Some(Box::pin(status));
        let mut message = String::new();
        let mut ticks = tokio::time::interval_at(
            tokio::time::Instant::now() + STATUS_DELAY,
            STATUS_INTERVAL,
        );
        loop {
            tokio::select! {
                biased;
                item = inner.next() => {
                    match item {
                        Some(item) => {
                            yield item;
                            break;
                        }
                        None => return,
                    }
                }
                _ = ticks.tick() => {
                    if let Some(pending) = status.take() {
                        message = pending.await;
                    }
                    yield Ok(StreamEvent::Status { message: message.clone() });
                }
            }
        }
        while let Some(item) = inner.next().await {
            yield item;
        }
    }
}

/// Stream limits a local provider config may set. Unset values use
/// [`local_first_token_timeout`] and the default stall limit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocalStreamLimits {
    pub first_token: Option<Duration>,
    pub stall: Option<Duration>,
}

impl LocalStreamLimits {
    /// The first-token limit to apply.
    #[must_use]
    pub fn first_token(&self) -> Duration {
        self.first_token.unwrap_or_else(local_first_token_timeout)
    }

    /// The stall limit to apply.
    #[must_use]
    pub fn stall(&self) -> Duration {
        self.stall
            .unwrap_or(crate::llm::provider_utils::SseLimits::DEFAULT_STALL)
    }
}

/// The server root of an OpenAI-compatible base URL: `base_url` without a
/// trailing `/v1`.
#[must_use]
pub fn server_root(base_url: &str) -> &str {
    let trimmed = base_url.trim_end_matches('/');
    trimmed
        .strip_suffix("/v1")
        .map_or(trimmed, |root| root.trim_end_matches('/'))
}
