//! llama.cpp / GGUF inference backend.
//!
//! Generation runs on a dedicated worker thread that owns the `!Send` model.
//! Each request creates a fresh `LlamaContext` borrowed from that model (no
//! lifetime transmute) and drops it before the next request.

use crate::errors::DripstoneError;
use crate::inference::{
    DeviceConfig, GenerationParams, InferenceBackend, InferenceConfig, ModelInfo,
};
use crate::loader::{
    DownloadOptions, ModelKind, ModelSource, ResourceBudget, atomic_copy, builtin_catalog,
    cache_root, ensure_downloaded, ensure_resource_budget, estimate_runtime_memory,
    huggingface_resolve_url, max_safe_n_ctx, sanitize_relative_path,
};
use async_stream::stream;
use async_trait::async_trait;
use futures_util::Stream;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;

/// Default context size when the model does not advertise one.
///
/// Sized for agent-style prompts (system + tools + history), not toy max_tokens.
const DEFAULT_N_CTX: u32 = 8192;
/// Hard ceiling for on-demand context growth.
const MAX_N_CTX: u32 = 32_768;
/// Default decode batch size (tokens per llama_decode call during prefill).
const DEFAULT_N_BATCH: u32 = 512;
/// Bound streaming channel so producers cannot unbounded-buffer token text.
const STREAM_CHANNEL_CAPACITY: usize = 64;

enum WorkerCommand {
    Generate {
        prompt: String,
        params: GenerationParams,
        /// When set, pieces are streamed; when `None`, text is returned via `reply`.
        piece_tx: Option<SyncSender<Result<String, DripstoneError>>>,
        reply: SyncSender<Result<String, DripstoneError>>,
    },
    Shutdown,
}

/// llama.cpp-backed GGUF inference.
pub struct LlamaCppInference {
    model_info: ModelInfo,
    cmd_tx: SyncSender<WorkerCommand>,
    worker: Option<JoinHandle<()>>,
}

struct WorkerState {
    backend: LlamaBackend,
    model: LlamaModel,
    /// Soft target context; may grow up to [`MAX_N_CTX`] when memory allows.
    n_ctx: u32,
    n_batch: u32,
    /// On-disk weight footprint used for RAM preflight / context caps.
    weight_bytes: u64,
}

impl LlamaCppInference {
    /// Load a GGUF model from configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the model cannot be resolved, downloaded, or loaded.
    pub async fn new(config: InferenceConfig) -> Result<Self, DripstoneError> {
        let n_gpu_layers = gpu_layers_for_device(config.device);
        let model_path = resolve_gguf_path(&config.source).await?;
        let model_name = model_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("gguf-model")
            .to_string();

        let n_ctx = resolve_n_ctx(config.default_params.max_tokens);

        tokio::task::spawn_blocking(move || load_model(model_path, model_name, n_gpu_layers, n_ctx))
            .await
            .map_err(|error| {
                DripstoneError::storage_with_source("llamacpp load task failed", error)
            })?
    }
}

impl Drop for LlamaCppInference {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

fn load_model(
    model_path: PathBuf,
    model_name: String,
    n_gpu_layers: u32,
    n_ctx: u32,
) -> Result<LlamaCppInference, DripstoneError> {
    let weight_bytes = std::fs::metadata(&model_path)
        .map(|meta| meta.len())
        .map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to stat GGUF model '{}'", model_path.display()),
                error,
            )
        })?;

    let n_ctx = clamp_n_ctx_to_budget(n_ctx, weight_bytes)?;
    let n_batch = DEFAULT_N_BATCH.min(n_ctx).max(1);
    let estimate = estimate_runtime_memory(weight_bytes, n_ctx, None);
    let budget = ResourceBudget::from_host()?;
    ensure_resource_budget(&estimate, &budget)?;

    let backend = LlamaBackend::init().map_err(|error| {
        DripstoneError::storage_with_source("failed to initialize llama.cpp backend", error)
    })?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
    let model =
        LlamaModel::load_from_file(&backend, &model_path, &model_params).map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to load GGUF model '{}'", model_path.display()),
                error,
            )
        })?;

    let model_info = ModelInfo::new(model_name)
        .with_architecture("gguf")
        .with_context_length(n_ctx as usize);

    let (cmd_tx, cmd_rx) = mpsc::sync_channel::<WorkerCommand>(8);
    let worker = std::thread::Builder::new()
        .name("llamacpp-infer".into())
        .spawn(move || {
            let mut state = WorkerState {
                backend,
                model,
                n_ctx,
                n_batch,
                weight_bytes,
            };
            worker_loop(&mut state, cmd_rx);
        })
        .map_err(|error| {
            DripstoneError::storage_with_source("failed to spawn llama.cpp worker thread", error)
        })?;

    Ok(LlamaCppInference {
        model_info,
        cmd_tx,
        worker: Some(worker),
    })
}

fn worker_loop(state: &mut WorkerState, cmd_rx: Receiver<WorkerCommand>) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            WorkerCommand::Shutdown => break,
            WorkerCommand::Generate {
                prompt,
                params,
                piece_tx,
                reply,
            } => {
                let result = match piece_tx {
                    Some(tx) => generate_into(state, &prompt, &params, |piece| {
                        tx.send(Ok(piece)).map_err(|error| {
                            DripstoneError::storage(format!("llamacpp stream closed: {error}"))
                        })
                    })
                    .map(|()| String::new()),
                    None => {
                        let mut tokens = Vec::new();
                        generate_into(state, &prompt, &params, |piece| {
                            tokens.push(piece);
                            Ok(())
                        })
                        .map(|()| tokens.concat())
                    }
                };
                let _ = reply.send(result);
            }
        }
    }
}

fn generate_into(
    state: &mut WorkerState,
    prompt: &str,
    params: &GenerationParams,
    mut on_piece: impl FnMut(String) -> Result<(), DripstoneError>,
) -> Result<(), DripstoneError> {
    let prompt_tokens = state
        .model
        .str_to_token(prompt, AddBos::Always)
        .map_err(|error| {
            DripstoneError::storage_with_source("failed to tokenize prompt with llama.cpp", error)
        })?;
    if prompt_tokens.is_empty() {
        return Err(DripstoneError::configuration(
            "llama.cpp tokenization produced an empty prompt",
        ));
    }

    let max_tokens = params.max_tokens.max(1);
    grow_context_if_needed(state, prompt_tokens.len(), max_tokens)?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(state.n_ctx).or_else(|| NonZeroU32::new(DEFAULT_N_CTX)))
        .with_n_batch(state.n_batch);
    // Context borrows `model`/`backend` for this request only — no lifetime erase.
    let mut ctx = state
        .model
        .new_context(&state.backend, ctx_params)
        .map_err(|error| {
            DripstoneError::storage_with_source("failed to create llama.cpp context", error)
        })?;

    let n_batch = usize::try_from(state.n_batch).unwrap_or(512).max(1);
    let n_ctx = i32::try_from(ctx.n_ctx()).unwrap_or(i32::MAX);
    let prompt_len = i32::try_from(prompt_tokens.len()).unwrap_or(i32::MAX);
    if prompt_len >= n_ctx {
        return Err(DripstoneError::configuration(format!(
            "prompt length {prompt_len} exceeds llama.cpp context size {n_ctx} \
             (raise max context or shorten the system prompt)"
        )));
    }

    let mut batch = LlamaBatch::new(n_batch, 1);
    decode_prompt_chunked(&mut ctx, &mut batch, &prompt_tokens, n_batch)?;

    let mut sampler = build_sampler(params);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = prompt_len;
    let mut generated = 0_usize;
    let mut text = String::new();
    let max_stop_len = params
        .stop_sequences
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0);

    while generated < max_tokens && n_cur < n_ctx {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if state.model.is_eog_token(token) {
            break;
        }

        let piece = state
            .model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|error| {
                DripstoneError::storage_with_source("failed to decode llama.cpp token", error)
            })?;

        if emit_piece_with_stop(
            &mut text,
            piece,
            &params.stop_sequences,
            max_stop_len,
            &mut on_piece,
        )? {
            break;
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true).map_err(|error| {
            DripstoneError::storage_with_source("failed to queue next llama.cpp token", error)
        })?;
        ctx.decode(&mut batch).map_err(|error| {
            DripstoneError::storage_with_source("llama.cpp decode step failed", error)
        })?;
        n_cur += 1;
        generated += 1;
    }

    Ok(())
}

fn grow_context_if_needed(
    state: &mut WorkerState,
    prompt_token_count: usize,
    max_tokens: usize,
) -> Result<(), DripstoneError> {
    let prompt_len_u = u32::try_from(prompt_token_count).unwrap_or(u32::MAX);
    let needed_ctx = prompt_len_u
        .saturating_add(u32::try_from(max_tokens).unwrap_or(1))
        .saturating_add(64)
        .clamp(DEFAULT_N_CTX, MAX_N_CTX);
    if needed_ctx <= state.n_ctx {
        return Ok(());
    }
    let allowed = clamp_n_ctx_to_budget(needed_ctx, state.weight_bytes)?;
    if allowed < needed_ctx {
        return Err(DripstoneError::configuration(format!(
            "prompt needs llama.cpp context {needed_ctx} but host memory only safely supports \
             {allowed}; shorten the prompt or free RAM"
        )));
    }
    tracing::info!(
        old_n_ctx = state.n_ctx,
        new_n_ctx = allowed,
        prompt_tokens = prompt_token_count,
        "growing llama.cpp context for prompt"
    );
    state.n_ctx = allowed;
    state.n_batch = DEFAULT_N_BATCH.min(allowed).max(1);
    Ok(())
}

/// Returns `true` when a stop sequence was hit and generation should stop.
fn emit_piece_with_stop(
    text: &mut String,
    piece: String,
    stop_sequences: &[String],
    max_stop_len: usize,
    mut on_piece: impl FnMut(String) -> Result<(), DripstoneError>,
) -> Result<bool, DripstoneError> {
    if piece.is_empty() {
        return Ok(false);
    }
    text.push_str(&piece);
    if max_stop_len > 0
        && let Some(stop) = stop_sequences
            .iter()
            .find(|sequence| text.contains(sequence.as_str()))
    {
        if let Some(idx) = text.find(stop.as_str()) {
            let kept = text[..idx].to_string();
            let emitted_prefix_len = text.len() - piece.len();
            if kept.len() > emitted_prefix_len {
                on_piece(kept[emitted_prefix_len..].to_string())?;
            }
        }
        return Ok(true);
    }
    on_piece(piece)?;
    if max_stop_len > 0 && text.len() > max_stop_len.saturating_mul(4) {
        let keep_from = text.len().saturating_sub(max_stop_len.saturating_mul(2));
        text.replace_range(..keep_from, "");
    }
    Ok(false)
}

fn clamp_n_ctx_to_budget(requested: u32, weight_bytes: u64) -> Result<u32, DripstoneError> {
    let budget = ResourceBudget::from_host()?;
    let safe = max_safe_n_ctx(weight_bytes, &budget, None, MAX_N_CTX);
    if safe == 0 {
        return Err(DripstoneError::configuration(format!(
            "insufficient memory to allocate any llama.cpp context for {} byte weights \
             ({} available, {} reserved)",
            weight_bytes, budget.available_ram_bytes, budget.reserve_ram_bytes
        )));
    }
    let clamped = requested.min(safe).max(1);
    if clamped < requested {
        tracing::warn!(
            requested,
            clamped,
            available = budget.available_ram_bytes,
            "clamping llama.cpp n_ctx to host memory budget"
        );
    }
    let estimate = estimate_runtime_memory(weight_bytes, clamped, None);
    ensure_resource_budget(&estimate, &budget)?;
    Ok(clamped)
}

fn decode_prompt_chunked(
    ctx: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch,
    prompt_tokens: &[LlamaToken],
    n_batch: usize,
) -> Result<(), DripstoneError> {
    let prompt_len = prompt_tokens.len();
    let mut offset = 0_usize;
    while offset < prompt_len {
        batch.clear();
        let end = (offset + n_batch).min(prompt_len);
        for (i, token) in prompt_tokens[offset..end].iter().enumerate() {
            let pos = i32::try_from(offset + i).unwrap_or(i32::MAX);
            let is_last = offset + i + 1 == prompt_len;
            batch.add(*token, pos, &[0], is_last).map_err(|error| {
                DripstoneError::storage_with_source("failed to fill llama.cpp batch", error)
            })?;
        }
        ctx.decode(batch).map_err(|error| {
            DripstoneError::storage_with_source("llama.cpp prompt decode failed", error)
        })?;
        offset = end;
    }
    Ok(())
}

fn resolve_n_ctx(max_tokens: usize) -> u32 {
    let needed = max_tokens.saturating_add(1024).max(DEFAULT_N_CTX as usize);
    u32::try_from(needed.min(MAX_N_CTX as usize)).unwrap_or(DEFAULT_N_CTX)
}

fn gpu_layers_for_device(device: DeviceConfig) -> u32 {
    match device {
        DeviceConfig::Cpu => 0,
        DeviceConfig::Cuda { .. } | DeviceConfig::Metal | DeviceConfig::Mlx => 999,
        DeviceConfig::Auto => 999,
    }
}

fn build_sampler(params: &GenerationParams) -> LlamaSampler {
    let seed = fastrand::u32(..);
    if params.temperature <= 0.0 {
        return LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    }

    let mut samplers = Vec::new();
    if let Some(top_k) = params.top_k {
        let top_k = i32::try_from(top_k).unwrap_or(i32::MAX);
        samplers.push(LlamaSampler::top_k(top_k));
    }
    if let Some(top_p) = params.top_p {
        samplers.push(LlamaSampler::top_p(top_p, 1));
    }
    if let Some(penalty) = params.repetition_penalty {
        samplers.push(LlamaSampler::penalties(64, penalty, 0.0, 0.0));
    }
    samplers.push(LlamaSampler::temp(params.temperature));
    samplers.push(LlamaSampler::dist(seed));
    LlamaSampler::chain_simple(samplers)
}

async fn resolve_gguf_path(source: &ModelSource) -> Result<PathBuf, DripstoneError> {
    match source {
        ModelSource::Local { path } => {
            source.validate_local()?;
            resolve_local_gguf(path)
        }
        ModelSource::HuggingFace { repo, revision } => {
            let (filename, expected_bytes) = resolve_gguf_artifact_for_repo(repo);
            download_gguf_to_cache(repo, revision.as_deref(), &filename, expected_bytes).await
        }
    }
}

fn resolve_gguf_artifact_for_repo(repo: &str) -> (String, Option<u64>) {
    if let Some(spec) = builtin_catalog()
        .iter()
        .find(|spec| spec.repo == repo && matches!(spec.kind, ModelKind::Gguf))
        && let Some(artifact) = spec.artifacts.first()
    {
        return (artifact.remote_path.to_string(), artifact.expected_bytes);
    }
    (guess_gguf_filename(repo), None)
}

fn resolve_local_gguf(path: &Path) -> Result<PathBuf, DripstoneError> {
    if path.is_file() {
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return Ok(path.to_path_buf());
        }
        return Err(DripstoneError::configuration(format!(
            "local llama.cpp source must be a .gguf file: {}",
            path.display()
        )));
    }

    if path.is_dir() {
        let mut matches = Vec::new();
        let entries = std::fs::read_dir(path).map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to read GGUF directory '{}'", path.display()),
                error,
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                DripstoneError::storage_with_source(
                    format!(
                        "failed to read GGUF directory entry under '{}'",
                        path.display()
                    ),
                    error,
                )
            })?;
            let candidate = entry.path();
            if candidate
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            {
                matches.push(candidate);
            }
        }
        matches.sort();
        return match matches.as_slice() {
            [only] => Ok(only.clone()),
            [] => Err(DripstoneError::storage(format!(
                "no .gguf file found under '{}'",
                path.display()
            ))),
            _ => Err(DripstoneError::configuration(format!(
                "multiple .gguf files found under '{}'; pass an explicit file path",
                path.display()
            ))),
        };
    }

    Err(DripstoneError::storage(format!(
        "GGUF path does not exist: {}",
        path.display()
    )))
}

fn guess_gguf_filename(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    if name.to_ascii_lowercase().ends_with(".gguf") {
        return name.to_string();
    }
    format!("{name}.gguf")
}

async fn download_gguf_to_cache(
    repo: &str,
    revision: Option<&str>,
    filename: &str,
    expected_bytes: Option<u64>,
) -> Result<PathBuf, DripstoneError> {
    let repo = repo.to_string();
    let revision = revision.map(str::to_string);
    let filename = filename.to_string();
    tokio::task::spawn_blocking(move || {
        let cache_root = gguf_cache_root()
            .join(sanitize_component(&repo))
            .join(sanitize_component(revision.as_deref().unwrap_or("main")));
        let dest = sanitize_relative_path(&cache_root, &filename)?;
        let url = huggingface_resolve_url(&repo, revision.as_deref(), &filename);
        let mut options = DownloadOptions::default().with_expected_bytes(expected_bytes);
        if expected_bytes.is_some() {
            options = options.with_check_remote_size(false);
        }
        ensure_downloaded(&url, &dest, &options)
    })
    .await
    .map_err(|error| DripstoneError::storage_with_source("gguf download task failed", error))?
}

fn gguf_cache_root() -> PathBuf {
    cache_root().join("gguf")
}

fn sanitize_component(component: &str) -> String {
    let sanitized: String = component
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".into()
    } else {
        sanitized
    }
}

/// Copy a hub-cached GGUF into `dest` with disk preflight (used by caliche bootstrap).
///
/// # Errors
///
/// Returns an error when copy or verification fails.
pub fn materialize_gguf_file(source: &Path, dest: &Path) -> Result<(), DripstoneError> {
    let options = DownloadOptions::default();
    atomic_copy(source, dest, &options)
}

#[async_trait]
impl InferenceBackend for LlamaCppInference {
    async fn generate(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, DripstoneError> {
        let prompt = prompt.to_string();
        let params = params.clone();
        let cmd_tx = self.cmd_tx.clone();
        tokio::task::spawn_blocking(move || {
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            cmd_tx
                .send(WorkerCommand::Generate {
                    prompt,
                    params,
                    piece_tx: None,
                    reply: reply_tx,
                })
                .map_err(|_| DripstoneError::storage("llama.cpp worker is shut down"))?;
            reply_rx
                .recv()
                .map_err(|_| DripstoneError::storage("llama.cpp worker dropped reply channel"))?
        })
        .await
        .map_err(|error| {
            DripstoneError::storage_with_source("llamacpp generate task failed", error)
        })?
    }

    fn stream(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Pin<Box<dyn Stream<Item = Result<String, DripstoneError>> + Send + '_>> {
        let prompt = prompt.to_string();
        let params = params.clone();
        let cmd_tx = self.cmd_tx.clone();

        Box::pin(stream! {
            let (piece_tx, piece_rx) = mpsc::sync_channel(STREAM_CHANNEL_CAPACITY);
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            let mut task = tokio::task::spawn_blocking(move || {
                cmd_tx
                    .send(WorkerCommand::Generate {
                        prompt,
                        params,
                        piece_tx: Some(piece_tx),
                        reply: reply_tx,
                    })
                    .map_err(|_| DripstoneError::storage("llama.cpp worker is shut down"))?;
                reply_rx
                    .recv()
                    .map_err(|_| DripstoneError::storage("llama.cpp worker dropped reply channel"))?
            });

            loop {
                // Drain pieces without blocking the async runtime forever.
                match piece_rx.try_recv() {
                    Ok(Ok(token)) => {
                        yield Ok(token);
                        continue;
                    }
                    Ok(Err(error)) => {
                        yield Err(error);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // Wait for worker completion, then drain.
                        break;
                    }
                }

                tokio::select! {
                    biased;
                    join = &mut task => {
                        match join {
                            Ok(Ok(_)) => {}
                            Ok(Err(error)) => yield Err(error),
                            Err(error) => yield Err(DripstoneError::storage_with_source(
                                "llamacpp stream task failed",
                                error,
                            )),
                        }
                        while let Ok(piece) = piece_rx.recv() {
                            yield piece;
                        }
                        break;
                    }
                    () = tokio::time::sleep(std::time::Duration::from_millis(2)) => {}
                }
            }
        })
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn supports_kv_cache(&self) -> bool {
        // Intra-request KV lives inside a per-request LlamaContext that is dropped
        // after each generate/stream call. Cross-request prefix cache is not exposed.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_local_gguf_requires_extension() {
        let temp = tempfile::tempdir().expect("tempdir");
        let bad = temp.path().join("model.bin");
        std::fs::write(&bad, b"x").expect("write");
        assert!(resolve_local_gguf(&bad).is_err());
    }

    #[test]
    fn resolve_local_gguf_picks_single_file_in_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let gguf = temp.path().join("model.gguf");
        std::fs::write(&gguf, b"gguf").expect("write");
        assert_eq!(resolve_local_gguf(temp.path()).expect("resolve"), gguf);
    }

    #[test]
    fn guess_filename_from_repo() {
        assert_eq!(
            guess_gguf_filename("someone/TinyLlama-1.1B-Chat-v1.0-GGUF"),
            "TinyLlama-1.1B-Chat-v1.0-GGUF.gguf"
        );
        assert_eq!(
            guess_gguf_filename("dir/model.Q4_K_M.gguf"),
            "model.Q4_K_M.gguf"
        );
    }

    #[test]
    fn catalog_repo_resolves_nested_gguf_path() {
        let (filename, expected) = resolve_gguf_artifact_for_repo("ggml-org/models-moved");
        assert_eq!(filename, "tinyllamas/stories15M-q4_0.gguf");
        assert_eq!(expected, Some(19_077_344));
    }

    #[test]
    fn gpu_layers_zero_on_cpu() {
        assert_eq!(gpu_layers_for_device(DeviceConfig::Cpu), 0);
        assert!(gpu_layers_for_device(DeviceConfig::Metal) > 0);
        assert!(gpu_layers_for_device(DeviceConfig::Auto) > 0);
    }

    #[test]
    fn resolve_n_ctx_balances_headroom_and_cap() {
        assert_eq!(resolve_n_ctx(16), DEFAULT_N_CTX);
        assert_eq!(resolve_n_ctx(9000), 10_024);
        assert_eq!(resolve_n_ctx(40_000), MAX_N_CTX);
    }
}
