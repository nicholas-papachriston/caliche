use super::config::{
    CandleBootstrapConfig, GgufBootstrapConfig, LocalInferenceBootstrapConfig,
    LocalInferenceConfig, OrtBootstrapConfig,
};
use super::provider::LocalInferenceProvider;
use crate::llm::env_compat::env_nonempty;
use dripstone::inference::InferenceBackendType;
use dripstone::loader::{
    DownloadOptions, HuggingFaceLoader, ModelSource, atomic_copy, ensure_downloaded,
    huggingface_resolve_url, huggingface_token_from_env, normalize_huggingface_token,
    sanitize_relative_path,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "local-inference-base")]
impl LocalInferenceBootstrapConfig {
    /// Set an explicit `HuggingFace` token for private or gated model downloads.
    #[must_use]
    pub fn with_huggingface_token(mut self, token: impl Into<String>) -> Self {
        self.huggingface_token = normalize_huggingface_token(token.into());
        self
    }

    /// Clear any explicit `HuggingFace` token so env/default credential discovery is used.
    #[must_use]
    pub fn without_huggingface_token(mut self) -> Self {
        self.huggingface_token = None;
        self
    }

    /// Store downloaded artifacts in this exact directory for the active backend.
    #[must_use]
    pub fn with_model_storage_dir(mut self, local_dir: impl Into<PathBuf>) -> Self {
        self.set_model_storage_dir(local_dir);
        self
    }

    pub(crate) fn set_model_storage_dir(&mut self, local_dir: impl Into<PathBuf>) {
        let local_dir = local_dir.into();
        if let Some(candle) = self.candle.as_mut() {
            candle.local_dir.clone_from(&local_dir);
        }
        if let Some(ort) = self.ort.as_mut() {
            ort.local_dir.clone_from(&local_dir);
        }
        if let Some(gguf) = self.gguf.as_mut() {
            gguf.local_dir = local_dir;
        }
    }

    pub(crate) fn apply(&self, config: &mut LocalInferenceConfig) -> anyhow::Result<()> {
        if self.create_huggingface_cache_dir {
            self.ensure_huggingface_cache_dir()?;
        }

        match config.backend {
            InferenceBackendType::Ort => {
                if let Some(ort) = &self.ort {
                    self.bootstrap_ort(config, ort)?;
                }
            }
            InferenceBackendType::Candle | InferenceBackendType::Mlx => {
                if let Some(candle) = &self.candle {
                    self.bootstrap_candle(config, candle)?;
                }
            }
            InferenceBackendType::LlamaCpp => {
                if let Some(gguf) = &self.gguf {
                    self.bootstrap_gguf(config, gguf)?;
                }
            }
        }

        Ok(())
    }

    fn ensure_huggingface_cache_dir(&self) -> anyhow::Result<()> {
        let cache_dir = if let Some(dir) = &self.huggingface_cache_dir {
            dir.clone()
        } else if let Ok(hf_home) = std::env::var("HF_HOME") {
            PathBuf::from(hf_home)
        } else {
            let home = std::env::var("HOME").map_err(|e| {
                anyhow::anyhow!("unable to resolve HOME for HuggingFace cache: {e}")
            })?;
            PathBuf::from(home).join(".cache/huggingface")
        };

        fs::create_dir_all(&cache_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to create HuggingFace cache dir {}: {}",
                cache_dir.display(),
                e
            )
        })
    }

    fn bootstrap_candle(
        &self,
        config: &mut LocalInferenceConfig,
        candle: &CandleBootstrapConfig,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(&candle.local_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to create Candle model directory {}: {}",
                candle.local_dir.display(),
                e
            )
        })?;

        let (repo, revision) = match (&candle.repo, config.source.huggingface_repo()) {
            (Some(repo), _) => (repo.clone(), candle.revision.clone()),
            (None, Some((repo, revision))) => (
                repo.to_string(),
                candle
                    .revision
                    .clone()
                    .or_else(|| revision.map(ToString::to_string)),
            ),
            (None, None) => {
                return Err(anyhow::anyhow!(
                    "candle bootstrap requires repo override or HuggingFace source"
                ));
            }
        };

        let config_target = candle.local_dir.join("config.json");
        let tokenizer_target = candle.local_dir.join("tokenizer.json");

        if !config_target.exists() {
            self.copy_or_download_hf_file(
                &repo,
                revision.as_deref(),
                "config.json",
                &config_target,
            )?;
        }
        if !tokenizer_target.exists() {
            self.copy_or_download_hf_file(
                &repo,
                revision.as_deref(),
                "tokenizer.json",
                &tokenizer_target,
            )?;
        }

        if !Self::has_candle_weights(&candle.local_dir) {
            let model_target = candle.local_dir.join("model.safetensors");
            if self
                .copy_or_download_hf_file(
                    &repo,
                    revision.as_deref(),
                    "model.safetensors",
                    &model_target,
                )
                .is_err()
            {
                let index_target = candle.local_dir.join("model.safetensors.index.json");
                self.copy_or_download_hf_file(
                    &repo,
                    revision.as_deref(),
                    "model.safetensors.index.json",
                    &index_target,
                )?;
                self.download_sharded_weights_from_index(
                    &repo,
                    revision.as_deref(),
                    &candle.local_dir,
                    &index_target,
                )?;
            }
        }

        if candle.redirect_source_to_local {
            config.source = ModelSource::local(candle.local_dir.clone());
        }

        Ok(())
    }

    fn bootstrap_gguf(
        &self,
        config: &mut LocalInferenceConfig,
        gguf: &GgufBootstrapConfig,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(&gguf.local_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to create GGUF model directory {}: {}",
                gguf.local_dir.display(),
                e
            )
        })?;

        let (repo, revision) = match (&gguf.repo, config.source.huggingface_repo()) {
            (Some(repo), _) => (repo.clone(), gguf.revision.clone()),
            (None, Some((repo, revision))) => (
                repo.to_string(),
                gguf.revision
                    .clone()
                    .or_else(|| revision.map(ToString::to_string)),
            ),
            (None, None) => {
                if let ModelSource::Local { path } = &config.source {
                    let resolved = if path.is_file() {
                        path.clone()
                    } else {
                        // Keep local directory sources as-is for llama.cpp dir resolution.
                        path.clone()
                    };
                    if gguf.redirect_source_to_local {
                        config.source = ModelSource::local(resolved);
                    }
                    return Ok(());
                }
                return Err(anyhow::anyhow!(
                    "gguf bootstrap requires repo override, HuggingFace source, or local .gguf path"
                ));
            }
        };

        let filename = gguf
            .filename
            .clone()
            .unwrap_or_else(|| LocalInferenceProvider::resolve_gguf_filename_for_repo(&repo));

        let target = sanitize_relative_path(&gguf.local_dir, &filename)
            .map_err(|e| anyhow::anyhow!("unsafe GGUF filename '{filename}': {e}"))?;
        if !target.exists() {
            self.copy_or_download_hf_file(&repo, revision.as_deref(), &filename, &target)?;
        }

        if gguf.redirect_source_to_local {
            config.source = ModelSource::local(target);
        }

        Ok(())
    }

    fn bootstrap_ort(
        &self,
        config: &mut LocalInferenceConfig,
        ort: &OrtBootstrapConfig,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(&ort.local_dir).map_err(|e| {
            anyhow::anyhow!(
                "failed to create ORT model directory {}: {}",
                ort.local_dir.display(),
                e
            )
        })?;

        let tokenizer_target = ort.local_dir.join("tokenizer.json");
        if !tokenizer_target.exists() {
            self.copy_or_download_hf_file(
                &ort.repo,
                ort.revision.as_deref(),
                "tokenizer.json",
                &tokenizer_target,
            )?;
        }

        let has_onnx = ort
            .onnx_candidates
            .iter()
            .any(|candidate| ort.local_dir.join(candidate).exists());
        if !has_onnx {
            let mut downloaded = false;
            for candidate in &ort.onnx_candidates {
                let target = sanitize_relative_path(&ort.local_dir, candidate).map_err(|e| {
                    anyhow::anyhow!("unsafe ONNX candidate path '{candidate}': {e}")
                })?;
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent).map_err(|e| {
                        anyhow::anyhow!(
                            "failed to create ONNX parent directory {}: {}",
                            parent.display(),
                            e
                        )
                    })?;
                }

                if self
                    .copy_or_download_hf_file(
                        &ort.repo,
                        ort.revision.as_deref(),
                        candidate,
                        &target,
                    )
                    .is_ok()
                {
                    downloaded = true;
                    let canonical_target = ort.local_dir.join("model.onnx");
                    if target != canonical_target && !canonical_target.exists() {
                        fs::copy(&target, &canonical_target).map_err(|e| {
                            anyhow::anyhow!(
                                "failed to copy ONNX file {} to {}: {}",
                                target.display(),
                                canonical_target.display(),
                                e
                            )
                        })?;
                    }
                    break;
                }
            }

            if !downloaded {
                return Err(anyhow::anyhow!(
                    "failed to bootstrap ONNX artifacts from repository {}",
                    ort.repo
                ));
            }
        }

        if ort.redirect_source_to_local {
            config.source = ModelSource::local(ort.local_dir.join("model.onnx"));
        }

        Ok(())
    }

    fn has_candle_weights(model_dir: &Path) -> bool {
        model_dir.join("model.safetensors").exists()
            || model_dir.join("pytorch_model.safetensors").exists()
            || model_dir.join("model-00001-of-00001.safetensors").exists()
            || model_dir.join("consolidated.safetensors").exists()
            || model_dir.join("model.safetensors.index.json").exists()
    }

    fn download_options(&self) -> DownloadOptions {
        let mut options = DownloadOptions::default()
            .with_max_bytes(self.max_download_bytes)
            .with_huggingface_token(self.effective_huggingface_token().as_deref());
        if let Some(reserve) = env_nonempty("CALICHE_LOCAL_INFERENCE_DISK_RESERVE_BYTES")
            .and_then(|value| value.parse::<u64>().ok())
        {
            options = options.with_reserve_free_bytes(reserve);
        }
        options
    }

    fn copy_or_download_hf_file(
        &self,
        repo: &str,
        revision: Option<&str>,
        remote_file: &str,
        target: &Path,
    ) -> anyhow::Result<()> {
        if target.is_file() {
            return Ok(());
        }

        let token = self.effective_huggingface_token();
        let options = self.download_options();
        let loader = HuggingFaceLoader::new_with_token(repo, revision, token)
            .map_err(|e| {
                anyhow::anyhow!("failed to initialize HuggingFace loader for {repo}: {e}")
            })?
            .with_curl_fallback(self.allow_curl_fallback)
            .with_download_options(options.clone());

        match loader.get_file(remote_file) {
            Ok(src) => atomic_copy(&src, target, &options).map_err(|e| {
                anyhow::anyhow!(
                    "failed to copy {} from {} to {}: {}",
                    remote_file,
                    src.display(),
                    target.display(),
                    e
                )
            })?,
            Err(loader_error) if self.allow_curl_fallback => {
                // Hub path failed even with loader curl fallback; download
                // directly to the bootstrap target (still locked/verified).
                let url = huggingface_resolve_url(repo, revision, remote_file);
                ensure_downloaded(&url, target, &options).map_err(|e| {
                    anyhow::anyhow!(
                        "failed to fetch {remote_file} from {repo} via hub ({loader_error}) and direct download ({e})"
                    )
                })?;
            }
            Err(loader_error) => {
                return Err(anyhow::anyhow!(
                    "failed to fetch {remote_file} from {repo}: {loader_error}"
                ));
            }
        }

        Ok(())
    }

    fn effective_huggingface_token(&self) -> Option<String> {
        self.huggingface_token
            .as_deref()
            .and_then(normalize_huggingface_token)
            .or_else(huggingface_token_from_env)
    }

    fn download_sharded_weights_from_index(
        &self,
        repo: &str,
        revision: Option<&str>,
        model_dir: &Path,
        index_path: &Path,
    ) -> anyhow::Result<()> {
        let index_content = fs::read_to_string(index_path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {}", index_path.display(), e))?;
        let index_json: serde_json::Value = serde_json::from_str(&index_content)
            .map_err(|e| anyhow::anyhow!("failed to parse safetensors index JSON: {e}"))?;

        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("safetensors index missing 'weight_map' object"))?;

        let mut shard_files = BTreeSet::new();
        for value in weight_map.values() {
            if let Some(file_name) = value.as_str() {
                shard_files.insert(file_name.to_string());
            }
        }
        if shard_files.is_empty() {
            return Err(anyhow::anyhow!(
                "safetensors index weight_map did not include shard files"
            ));
        }

        for shard in shard_files {
            let target = sanitize_relative_path(model_dir, &shard).map_err(|e| {
                anyhow::anyhow!("unsafe shard path '{shard}' in safetensors index: {e}")
            })?;
            if !target.exists() {
                self.copy_or_download_hf_file(repo, revision, &shard, &target)?;
            }
        }

        Ok(())
    }
}
