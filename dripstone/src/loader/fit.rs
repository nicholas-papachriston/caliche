//! Model fit estimation against host capabilities.
//!
//! Callers use this to tell users which catalog models are safe to load
//! before downloading or instantiating them.

use crate::errors::DripstoneError;
use crate::loader::catalog::{ModelKind, ModelSpec, builtin_catalog, find_model};
use crate::loader::resources::{
    HostMemory, ResourceBudget, ResourceEstimate, default_kv_bytes_per_token,
    estimate_runtime_memory, max_safe_n_ctx, reserve_ram_bytes_from_env,
};
use crate::runtime::RuntimeConfig;

pub use crate::loader::resources::ModelResourceHint;

/// How comfortably a model fits the current host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitStatus {
    /// Estimated usage is well under the usable budget (< 70%).
    FitsComfortably,
    /// Fits, but uses most of the usable budget (>= 70%).
    FitsTight,
    /// Would exceed usable RAM (or VRAM when probed).
    ExceedsMemory,
    /// Not enough metadata to judge.
    Unknown,
}

/// Options controlling fit evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitOptions {
    /// Target context length for the estimate (defaults per-model / 8192).
    pub n_ctx: Option<u32>,
    /// RAM reserve override; `None` uses env / default.
    pub reserve_ram_bytes: Option<u64>,
    /// Preferred device for accelerator-aware messaging.
    pub device: RuntimeConfig,
    /// Optional destination path used for disk headroom reporting.
    pub disk_path: Option<std::path::PathBuf>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            n_ctx: None,
            reserve_ram_bytes: None,
            device: RuntimeConfig::Auto,
            disk_path: None,
        }
    }
}

impl FitOptions {
    /// Set the evaluation context length.
    #[must_use]
    pub const fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = Some(n_ctx);
        self
    }

    /// Set the RAM reserve override.
    #[must_use]
    pub const fn with_reserve_ram_bytes(mut self, bytes: u64) -> Self {
        self.reserve_ram_bytes = Some(bytes);
        self
    }

    /// Set the device preference used in messaging.
    #[must_use]
    pub const fn with_device(mut self, device: RuntimeConfig) -> Self {
        self.device = device;
        self
    }
}

/// Host snapshot exposed to the rest of the ecosystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCapabilities {
    /// Physical RAM.
    pub total_ram_bytes: u64,
    /// Currently available RAM.
    pub available_ram_bytes: u64,
    /// Optional free VRAM.
    pub available_vram_bytes: Option<u64>,
    /// Optional free disk near `FitOptions::disk_path`.
    pub available_disk_bytes: Option<u64>,
    /// Resolved runtime preference for this host/build.
    pub recommended_device: RuntimeConfig,
    /// Human-readable notes (accelerator detection, unified memory, …).
    pub notes: Vec<String>,
}

/// Fit result for one catalog model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFit {
    /// Catalog id.
    pub model_id: String,
    /// Display name.
    pub display_name: String,
    /// Backend family.
    pub kind: ModelKind,
    /// Fit classification.
    pub status: FitStatus,
    /// Estimated weight bytes used for the judgment.
    pub estimated_weight_bytes: u64,
    /// Estimated total runtime bytes at `recommended_n_ctx`.
    pub estimated_runtime_bytes: u64,
    /// Context length used for the primary estimate.
    pub recommended_n_ctx: u32,
    /// Largest context that still fits the budget.
    pub max_safe_n_ctx: u32,
    /// Usable RAM after reserve.
    pub usable_ram_bytes: u64,
    /// Explanation lines for CLI / doctor output.
    pub reasons: Vec<String>,
}

/// Extensible probe surface for host + model capability checks.
pub trait CapabilityProbe: Send + Sync {
    /// Probe host memory / accelerators.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be measured.
    fn host_capabilities(&self, options: &FitOptions) -> Result<HostCapabilities, DripstoneError>;

    /// Estimate fit for a single catalog entry.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be measured.
    fn estimate_model(
        &self,
        spec: &ModelSpec,
        options: &FitOptions,
    ) -> Result<ModelFit, DripstoneError>;

    /// Estimate fit for every built-in catalog entry.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be measured.
    fn fit_catalog(&self, options: &FitOptions) -> Result<Vec<ModelFit>, DripstoneError> {
        let mut out = Vec::with_capacity(builtin_catalog().len());
        for spec in builtin_catalog() {
            out.push(self.estimate_model(spec, options)?);
        }
        Ok(out)
    }

    /// Estimate fit by catalog id.
    ///
    /// # Errors
    ///
    /// Returns an error when the id is unknown or the host cannot be measured.
    fn estimate_model_id(
        &self,
        id: &str,
        options: &FitOptions,
    ) -> Result<ModelFit, DripstoneError> {
        let spec = find_model(id).ok_or_else(|| {
            DripstoneError::configuration(format!("unknown local model catalog id '{id}'"))
        })?;
        self.estimate_model(spec, options)
    }
}

/// Default probe using OS memory sysctls / `nvidia-smi` and catalog hints.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultCapabilityProbe;

impl CapabilityProbe for DefaultCapabilityProbe {
    fn host_capabilities(&self, options: &FitOptions) -> Result<HostCapabilities, DripstoneError> {
        let memory = HostMemory::probe()?;
        let available_disk_bytes = options
            .disk_path
            .as_ref()
            .and_then(|path| crate::loader::disk::available_disk_bytes(path).ok());
        let recommended = options.device.resolve_auto();
        let mut notes = Vec::new();
        if let Some(reason) = recommended.reason.clone() {
            notes.push(reason);
        }
        if memory.available_vram_bytes.is_none() {
            if cfg!(target_os = "macos") {
                notes.push("unified memory: VRAM shares host RAM".into());
            } else {
                notes.push("VRAM probe unavailable (nvidia-smi not present or failed)".into());
            }
        }
        Ok(HostCapabilities {
            total_ram_bytes: memory.total_ram_bytes,
            available_ram_bytes: memory.available_ram_bytes,
            available_vram_bytes: memory.available_vram_bytes,
            available_disk_bytes,
            recommended_device: recommended.selected,
            notes,
        })
    }

    fn estimate_model(
        &self,
        spec: &ModelSpec,
        options: &FitOptions,
    ) -> Result<ModelFit, DripstoneError> {
        let host = HostMemory::probe()?;
        let reserve = options
            .reserve_ram_bytes
            .unwrap_or_else(reserve_ram_bytes_from_env);
        let budget = ResourceBudget::from_host_memory(&host, reserve);
        Ok(estimate_model_fit(spec, options, &budget))
    }
}

/// Pure fit evaluation against a provided budget (testable without OS probes).
#[must_use]
pub fn estimate_model_fit(
    spec: &ModelSpec,
    options: &FitOptions,
    budget: &ResourceBudget,
) -> ModelFit {
    let weight_bytes = resolve_weight_bytes(spec);
    let kv_bpt = spec
        .resources
        .kv_bytes_per_token
        .or_else(|| weight_bytes.map(default_kv_bytes_per_token));
    let recommended_n_ctx = options.n_ctx.unwrap_or_else(|| {
        spec.resources
            .recommended_n_ctx
            .unwrap_or_else(|| default_n_ctx_for_kind(spec.kind))
    });

    let mut reasons = Vec::new();
    let Some(weight_bytes) = weight_bytes else {
        reasons.push("weight size unknown; download or set catalog resource hints".into());
        return ModelFit {
            model_id: spec.id.to_string(),
            display_name: spec.display_name.to_string(),
            kind: spec.kind,
            status: FitStatus::Unknown,
            estimated_weight_bytes: 0,
            estimated_runtime_bytes: 0,
            recommended_n_ctx,
            max_safe_n_ctx: 0,
            usable_ram_bytes: budget.usable_ram_bytes(),
            reasons,
        };
    };

    let estimate = estimate_model_resources(spec, recommended_n_ctx)
        .unwrap_or_else(|| estimate_runtime_memory(weight_bytes, recommended_n_ctx, kv_bpt));
    if let Some(min_ram) = spec.resources.min_ram_bytes {
        reasons.push(format!("catalog minimum RAM hint: {min_ram} bytes"));
    }

    let max_ctx = max_safe_n_ctx(weight_bytes, budget, kv_bpt, 32_768);
    let usable = budget.usable_ram_bytes();
    let runtime = estimate.total_bytes();
    let status = classify_fit(runtime, usable, spec.resources.min_ram_bytes, budget);

    match status {
        FitStatus::FitsComfortably => reasons.push(format!(
            "fits comfortably at n_ctx={recommended_n_ctx} (~{runtime} of {usable} usable bytes)"
        )),
        FitStatus::FitsTight => reasons.push(format!(
            "tight fit at n_ctx={recommended_n_ctx} (~{runtime} of {usable} usable bytes); consider lower context"
        )),
        FitStatus::ExceedsMemory => reasons.push(format!(
            "exceeds memory at n_ctx={recommended_n_ctx} (~{runtime} needed, {usable} usable); max safe n_ctx≈{max_ctx}"
        )),
        FitStatus::Unknown => {}
    }

    if max_ctx > 0 && max_ctx < recommended_n_ctx {
        reasons.push(format!(
            "reduce context to <= {max_ctx} tokens to stay within reserve"
        ));
    }

    ModelFit {
        model_id: spec.id.to_string(),
        display_name: spec.display_name.to_string(),
        kind: spec.kind,
        status,
        estimated_weight_bytes: weight_bytes,
        estimated_runtime_bytes: runtime,
        recommended_n_ctx,
        max_safe_n_ctx: max_ctx,
        usable_ram_bytes: usable,
        reasons,
    }
}

/// Convenience: fit the full catalog with the default probe.
///
/// # Errors
///
/// Returns an error when the host cannot be probed.
pub fn fit_catalog(options: &FitOptions) -> Result<Vec<ModelFit>, DripstoneError> {
    DefaultCapabilityProbe.fit_catalog(options)
}

/// Convenience: probe host capabilities with the default probe.
///
/// # Errors
///
/// Returns an error when the host cannot be probed.
pub fn host_capabilities(options: &FitOptions) -> Result<HostCapabilities, DripstoneError> {
    DefaultCapabilityProbe.host_capabilities(options)
}

/// Build a [`ResourceEstimate`] for a catalog model at `n_ctx`.
#[must_use]
pub fn estimate_model_resources(spec: &ModelSpec, n_ctx: u32) -> Option<ResourceEstimate> {
    let weight_bytes = resolve_weight_bytes(spec)?;
    let kv_bpt = spec
        .resources
        .kv_bytes_per_token
        .or_else(|| Some(default_kv_bytes_per_token(weight_bytes)));
    Some(estimate_runtime_memory(weight_bytes, n_ctx, kv_bpt))
}

fn resolve_weight_bytes(spec: &ModelSpec) -> Option<u64> {
    if let Some(bytes) = spec.resources.weight_bytes {
        return Some(bytes);
    }
    let mut sum = 0_u64;
    let mut saw = false;
    for artifact in spec.artifacts {
        if let Some(bytes) = artifact.expected_bytes {
            sum = sum.saturating_add(bytes);
            saw = true;
        }
    }
    if saw { Some(sum) } else { None }
}

const fn default_n_ctx_for_kind(kind: ModelKind) -> u32 {
    match kind {
        ModelKind::Gguf => 8192,
        ModelKind::Candle | ModelKind::Ort => 2048,
        ModelKind::Embedding => 512,
    }
}

fn classify_fit(
    runtime_bytes: u64,
    usable_bytes: u64,
    min_ram_bytes: Option<u64>,
    budget: &ResourceBudget,
) -> FitStatus {
    if let Some(min_ram) = min_ram_bytes
        && budget.available_ram_bytes < min_ram
    {
        return FitStatus::ExceedsMemory;
    }
    if usable_bytes == 0 || runtime_bytes > usable_bytes {
        return FitStatus::ExceedsMemory;
    }
    // Tight when using 70%+ of usable budget.
    if runtime_bytes.saturating_mul(10) >= usable_bytes.saturating_mul(7) {
        FitStatus::FitsTight
    } else {
        FitStatus::FitsComfortably
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::resources::with_memory_overrides;

    fn smoke_spec(weight: u64) -> ModelSpec {
        ModelSpec {
            id: "test-smoke",
            display_name: "Test Smoke",
            repo: "example/test",
            revision: None,
            kind: ModelKind::Gguf,
            artifacts: &[],
            tags: &["gguf", "smoke"],
            resources: ModelResourceHint {
                weight_bytes: Some(weight),
                kv_bytes_per_token: Some(512),
                recommended_n_ctx: Some(4096),
                min_ram_bytes: None,
            },
        }
    }

    #[test]
    fn tiny_model_fits_comfortably_on_generous_budget() {
        let budget = ResourceBudget {
            reserve_ram_bytes: 512 * 1024 * 1024,
            available_ram_bytes: 16 * 1024 * 1024 * 1024,
            available_vram_bytes: None,
        };
        let fit = estimate_model_fit(&smoke_spec(20_000_000), &FitOptions::default(), &budget);
        assert_eq!(fit.status, FitStatus::FitsComfortably);
        assert!(fit.max_safe_n_ctx >= fit.recommended_n_ctx);
    }

    #[test]
    fn huge_model_exceeds_small_budget() {
        let budget = ResourceBudget {
            reserve_ram_bytes: 1_000_000_000,
            available_ram_bytes: 2_000_000_000,
            available_vram_bytes: None,
        };
        let fit = estimate_model_fit(&smoke_spec(8_000_000_000), &FitOptions::default(), &budget);
        assert_eq!(fit.status, FitStatus::ExceedsMemory);
        assert!(fit.max_safe_n_ctx < 4096);
    }

    #[test]
    fn builtin_catalog_fit_runs_with_probe_override() {
        with_memory_overrides(
            Some(32 * 1024 * 1024 * 1024),
            Some(24 * 1024 * 1024 * 1024),
            Some(None),
            || {
                let fits = fit_catalog(&FitOptions::default()).expect("fit catalog");
                assert!(fits.iter().any(|f| f.model_id == "stories15m-gguf"));
                let stories = fits
                    .iter()
                    .find(|f| f.model_id == "stories15m-gguf")
                    .expect("stories");
                assert_ne!(stories.status, FitStatus::ExceedsMemory);
            },
        );
    }
}
