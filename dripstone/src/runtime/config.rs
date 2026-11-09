//! Shared runtime/device configuration for inference and embeddings.

use serde::{Deserialize, Serialize};

/// Unified runtime/device intent for GPU or CPU execution.
///
/// Used by both inference and embedding pipelines. Backends map this to
/// backend-specific device handles (e.g. Candle `Device`, ORT execution providers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RuntimeConfig {
    /// Auto-detect best available device (MLX on Apple when enabled, then CUDA, Metal, CPU).
    #[default]
    Auto,
    /// MLX runtime (Apple Silicon only, feature-gated).
    Mlx,
    /// CUDA GPU (NVIDIA).
    Cuda {
        /// GPU device ID (0 for first GPU).
        device_id: usize,
    },
    /// Metal GPU (Apple Silicon, via Candle or ORT `CoreML`).
    Metal,
    /// CPU only.
    Cpu,
}

impl RuntimeConfig {
    /// Resolve `Auto` to a concrete runtime preference based on platform and features.
    ///
    /// This is compile-time / `cfg` selection only — it does **not** measure host RAM or
    /// VRAM. For memory-aware “can I load this model?” checks, use
    /// [`crate::loader::CapabilityProbe`] / [`crate::loader::fit_catalog`] (or the
    /// Caliche `LocalInferenceProvider` wrappers). Backends still validate device
    /// availability when creating sessions.
    #[must_use]
    pub fn resolve_auto(self) -> ResolvedRuntime {
        if self != Self::Auto {
            return ResolvedRuntime {
                selected: self,
                available: true,
                reason: None,
            };
        }

        let (selected, reason) = Self::resolve_auto_inner();
        ResolvedRuntime {
            selected,
            available: true,
            reason: Some(reason),
        }
    }

    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn resolve_auto_inner() -> (Self, String) {
        if cfg!(all(target_vendor = "apple", feature = "local-mlx")) {
            return (
                Self::Mlx,
                "auto: MLX preferred on Apple (local-mlx enabled)".to_string(),
            );
        }
        if cfg!(target_vendor = "apple") {
            return (Self::Metal, "auto: Metal preferred on Apple".to_string());
        }
        if cfg!(any(feature = "inference-candle", feature = "embedding-ort")) {
            return (
                Self::Cuda { device_id: 0 },
                "auto: CUDA preferred on non-Apple (backend will validate)".to_string(),
            );
        }
        (Self::Cpu, "auto: CPU (no GPU backend enabled)".to_string())
    }
}

/// Result of resolving a runtime: selected backend and optional availability/reason.
#[derive(Debug, Clone)]
pub struct ResolvedRuntime {
    /// The runtime to use.
    pub selected: RuntimeConfig,
    /// Whether the runtime is expected to be available (backends may still fail at init).
    pub available: bool,
    /// Human-readable reason for selection (e.g. for logging).
    pub reason: Option<String>,
}
