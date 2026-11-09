//! Host memory / VRAM probes and runtime resource budgets.
//!
//! Disk preflight lives in [`crate::loader::disk`]; this module covers RAM and
//! best-effort accelerator memory so loaders can refuse oversized models before
//! the process is OOM-killed.

use crate::errors::DripstoneError;
use std::process::Command;

/// Optional resource metadata attached to a catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelResourceHint {
    /// Known weight footprint in bytes (sum of primary artifacts when known).
    pub weight_bytes: Option<u64>,
    /// Estimated KV-cache bytes consumed per context token.
    pub kv_bytes_per_token: Option<u64>,
    /// Catalog-recommended context length for interactive use.
    pub recommended_n_ctx: Option<u32>,
    /// Explicit minimum RAM required (overrides heuristic when set).
    pub min_ram_bytes: Option<u64>,
}

/// Default RAM headroom that must remain after loading a model (1.5 GiB).
pub const DEFAULT_RESERVE_RAM_BYTES: u64 = (3 * 1024 * 1024 * 1024) / 2;

/// Environment override for the RAM reserve (`DRIPSTONE_RESERVE_RAM_BYTES`).
pub const RESERVE_RAM_ENV: &str = "DRIPSTONE_RESERVE_RAM_BYTES";

#[cfg(test)]
thread_local! {
    static AVAILABLE_RAM_OVERRIDE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static TOTAL_RAM_OVERRIDE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static VRAM_OVERRIDE: std::cell::Cell<Option<Option<u64>>> = const { std::cell::Cell::new(None) };
}

/// Snapshot of host memory capacity relevant to local inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// Physical RAM size in bytes.
    pub total_ram_bytes: u64,
    /// Estimated currently available RAM in bytes.
    pub available_ram_bytes: u64,
    /// Best-effort free VRAM when an accelerator probe succeeds.
    pub available_vram_bytes: Option<u64>,
}

impl HostMemory {
    /// Probe the current host. Results are not cached; callers that need a stable
    /// view for a CLI session should hold the returned value.
    ///
    /// # Errors
    ///
    /// Returns an error when total/available RAM cannot be measured.
    pub fn probe() -> Result<Self, DripstoneError> {
        #[cfg(test)]
        if let (Some(total), Some(available)) =
            (TOTAL_RAM_OVERRIDE.get(), AVAILABLE_RAM_OVERRIDE.get())
        {
            let vram = VRAM_OVERRIDE.get().unwrap_or_else(probe_vram_bytes_os);
            return Ok(Self {
                total_ram_bytes: total,
                available_ram_bytes: available,
                available_vram_bytes: vram,
            });
        }

        let (total_ram_bytes, available_ram_bytes) = probe_ram_bytes_os()?;
        let available_vram_bytes = probe_vram_bytes_os();
        Ok(Self {
            total_ram_bytes,
            available_ram_bytes,
            available_vram_bytes,
        })
    }
}

/// Budget used by load-time / context-growth preflight checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBudget {
    /// Bytes that must remain free after the allocation.
    pub reserve_ram_bytes: u64,
    /// Available RAM at budget construction time.
    pub available_ram_bytes: u64,
    /// Optional VRAM ceiling (unified-memory hosts leave this `None`).
    pub available_vram_bytes: Option<u64>,
}

impl ResourceBudget {
    /// Build a budget from a live host probe and optional env reserve override.
    ///
    /// # Errors
    ///
    /// Returns an error when the host cannot be probed.
    pub fn from_host() -> Result<Self, DripstoneError> {
        let host = HostMemory::probe()?;
        Ok(Self::from_host_memory(&host, reserve_ram_bytes_from_env()))
    }

    /// Construct a budget from an existing memory snapshot.
    #[must_use]
    pub const fn from_host_memory(host: &HostMemory, reserve_ram_bytes: u64) -> Self {
        Self {
            reserve_ram_bytes,
            available_ram_bytes: host.available_ram_bytes,
            available_vram_bytes: host.available_vram_bytes,
        }
    }

    /// Bytes that may be consumed by weights + KV + runtime overhead.
    #[must_use]
    pub fn usable_ram_bytes(&self) -> u64 {
        self.available_ram_bytes
            .saturating_sub(self.reserve_ram_bytes)
    }
}

/// Estimated memory needed to run a model at a given context size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceEstimate {
    /// On-disk / mmap weight footprint.
    pub weight_bytes: u64,
    /// Estimated KV-cache bytes for the target context.
    pub kv_cache_bytes: u64,
    /// Extra allocator / runtime overhead.
    pub overhead_bytes: u64,
    /// Context length the estimate was computed for.
    pub n_ctx: u32,
}

impl ResourceEstimate {
    /// Total bytes expected to be resident while the model is loaded.
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.weight_bytes
            .saturating_add(self.kv_cache_bytes)
            .saturating_add(self.overhead_bytes)
    }
}

/// Ensure `estimate` fits inside `budget`.
///
/// # Errors
///
/// Returns an error when the estimate would leave less than the configured reserve.
pub fn ensure_resource_budget(
    estimate: &ResourceEstimate,
    budget: &ResourceBudget,
) -> Result<(), DripstoneError> {
    let needed = estimate.total_bytes();
    let usable = budget.usable_ram_bytes();
    if needed > usable {
        return Err(DripstoneError::configuration(format!(
            "insufficient memory for local inference: need ~{needed} bytes \
             (weights {} + kv {} + overhead {}) at n_ctx={}, \
             have {} usable ({} available − {} reserve)",
            estimate.weight_bytes,
            estimate.kv_cache_bytes,
            estimate.overhead_bytes,
            estimate.n_ctx,
            usable,
            budget.available_ram_bytes,
            budget.reserve_ram_bytes
        )));
    }

    if let Some(vram) = budget.available_vram_bytes {
        // Discrete GPUs: refuse when weights alone exceed free VRAM. KV may stay in host RAM.
        if estimate.weight_bytes > vram {
            return Err(DripstoneError::configuration(format!(
                "insufficient VRAM for local inference weights: need ~{} bytes, have {vram} free",
                estimate.weight_bytes
            )));
        }
    }
    Ok(())
}

/// Resolve the RAM reserve, honoring [`RESERVE_RAM_ENV`] when set.
#[must_use]
pub fn reserve_ram_bytes_from_env() -> u64 {
    std::env::var(RESERVE_RAM_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_RESERVE_RAM_BYTES)
}

/// Heuristic KV bytes per token when the catalog does not supply a hint.
///
/// Scales gently with weight size so tiny smoke models stay cheap while larger
/// GGUF/Candle checkpoints reserve more headroom per context token.
#[must_use]
pub fn default_kv_bytes_per_token(weight_bytes: u64) -> u64 {
    const MIN: u64 = 256;
    const MAX: u64 = 16 * 1024;
    // ~0.5 bytes of KV budget per weight KiB, clamped.
    let scaled = (weight_bytes / 2048).clamp(MIN, MAX);
    scaled
}

/// Estimate runtime memory for weights + KV at `n_ctx`.
#[must_use]
pub fn estimate_runtime_memory(
    weight_bytes: u64,
    n_ctx: u32,
    kv_bytes_per_token: Option<u64>,
) -> ResourceEstimate {
    let per_token = kv_bytes_per_token.unwrap_or_else(|| default_kv_bytes_per_token(weight_bytes));
    let kv_cache_bytes = u64::from(n_ctx).saturating_mul(per_token);
    // Scratch tensors, sampler state, fragmentation.
    let overhead_bytes = (weight_bytes / 20).clamp(32 * 1024 * 1024, 512 * 1024 * 1024);
    ResourceEstimate {
        weight_bytes,
        kv_cache_bytes,
        overhead_bytes,
        n_ctx,
    }
}

/// Largest `n_ctx` that still fits in `budget` for the given weight footprint.
#[must_use]
pub fn max_safe_n_ctx(
    weight_bytes: u64,
    budget: &ResourceBudget,
    kv_bytes_per_token: Option<u64>,
    hard_max: u32,
) -> u32 {
    let per_token = kv_bytes_per_token
        .unwrap_or_else(|| default_kv_bytes_per_token(weight_bytes))
        .max(1);
    let base = estimate_runtime_memory(weight_bytes, 0, Some(per_token));
    let fixed = base.weight_bytes.saturating_add(base.overhead_bytes);
    let usable = budget.usable_ram_bytes();
    if usable <= fixed {
        return 0;
    }
    let from_ram = ((usable - fixed) / per_token).min(u64::from(hard_max));
    u32::try_from(from_ram).unwrap_or(hard_max)
}

fn probe_ram_bytes_os() -> Result<(u64, u64), DripstoneError> {
    #[cfg(target_os = "linux")]
    {
        return probe_ram_linux();
    }
    #[cfg(target_os = "macos")]
    {
        return probe_ram_macos();
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Err(DripstoneError::storage(
            "RAM probe is not implemented on this platform",
        ))
    }
}

#[cfg(target_os = "linux")]
fn probe_ram_linux() -> Result<(u64, u64), DripstoneError> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|error| {
        DripstoneError::storage_with_source("failed to read /proc/meminfo", error)
    })?;
    let total = parse_meminfo_kib(&text, "MemTotal")
        .ok_or_else(|| DripstoneError::storage("MemTotal missing from /proc/meminfo"))?;
    let available = parse_meminfo_kib(&text, "MemAvailable")
        .or_else(|| parse_meminfo_kib(&text, "MemFree"))
        .ok_or_else(|| {
            DripstoneError::storage("MemAvailable/MemFree missing from /proc/meminfo")
        })?;
    Ok((total.saturating_mul(1024), available.saturating_mul(1024)))
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(text: &str, key: &str) -> Option<u64> {
    let prefix = format!("{key}:");
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&prefix) {
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kib);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn probe_ram_macos() -> Result<(u64, u64), DripstoneError> {
    let total = sysctl_u64(c"hw.memsize")
        .map_err(|error| DripstoneError::storage(format!("sysctl hw.memsize failed: {error}")))?;
    let page_size = sysctl_u64(c"hw.pagesize").unwrap_or(4096);
    let free = sysctl_u64(c"vm.page_free_count").unwrap_or(0);
    let speculative = sysctl_u64(c"vm.page_speculative_count").unwrap_or(0);
    let purgeable = sysctl_u64(c"vm.page_purgeable_count").unwrap_or(0);
    let inactive = sysctl_u64(c"vm.page_inactive_count").unwrap_or(0);
    let available_pages = free
        .saturating_add(speculative)
        .saturating_add(purgeable)
        .saturating_add(inactive / 2);
    let available = available_pages.saturating_mul(page_size).min(total);
    // If vm.* sysctls are unavailable, keep a conservative fraction of total.
    let available = if available == 0 { total / 2 } else { available };
    Ok((total, available))
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &std::ffi::CStr) -> Result<u64, String> {
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: name is a valid C string; value/size point at a stack u64.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast::<libc::c_void>(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(value)
}

fn probe_vram_bytes_os() -> Option<u64> {
    #[cfg(test)]
    if let Some(overridden) = VRAM_OVERRIDE.get() {
        return overridden;
    }

    // Apple Silicon uses unified memory; discrete VRAM probes do not apply.
    if cfg!(target_os = "macos") {
        return None;
    }

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut total_mib = 0_u64;
    let mut saw_any = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(mib) = trimmed.parse::<u64>() {
            total_mib = total_mib.saturating_add(mib);
            saw_any = true;
        }
    }
    if saw_any {
        Some(total_mib.saturating_mul(1024 * 1024))
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) fn with_memory_overrides<T>(
    total_ram: Option<u64>,
    available_ram: Option<u64>,
    vram: Option<Option<u64>>,
    f: impl FnOnce() -> T,
) -> T {
    TOTAL_RAM_OVERRIDE.with(|total_cell| {
        AVAILABLE_RAM_OVERRIDE.with(|available_cell| {
            VRAM_OVERRIDE.with(|vram_cell| {
                let prev_total = total_cell.replace(total_ram);
                let prev_available = available_cell.replace(available_ram);
                let prev_vram = vram_cell.replace(vram);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
                total_cell.set(prev_total);
                available_cell.set(prev_available);
                vram_cell.set(prev_vram);
                match result {
                    Ok(value) => value,
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_grows_with_context() {
        let small = estimate_runtime_memory(20_000_000, 2048, Some(512));
        let large = estimate_runtime_memory(20_000_000, 8192, Some(512));
        assert!(large.total_bytes() > small.total_bytes());
        assert_eq!(large.kv_cache_bytes, 8192 * 512);
    }

    #[test]
    fn ensure_budget_rejects_when_too_small() {
        let estimate = estimate_runtime_memory(8_000_000_000, 8192, Some(1024));
        let budget = ResourceBudget {
            reserve_ram_bytes: 1_000_000_000,
            available_ram_bytes: 2_000_000_000,
            available_vram_bytes: None,
        };
        let err = ensure_resource_budget(&estimate, &budget).expect_err("should fail");
        assert!(err.to_string().contains("insufficient memory"));
    }

    #[test]
    fn usable_ram_saturates_to_zero_when_reserve_exceeds_available() {
        let budget = ResourceBudget {
            reserve_ram_bytes: 8_000_000_000,
            available_ram_bytes: 1_000_000_000,
            available_vram_bytes: None,
        };
        assert_eq!(budget.usable_ram_bytes(), 0);
    }

    #[test]
    fn ensure_budget_rejects_insufficient_vram() {
        let estimate = estimate_runtime_memory(4_000_000_000, 2048, Some(512));
        let budget = ResourceBudget {
            reserve_ram_bytes: 512 * 1024 * 1024,
            available_ram_bytes: 32 * 1024 * 1024 * 1024,
            available_vram_bytes: Some(1_000_000_000),
        };
        let err = ensure_resource_budget(&estimate, &budget).expect_err("vram should fail");
        assert!(err.to_string().to_ascii_lowercase().contains("vram"));
    }

    #[test]
    #[serial_test::serial(env)]
    fn reserve_ram_from_env_ignores_invalid_and_honors_zero() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set(RESERVE_RAM_ENV, "not-a-number");
        assert_eq!(reserve_ram_bytes_from_env(), DEFAULT_RESERVE_RAM_BYTES);
        env.set(RESERVE_RAM_ENV, "0");
        assert_eq!(reserve_ram_bytes_from_env(), 0);
        env.remove(RESERVE_RAM_ENV);
        assert_eq!(reserve_ram_bytes_from_env(), DEFAULT_RESERVE_RAM_BYTES);
    }

    #[test]
    fn max_safe_n_ctx_respects_budget() {
        let budget = ResourceBudget {
            reserve_ram_bytes: 512 * 1024 * 1024,
            available_ram_bytes: 4 * 1024 * 1024 * 1024,
            available_vram_bytes: None,
        };
        let n_ctx = max_safe_n_ctx(512 * 1024 * 1024, &budget, Some(1024), 32_768);
        assert!(n_ctx > 0);
        assert!(n_ctx <= 32_768);
        let estimate = estimate_runtime_memory(512 * 1024 * 1024, n_ctx, Some(1024));
        ensure_resource_budget(&estimate, &budget).expect("max safe should fit");
    }

    #[test]
    fn host_probe_returns_positive_on_this_machine() {
        let host = HostMemory::probe().expect("probe ram");
        assert!(host.total_ram_bytes > 0);
        assert!(host.available_ram_bytes > 0);
        assert!(host.available_ram_bytes <= host.total_ram_bytes);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_meminfo_reads_keys() {
        let sample = "\
MemTotal:       16384000 kB
MemFree:         2048000 kB
MemAvailable:    8192000 kB
";
        assert_eq!(parse_meminfo_kib(sample, "MemTotal"), Some(16_384_000));
        assert_eq!(parse_meminfo_kib(sample, "MemAvailable"), Some(8_192_000));
    }
}
