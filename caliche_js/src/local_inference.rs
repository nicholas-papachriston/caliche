//! Local inference host/catalog fit probes for the JavaScript SDK.

#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use caliche::llm::{FitOptions, FitStatus, HostCapabilities, LocalInferenceProvider, ModelFit};
use napi_derive::napi;

/// Host memory snapshot for local inference planning.
#[napi(object)]
pub struct JsHostCapabilities {
    pub total_ram_bytes: f64,
    pub available_ram_bytes: f64,
    pub available_vram_bytes: Option<f64>,
    pub available_disk_bytes: Option<f64>,
    pub recommended_device: String,
    pub notes: Vec<String>,
}

/// Fit result for one catalog model.
#[napi(object)]
pub struct JsModelFit {
    pub model_id: String,
    pub display_name: String,
    pub kind: String,
    pub status: String,
    pub estimated_weight_bytes: f64,
    pub estimated_runtime_bytes: f64,
    pub recommended_n_ctx: u32,
    pub max_safe_n_ctx: u32,
    pub usable_ram_bytes: f64,
    pub reasons: Vec<String>,
}

fn u64_to_js(value: u64, field: &str) -> napi::Result<f64> {
    const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
    if value > MAX_SAFE_INTEGER {
        return Err(napi::Error::from_reason(format!(
            "{field} exceeds JavaScript's maximum safe integer"
        )));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "safe-integer bound makes this conversion exact"
    )]
    Ok(value as f64)
}

fn fit_status_name(status: FitStatus) -> &'static str {
    match status {
        FitStatus::FitsComfortably => "fits_comfortably",
        FitStatus::FitsTight => "fits_tight",
        FitStatus::ExceedsMemory => "exceeds_memory",
        FitStatus::Unknown => "unknown",
    }
}

fn host_to_js(host: HostCapabilities) -> napi::Result<JsHostCapabilities> {
    Ok(JsHostCapabilities {
        total_ram_bytes: u64_to_js(host.total_ram_bytes, "total_ram_bytes")?,
        available_ram_bytes: u64_to_js(host.available_ram_bytes, "available_ram_bytes")?,
        available_vram_bytes: host
            .available_vram_bytes
            .map(|bytes| u64_to_js(bytes, "available_vram_bytes"))
            .transpose()?,
        available_disk_bytes: host
            .available_disk_bytes
            .map(|bytes| u64_to_js(bytes, "available_disk_bytes"))
            .transpose()?,
        recommended_device: format!("{:?}", host.recommended_device),
        notes: host.notes,
    })
}

fn fit_to_js(fit: ModelFit) -> napi::Result<JsModelFit> {
    Ok(JsModelFit {
        model_id: fit.model_id,
        display_name: fit.display_name,
        kind: format!("{:?}", fit.kind).to_ascii_lowercase(),
        status: fit_status_name(fit.status).to_string(),
        estimated_weight_bytes: u64_to_js(fit.estimated_weight_bytes, "estimated_weight_bytes")?,
        estimated_runtime_bytes: u64_to_js(fit.estimated_runtime_bytes, "estimated_runtime_bytes")?,
        recommended_n_ctx: fit.recommended_n_ctx,
        max_safe_n_ctx: fit.max_safe_n_ctx,
        usable_ram_bytes: u64_to_js(fit.usable_ram_bytes, "usable_ram_bytes")?,
        reasons: fit.reasons,
    })
}

/// Probe host RAM/VRAM for local inference planning.
#[napi(catch_unwind)]
pub fn probe_local_inference_host() -> napi::Result<JsHostCapabilities> {
    let host = LocalInferenceProvider::probe_host_capabilities(FitOptions::default())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    host_to_js(host)
}

/// Estimate fit for every built-in local inference catalog model.
#[napi(catch_unwind)]
pub fn fit_local_inference_catalog() -> napi::Result<Vec<JsModelFit>> {
    let fits = LocalInferenceProvider::fit_catalog_models(FitOptions::default())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    fits.into_iter().map(fit_to_js).collect()
}

/// Estimate fit for one catalog id.
#[napi(catch_unwind)]
pub fn fit_local_inference_model(id: String) -> napi::Result<JsModelFit> {
    let fit = LocalInferenceProvider::fit_catalog_model(&id, FitOptions::default())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    fit_to_js(fit)
}

/// Catalog models that fit comfortably or tightly on this host.
#[napi(catch_unwind)]
pub fn safe_local_inference_models() -> napi::Result<Vec<JsModelFit>> {
    let fits = LocalInferenceProvider::safe_catalog_models(FitOptions::default())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    fits.into_iter().map(fit_to_js).collect()
}
