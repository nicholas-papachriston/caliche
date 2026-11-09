//! Local inference host/catalog fit probes for the Python SDK.

use caliche::llm::{FitOptions, FitStatus, HostCapabilities, LocalInferenceProvider, ModelFit};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn fit_status_name(status: FitStatus) -> &'static str {
    match status {
        FitStatus::FitsComfortably => "fits_comfortably",
        FitStatus::FitsTight => "fits_tight",
        FitStatus::ExceedsMemory => "exceeds_memory",
        FitStatus::Unknown => "unknown",
    }
}

fn host_to_dict(py: Python<'_>, host: HostCapabilities) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("total_ram_bytes", host.total_ram_bytes)?;
    dict.set_item("available_ram_bytes", host.available_ram_bytes)?;
    dict.set_item("available_vram_bytes", host.available_vram_bytes)?;
    dict.set_item("available_disk_bytes", host.available_disk_bytes)?;
    dict.set_item(
        "recommended_device",
        format!("{:?}", host.recommended_device),
    )?;
    dict.set_item("notes", host.notes)?;
    Ok(dict.into())
}

fn fit_to_dict(py: Python<'_>, fit: ModelFit) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("model_id", fit.model_id)?;
    dict.set_item("display_name", fit.display_name)?;
    dict.set_item("kind", format!("{:?}", fit.kind).to_ascii_lowercase())?;
    dict.set_item("status", fit_status_name(fit.status))?;
    dict.set_item("estimated_weight_bytes", fit.estimated_weight_bytes)?;
    dict.set_item("estimated_runtime_bytes", fit.estimated_runtime_bytes)?;
    dict.set_item("recommended_n_ctx", fit.recommended_n_ctx)?;
    dict.set_item("max_safe_n_ctx", fit.max_safe_n_ctx)?;
    dict.set_item("usable_ram_bytes", fit.usable_ram_bytes)?;
    dict.set_item("reasons", fit.reasons)?;
    Ok(dict.into())
}

/// Probe host RAM/VRAM for local inference planning.
#[pyfunction]
pub fn probe_local_inference_host(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let host = LocalInferenceProvider::probe_host_capabilities(FitOptions::default())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    host_to_dict(py, host)
}

/// Estimate fit for every built-in local inference catalog model.
#[pyfunction]
pub fn fit_local_inference_catalog(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    let fits = LocalInferenceProvider::fit_catalog_models(FitOptions::default())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    fits.into_iter().map(|fit| fit_to_dict(py, fit)).collect()
}

/// Estimate fit for one catalog id.
#[pyfunction]
pub fn fit_local_inference_model(py: Python<'_>, id: &str) -> PyResult<Py<PyDict>> {
    let fit = LocalInferenceProvider::fit_catalog_model(id, FitOptions::default())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    fit_to_dict(py, fit)
}

/// Catalog models that fit comfortably or tightly on this host.
#[pyfunction]
pub fn safe_local_inference_models(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    let fits = LocalInferenceProvider::safe_catalog_models(FitOptions::default())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    fits.into_iter().map(|fit| fit_to_dict(py, fit)).collect()
}
