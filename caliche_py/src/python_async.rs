//! Shared utilities for handling Python async coroutines from Rust.
//!
//! Prefer bridging into the active asyncio loop (via pyo3-async-runtimes /
//! `TaskLocals`). Fall back to a worker-thread `asyncio.run` only when no loop
//! context is available.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use std::time::Duration;
use tracing::debug;

async fn run_coroutine_on_worker(py_result: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let py_result_clone = Python::attach(|py| py_result.clone_ref(py));
    let join = tokio::task::spawn_blocking(move || {
        Python::attach(|py| -> PyResult<Py<PyAny>> {
            let asyncio = py.import("asyncio")?;
            let run = asyncio.getattr("run")?;
            run.call1((py_result_clone.bind(py),))
                .map(Bound::unbind)
                .map_err(|error| {
                    let cancelled = (|| -> PyResult<bool> {
                        let builtins = py.import("builtins")?;
                        let isinstance = builtins.getattr("isinstance")?;
                        let keyboard_interrupt = builtins.getattr("KeyboardInterrupt")?;
                        let cancelled_error = asyncio.getattr("CancelledError")?;
                        let is_keyboard_interrupt: bool = isinstance
                            .call1((error.value(py), keyboard_interrupt))?
                            .extract()?;
                        let is_cancelled: bool = isinstance
                            .call1((error.value(py), cancelled_error))?
                            .extract()?;
                        Ok(is_keyboard_interrupt || is_cancelled)
                    })()
                    .unwrap_or(false);
                    if cancelled {
                        PyRuntimeError::new_err("Operation cancelled by user")
                    } else {
                        error
                    }
                })
        })
    });

    match tokio::time::timeout(Duration::from_secs(300), join).await {
        Ok(Ok(result)) => result,
        Ok(Err(join_err)) => Err(PyRuntimeError::new_err(format!(
            "Python coroutine worker failed: {join_err}"
        ))),
        Err(_) => Err(PyRuntimeError::new_err(
            "Python coroutine timed out after 5 minutes",
        )),
    }
}

/// Await a Python value that may be a coroutine.
///
/// Sync values are returned unchanged. Coroutines are awaited via:
/// 1. `pyo3-async-runtimes` (running loop or `TaskLocals`), or
/// 2. a worker-thread `asyncio.run` fallback.
pub async fn call_python_maybe_async(py_result: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let is_coroutine = Python::attach(|py| -> PyResult<bool> {
        let asyncio = py.import("asyncio")?;
        let iscoroutine = asyncio.getattr("iscoroutine")?;
        iscoroutine.call1((py_result.bind(py),))?.extract()
    })?;

    if !is_coroutine {
        return Ok(py_result);
    }

    debug!("awaiting Python coroutine from Rust");

    // Try bridging into asyncio (TaskLocals from future_into_py, or a running loop).
    let bridged =
        Python::attach(|py| pyo3_async_runtimes::tokio::into_future(py_result.bind(py).clone()));

    let Ok(fut) = bridged else {
        return run_coroutine_on_worker(py_result).await;
    };
    tokio::time::timeout(Duration::from_secs(300), fut)
        .await
        .map_or_else(
            |_| {
                Err(PyRuntimeError::new_err(
                    "Python coroutine timed out after 5 minutes",
                ))
            },
            std::convert::identity,
        )
}

/// Invoke a Python result, skipping coroutine detection when `is_async` is false.
pub async fn call_python_result(py_result: Py<PyAny>, is_async: bool) -> PyResult<Py<PyAny>> {
    if is_async {
        call_python_maybe_async(py_result).await
    } else {
        Ok(py_result)
    }
}
