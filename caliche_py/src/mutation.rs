//! Python bindings for file mutation style (`fs` write/edit vs `apply_patch`).

use caliche::agent::tools::MutationStyle;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// How file mutations are exposed to the model.
///
/// `EditWrite` and `ApplyPatch` are mutually exclusive.
#[pyclass(name = "MutationStyle")]
#[derive(Clone, Copy)]
pub struct PyMutationStyle {
    pub(crate) inner: MutationStyle,
}

#[pymethods]
impl PyMutationStyle {
    #[new]
    fn new(style: &str) -> PyResult<Self> {
        let inner = match style.to_lowercase().as_str() {
            "editwrite" | "edit_write" | "edit-write" => MutationStyle::EditWrite,
            "applypatch" | "apply_patch" | "apply-patch" => MutationStyle::ApplyPatch,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Invalid mutation style: {style}. Must be one of: edit_write, apply_patch"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    const fn edit_write() -> Self {
        Self {
            inner: MutationStyle::EditWrite,
        }
    }

    #[staticmethod]
    const fn apply_patch() -> Self {
        Self {
            inner: MutationStyle::ApplyPatch,
        }
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "PyO3 protocol methods require borrowed Python instance receivers"
    )]
    const fn __str__(&self) -> &'static str {
        match self.inner {
            MutationStyle::EditWrite => "EditWrite",
            MutationStyle::ApplyPatch => "ApplyPatch",
        }
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "PyO3 protocol methods require borrowed Python instance receivers"
    )]
    fn __repr__(&self) -> String {
        format!("MutationStyle.{}", self.__str__())
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "PyO3 comparison protocol requires borrowed Python operands"
    )]
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl From<PyMutationStyle> for MutationStyle {
    fn from(value: PyMutationStyle) -> Self {
        value.inner
    }
}

impl From<MutationStyle> for PyMutationStyle {
    fn from(inner: MutationStyle) -> Self {
        Self { inner }
    }
}
