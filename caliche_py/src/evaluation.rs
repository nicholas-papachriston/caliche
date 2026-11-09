//! Python bindings for Caliche Evaluation Framework

use crate::conversions::{json_to_python_value, py_dict_to_hashmap};
use crate::get_runtime;
use caliche::evaluation::{
    ContainsEvaluator, EvaluationDataset, Evaluator, ExactMatchEvaluator, LLMEvaluator,
};
use caliche::llm::ProviderType;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Evaluators
// ============================================================================

/// Python wrapper for `ExactMatchEvaluator`
#[pyclass]
pub struct PyExactMatchEvaluator {
    inner: Arc<dyn Evaluator>,
}

#[pymethods]
impl PyExactMatchEvaluator {
    #[new]
    #[pyo3(signature = (case_sensitive = true))]
    pub fn new(case_sensitive: bool) -> Self {
        Self {
            inner: Arc::new(ExactMatchEvaluator::new(case_sensitive)) as Arc<dyn Evaluator>,
        }
    }

    /// Evaluate an output against expected output
    #[pyo3(signature = (output, expected, context=None))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        output: &str,
        expected: &str,
        context: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let evaluator_clone = self.inner.clone();
        let rt = get_runtime()?;

        let ctx = context.map(|d| py_dict_to_hashmap(d)).transpose()?;

        let result = rt.block_on(async {
            evaluator_clone
                .evaluate(
                    output,
                    expected,
                    ctx.as_ref()
                        .map(|h| h as &HashMap<String, serde_json::Value>),
                )
                .await
        });

        match result {
            Ok(eval_result) => {
                let dict = PyDict::new(py);
                dict.set_item("score", eval_result.score)?;
                dict.set_item("passed", eval_result.passed)?;
                dict.set_item(
                    "details",
                    json_to_python_value(py, &serde_json::json!(eval_result.details))?,
                )?;
                if let Some(error) = eval_result.error {
                    dict.set_item("error", error)?;
                }
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Evaluation failed: {e}"))),
        }
    }

    /// Get evaluator name
    pub fn name(&self) -> &str {
        self.inner.name()
    }
}

/// Python wrapper for `ContainsEvaluator`
#[pyclass]
pub struct PyContainsEvaluator {
    inner: Arc<dyn Evaluator>,
}

#[pymethods]
impl PyContainsEvaluator {
    #[new]
    #[pyo3(signature = (case_sensitive = false))]
    pub fn new(case_sensitive: bool) -> Self {
        Self {
            inner: Arc::new(ContainsEvaluator::new(case_sensitive)) as Arc<dyn Evaluator>,
        }
    }
    /// Evaluate an output against expected output
    #[pyo3(signature = (output, expected, context=None))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        output: &str,
        expected: &str,
        context: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let evaluator_clone = self.inner.clone();
        let rt = get_runtime()?;

        let ctx = context.map(|d| py_dict_to_hashmap(d)).transpose()?;

        let result = rt.block_on(async {
            evaluator_clone
                .evaluate(
                    output,
                    expected,
                    ctx.as_ref()
                        .map(|h| h as &HashMap<String, serde_json::Value>),
                )
                .await
        });

        match result {
            Ok(eval_result) => {
                let dict = PyDict::new(py);
                dict.set_item("score", eval_result.score)?;
                dict.set_item("passed", eval_result.passed)?;
                dict.set_item(
                    "details",
                    json_to_python_value(py, &serde_json::json!(eval_result.details))?,
                )?;
                if let Some(error) = eval_result.error {
                    dict.set_item("error", error)?;
                }
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Evaluation failed: {e}"))),
        }
    }

    /// Get evaluator name
    pub fn name(&self) -> &str {
        self.inner.name()
    }
}

/// Python wrapper for `LLMEvaluator`
#[pyclass]
pub struct PyLLMEvaluator {
    inner: Arc<dyn Evaluator>,
}

#[pymethods]
impl PyLLMEvaluator {
    #[new]
    #[pyo3(signature = (criteria, provider = None, model = None))]
    pub fn new(
        criteria: String,
        provider: Option<String>,
        model: Option<String>,
    ) -> PyResult<Self> {
        let mut evaluator = LLMEvaluator::new(criteria);
        if let Some(provider) = provider {
            let provider = provider
                .parse::<ProviderType>()
                .map_err(PyValueError::new_err)?;
            evaluator = evaluator.with_provider(provider);
        }
        if let Some(model) = model {
            evaluator = evaluator.with_model(model);
        }

        Ok(Self {
            inner: Arc::new(evaluator) as Arc<dyn Evaluator>,
        })
    }

    /// Evaluate an output against expected output
    #[pyo3(signature = (output, expected, context=None))]
    pub fn evaluate(
        &self,
        py: Python<'_>,
        output: &str,
        expected: &str,
        context: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let evaluator_clone = self.inner.clone();
        let rt = get_runtime()?;

        let ctx = context.map(|d| py_dict_to_hashmap(d)).transpose()?;

        let result = rt.block_on(async {
            evaluator_clone
                .evaluate(
                    output,
                    expected,
                    ctx.as_ref()
                        .map(|h| h as &HashMap<String, serde_json::Value>),
                )
                .await
        });

        match result {
            Ok(eval_result) => {
                let dict = PyDict::new(py);
                dict.set_item("score", eval_result.score)?;
                dict.set_item("passed", eval_result.passed)?;
                dict.set_item(
                    "details",
                    json_to_python_value(py, &serde_json::json!(eval_result.details))?,
                )?;
                if let Some(error) = eval_result.error {
                    dict.set_item("error", error)?;
                }
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Evaluation failed: {e}"))),
        }
    }

    /// Get evaluator name
    pub fn name(&self) -> &str {
        self.inner.name()
    }
}

// ============================================================================
// Dataset
// ============================================================================

/// Python wrapper for `EvaluationDataset`
#[pyclass]
pub struct PyEvaluationDataset {
    _inner: EvaluationDataset,
}

#[pymethods]
impl PyEvaluationDataset {
    /// Load dataset from JSONL file
    #[staticmethod]
    pub fn from_jsonl(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let rt = get_runtime()?;
        let result = rt.block_on(async { EvaluationDataset::from_jsonl(path).await });

        match result {
            Ok(dataset) => {
                let dict = PyDict::new(py);
                dict.set_item("name", dataset.name.clone())?;
                if let Some(desc) = &dataset.description {
                    dict.set_item("description", desc)?;
                }
                let test_cases_list = PyList::empty(py);
                for test_case in &dataset.test_cases {
                    let tc_dict = PyDict::new(py);
                    tc_dict.set_item("input", &test_case.input)?;
                    tc_dict.set_item("expected_output", &test_case.expected_output)?;
                    if let Some(ref ctx) = test_case.context {
                        tc_dict.set_item(
                            "context",
                            json_to_python_value(py, &serde_json::json!(ctx))?,
                        )?;
                    }
                    if let Some(ref meta) = test_case.metadata {
                        tc_dict.set_item(
                            "metadata",
                            json_to_python_value(py, &serde_json::json!(meta))?,
                        )?;
                    }
                    test_cases_list.append(tc_dict)?;
                }
                dict.set_item("test_cases", test_cases_list)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to load dataset: {e}"
            ))),
        }
    }

    /// Load dataset from JSON file
    #[staticmethod]
    pub fn from_json(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let rt = get_runtime()?;
        let result = rt.block_on(async { EvaluationDataset::from_json(path).await });

        match result {
            Ok(dataset) => {
                let dict = PyDict::new(py);
                dict.set_item("name", dataset.name.clone())?;
                if let Some(desc) = &dataset.description {
                    dict.set_item("description", desc)?;
                }
                let test_cases_list = PyList::empty(py);
                for test_case in &dataset.test_cases {
                    let tc_dict = PyDict::new(py);
                    tc_dict.set_item("input", &test_case.input)?;
                    tc_dict.set_item("expected_output", &test_case.expected_output)?;
                    if let Some(ref ctx) = test_case.context {
                        tc_dict.set_item(
                            "context",
                            json_to_python_value(py, &serde_json::json!(ctx))?,
                        )?;
                    }
                    if let Some(ref meta) = test_case.metadata {
                        tc_dict.set_item(
                            "metadata",
                            json_to_python_value(py, &serde_json::json!(meta))?,
                        )?;
                    }
                    test_cases_list.append(tc_dict)?;
                }
                dict.set_item("test_cases", test_cases_list)?;
                Ok(dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to load dataset: {e}"
            ))),
        }
    }
}
