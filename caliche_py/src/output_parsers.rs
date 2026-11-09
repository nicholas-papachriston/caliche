//! Python bindings for Caliche Output Parsers

use crate::conversions::json_to_python_dict;
use crate::get_runtime;
use caliche::output::{EnumOutputParser, JsonOutputParser, ListOutputParser, OutputParser};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python wrapper for `JsonOutputParser`
#[pyclass]
pub struct PyJsonOutputParser {
    inner: JsonOutputParser,
}

#[pymethods]
impl PyJsonOutputParser {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: JsonOutputParser::new(),
        }
    }

    /// Parse JSON output
    pub fn parse(&self, text: &str) -> PyResult<Py<PyAny>> {
        let parser_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { OutputParser::parse(&parser_clone, text).await });

        Python::attach(|py| match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to parse JSON: {e}"
            ))),
        })
    }

    /// Get format instructions
    pub fn get_format_instructions(&self) -> String {
        OutputParser::get_format_instructions(&self.inner)
    }

    pub fn name(&self) -> String {
        OutputParser::name(&self.inner).to_string()
    }

    pub fn with_pretty_print(&mut self, pretty: bool) {
        self.inner = self.inner.clone().with_pretty_print(pretty);
    }
}

/// Python wrapper for `ListOutputParser`
#[pyclass]
pub struct PyListOutputParser {
    inner: ListOutputParser,
}

#[pymethods]
impl PyListOutputParser {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ListOutputParser::new(),
        }
    }

    pub fn with_separator(&mut self, separator: String) {
        self.inner = self.inner.clone().with_separator(separator);
    }

    pub fn with_trim(&mut self, trim: bool) {
        self.inner = self.inner.clone().with_trim(trim);
    }

    /// Parse list output
    pub fn parse(&self, text: &str) -> PyResult<Vec<String>> {
        let parser_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { OutputParser::parse(&parser_clone, text).await });

        match result {
            Ok(items) => Ok(items),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to parse list: {e}"
            ))),
        }
    }

    /// Get format instructions
    pub fn get_format_instructions(&self) -> String {
        OutputParser::get_format_instructions(&self.inner)
    }

    pub fn name(&self) -> String {
        OutputParser::name(&self.inner).to_string()
    }
}

/// Python wrapper for `EnumOutputParser`
#[pyclass]
pub struct PyEnumOutputParser {
    inner: EnumOutputParser,
}

#[pymethods]
impl PyEnumOutputParser {
    #[new]
    pub fn new(choices: Vec<String>) -> Self {
        Self {
            inner: EnumOutputParser::new(choices),
        }
    }

    pub fn with_case_sensitive(&mut self, sensitive: bool) {
        self.inner = self.inner.clone().with_case_sensitive(sensitive);
    }

    /// Parse enum output
    pub fn parse(&self, text: &str) -> PyResult<String> {
        let parser_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { OutputParser::parse(&parser_clone, text).await });

        match result {
            Ok(choice) => Ok(choice),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to parse enum: {e}"
            ))),
        }
    }

    /// Get format instructions
    pub fn get_format_instructions(&self) -> String {
        OutputParser::get_format_instructions(&self.inner)
    }

    pub fn name(&self) -> String {
        OutputParser::name(&self.inner).to_string()
    }
}

/// Python wrapper for `StructuredOutputParser`
/// Since `StructuredOutputParser` is generic, we use `JsonOutputParser` internally
/// and let Python handle the deserialization from the JSON result
#[pyclass]
pub struct PyStructuredOutputParser {
    inner: JsonOutputParser,
    include_instructions: bool,
    example: Option<String>,
}

#[pymethods]
impl PyStructuredOutputParser {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: JsonOutputParser::new(),
            include_instructions: true,
            example: None,
        }
    }

    pub const fn with_instructions(&mut self, include: bool) {
        self.include_instructions = include;
    }

    pub fn with_example(&mut self, example: String) {
        self.example = Some(example);
    }

    /// Parse structured output (returns JSON dict that can be deserialized in Python)
    pub fn parse(&self, text: &str) -> PyResult<Py<PyAny>> {
        let parser_clone = self.inner.clone();
        let rt = get_runtime()?;

        let result = rt.block_on(async { OutputParser::parse(&parser_clone, text).await });

        Python::attach(|py| match result {
            Ok(value) => {
                let py_dict = json_to_python_dict(py, &value)?;
                Ok(py_dict.into())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to parse structured output: {e}"
            ))),
        })
    }

    /// Get format instructions
    pub fn get_format_instructions(&self) -> String {
        let mut instructions = String::from(
            "Your response should be valid JSON that can be parsed into the required structure.",
        );

        if let Some(ref example) = self.example {
            instructions.push_str("\n\nExample:\n");
            instructions.push_str(example);
        }

        if !self.include_instructions {
            return String::new();
        }

        instructions
    }

    #[expect(
        clippy::unused_self,
        reason = "Python instance method ABI requires a receiver"
    )]
    pub fn name(&self) -> String {
        "structured".to_string()
    }
}
