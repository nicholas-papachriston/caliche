//! Shared Python↔JSON conversion utilities
//!
//! This module provides centralized conversion functions between Python objects
//! and JSON (`serde_json::Value`) to eliminate code duplication across the codebase.

use caliche::messages::{Message, MessageRole};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use pythonize::{depythonize, pythonize};
use serde_json::Value;
use std::sync::Mutex;

/// Convert a Python value to a JSON Value
///
/// Uses pythonize for efficient conversion, with fallback for edge cases.
pub fn python_to_json_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    // Try pythonize first for efficiency
    if let Ok(json_value) = depythonize(value) {
        return Ok(json_value);
    }

    // Fallback to manual conversion for edge cases
    if value.is_none() {
        return Ok(Value::Null);
    }

    if let Ok(b) = value.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }

    if let Ok(f) = value.extract::<f64>()
        && let Some(n) = serde_json::Number::from_f64(f)
    {
        return Ok(Value::Number(n));
    }

    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::String(s));
    }

    if let Ok(list) = value.cast::<PyList>() {
        let mut vec = Vec::new();
        for item in list.iter() {
            vec.push(python_to_json_value(&item)?);
        }
        return Ok(Value::Array(vec));
    }

    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in dict {
            let key_str = key.extract::<String>()?;
            let json_value = python_to_json_value(&value)?;
            map.insert(key_str, json_value);
        }
        return Ok(Value::Object(map));
    }

    let type_name = value
        .get_type()
        .name()
        .map_or_else(|_| "unknown".to_string(), |name| name.to_string());
    Err(PyRuntimeError::new_err(format!(
        "Unsupported Python type: {type_name}"
    )))
}

/// Empty JSON Schema object used when a Python tool's `parameters()` method
/// is missing or returns an unconvertible value.
pub fn empty_parameters_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
        "required": []
    })
}

/// Call a Python tool's `parameters()` method, convert the result to JSON, and
/// cache it. Shared by `PyTool` and the in-process MCP tool adapter.
pub fn cached_tool_parameters(tool_obj: &Py<PyAny>, cache: &Mutex<Option<Value>>) -> Value {
    {
        if let Ok(guard) = cache.lock()
            && let Some(ref params) = *guard
        {
            return params.clone();
        }
    }

    let params = Python::attach(|py| {
        tool_obj
            .call_method0(py, "parameters")
            .and_then(|result| {
                let bound = result.bind(py);
                // Prefer structured conversion; fall back to parsing a JSON string
                // for tools that return schema text instead of a dict.
                python_to_json_value(bound).or_else(|_| {
                    bound.extract::<String>().and_then(|s| {
                        serde_json::from_str::<Value>(&s)
                            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON: {e}")))
                    })
                })
            })
            .unwrap_or_else(|_| empty_parameters_schema())
    });

    if let Ok(mut guard) = cache.lock() {
        *guard = Some(params.clone());
    }

    params
}

/// Convert a JSON Value to a Python object
///
/// Uses pythonize for efficient conversion.
pub fn json_to_python_value(py: Python, value: &Value) -> PyResult<Py<PyAny>> {
    // Try pythonize first for efficiency
    let bound = pythonize(py, value)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert JSON to Python: {e}")))?;
    Ok(bound.unbind())
}

/// Convert a JSON Value to a Python dict
///
/// Convenience function for converting JSON objects directly to `PyDict`.
pub fn json_to_python_dict<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyDict>> {
    match value {
        Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                let py_value = json_to_python_value(py, v)?;
                dict.set_item(k, py_value)?;
            }
            Ok(dict)
        }
        _ => Err(PyRuntimeError::new_err("Expected JSON object")),
    }
}

/// Convert a Python dict to a `HashMap`<String, Value>
///
/// Convenience function used by evaluation module.
pub fn py_dict_to_hashmap(
    dict: &Bound<'_, PyDict>,
) -> PyResult<std::collections::HashMap<String, Value>> {
    let mut map = std::collections::HashMap::new();
    for (key, value) in dict {
        let key_str = key.extract::<String>()?;
        let value_json = python_to_json_value(&value)?;
        map.insert(key_str, value_json);
    }
    Ok(map)
}

/// Convert a Python message object to a Rust Message
///
/// Accepts either a `PyMessage` object or a dict with 'role' and 'content' keys.
pub fn py_message_to_rust(msg: &Bound<'_, PyAny>) -> PyResult<Message> {
    // Try to extract as PyMessage first
    if let Ok(py_msg) = msg.extract::<crate::llm_client::PyMessage>() {
        return Ok(Message::from(py_msg));
    }

    // Try to extract as dict
    if let Ok(dict) = msg.cast::<PyDict>() {
        let role_str = dict
            .get_item("role")?
            .and_then(|r| r.extract::<String>().ok())
            .ok_or_else(|| PyRuntimeError::new_err("Message dict missing 'role' field"))?;

        let content = dict
            .get_item("content")?
            .and_then(|c| c.extract::<String>().ok())
            .ok_or_else(|| PyRuntimeError::new_err("Message dict missing 'content' field"))?;

        let role = match role_str.to_lowercase().as_str() {
            "system" => MessageRole::System,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };

        return Ok(Message::new(role, content));
    }

    Err(PyRuntimeError::new_err(
        "Message must be a PyMessage object or dict with 'role' and 'content' fields",
    ))
}

/// Convert a Rust Message to a Python dict
pub fn rust_message_to_py(py: Python, msg: &Message) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    let role_str = match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };

    dict.set_item("role", role_str)?;
    dict.set_item("content", msg.extract_text())?;

    if let Some(ref metadata) = msg.metadata {
        dict.set_item("metadata", json_to_python_value(py, metadata)?)?;
    }

    Ok(dict.into())
}
