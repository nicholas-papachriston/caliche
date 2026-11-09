//! Template variable system for parameterized templates.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Template variables definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariables {
    /// Map of variable name to variable definition
    #[serde(flatten)]
    pub variables: HashMap<String, VariableDefinition>,
}

/// Variable definition with type and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VariableDefinition {
    /// String variable
    #[serde(rename = "string")]
    String {
        /// Default value
        default: Option<String>,
        /// Whether this variable is required
        required: Option<bool>,
        /// Description of the variable
        description: Option<String>,
        /// Validation pattern (regex)
        pattern: Option<String>,
    },
    /// Integer variable
    #[serde(rename = "integer")]
    Integer {
        /// Default value
        default: Option<i64>,
        /// Whether this variable is required
        required: Option<bool>,
        /// Description of the variable
        description: Option<String>,
        /// Minimum value
        min: Option<i64>,
        /// Maximum value
        max: Option<i64>,
    },
    /// Enum variable
    #[serde(rename = "enum")]
    Enum {
        /// Possible values
        values: Vec<String>,
        /// Default value
        default: Option<String>,
        /// Whether this variable is required
        required: Option<bool>,
        /// Description of the variable
        description: Option<String>,
    },
    /// Boolean variable
    #[serde(rename = "boolean")]
    Boolean {
        /// Default value
        default: Option<bool>,
        /// Whether this variable is required
        required: Option<bool>,
        /// Description of the variable
        description: Option<String>,
    },
}

impl TemplateVariables {
    /// Create new empty variables
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Validate variable values against definitions
    pub fn validate(&self, values: &HashMap<String, serde_json::Value>) -> Result<()> {
        for (name, def) in &self.variables {
            validate_variable(name, def, values.get(name))?;
        }
        Ok(())
    }

    /// Apply default values to missing variables
    pub fn apply_defaults(&self, values: &mut HashMap<String, serde_json::Value>) {
        for (name, def) in &self.variables {
            if !values.contains_key(name) {
                match def {
                    VariableDefinition::String { default, .. } => {
                        if let Some(default_val) = default {
                            values.insert(
                                name.clone(),
                                serde_json::Value::String(default_val.clone()),
                            );
                        }
                    }
                    VariableDefinition::Integer { default, .. } => {
                        if let Some(default_val) = default {
                            values.insert(
                                name.clone(),
                                serde_json::Value::Number(serde_json::Number::from(*default_val)),
                            );
                        }
                    }
                    VariableDefinition::Enum { default, .. } => {
                        if let Some(default_val) = default {
                            values.insert(
                                name.clone(),
                                serde_json::Value::String(default_val.clone()),
                            );
                        }
                    }
                    VariableDefinition::Boolean { default, .. } => {
                        if let Some(default_val) = default {
                            values.insert(name.clone(), serde_json::Value::Bool(*default_val));
                        }
                    }
                }
            }
        }
    }
}

impl Default for TemplateVariables {
    fn default() -> Self {
        Self::new()
    }
}

fn require_if_needed(
    name: &str,
    required: Option<bool>,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    if let Some(true) = required {
        value.ok_or_else(|| anyhow::anyhow!("Required variable '{}' is missing", name))?;
    }
    Ok(())
}

fn validate_variable(
    name: &str,
    def: &VariableDefinition,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    match def {
        VariableDefinition::String {
            required, pattern, ..
        } => validate_string_variable(name, *required, pattern.as_deref(), value),
        VariableDefinition::Integer {
            required, min, max, ..
        } => validate_integer_variable(name, *required, *min, *max, value),
        VariableDefinition::Enum {
            values, required, ..
        } => validate_enum_variable(name, *required, values, value),
        VariableDefinition::Boolean { required, .. } => {
            validate_boolean_variable(name, *required, value)
        }
    }
}

fn validate_string_variable(
    name: &str,
    required: Option<bool>,
    pattern: Option<&str>,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    require_if_needed(name, required, value)?;
    let Some(val) = value else {
        return Ok(());
    };
    if !val.is_string() {
        return Err(anyhow::anyhow!("Variable '{}' must be a string", name));
    }
    let Some(pattern) = pattern else {
        return Ok(());
    };
    let re = regex::Regex::new(pattern)?;
    let str_val = val.as_str().ok_or_else(|| {
        anyhow::anyhow!("Variable '{}' must be a string for pattern matching", name)
    })?;
    if !re.is_match(str_val) {
        return Err(anyhow::anyhow!(
            "Variable '{}' does not match pattern: {}",
            name,
            pattern
        ));
    }
    Ok(())
}

fn validate_integer_variable(
    name: &str,
    required: Option<bool>,
    min: Option<i64>,
    max: Option<i64>,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    require_if_needed(name, required, value)?;
    let Some(val) = value else {
        return Ok(());
    };
    let int_val = val
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("Variable '{}' must be an integer", name))?;
    if let Some(min_val) = min
        && int_val < min_val
    {
        return Err(anyhow::anyhow!(
            "Variable '{}' must be >= {}",
            name,
            min_val
        ));
    }
    if let Some(max_val) = max
        && int_val > max_val
    {
        return Err(anyhow::anyhow!(
            "Variable '{}' must be <= {}",
            name,
            max_val
        ));
    }
    Ok(())
}

fn validate_enum_variable(
    name: &str,
    required: Option<bool>,
    values: &[String],
    value: Option<&serde_json::Value>,
) -> Result<()> {
    require_if_needed(name, required, value)?;
    let Some(val) = value else {
        return Ok(());
    };
    let str_val = val
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Variable '{}' must be a string", name))?;
    if !values.iter().any(|v| v == str_val) {
        return Err(anyhow::anyhow!(
            "Variable '{}' must be one of: {}",
            name,
            values.join(", ")
        ));
    }
    Ok(())
}

fn validate_boolean_variable(
    name: &str,
    required: Option<bool>,
    value: Option<&serde_json::Value>,
) -> Result<()> {
    require_if_needed(name, required, value)?;
    if let Some(val) = value
        && !val.is_boolean()
    {
        return Err(anyhow::anyhow!("Variable '{}' must be a boolean", name));
    }
    Ok(())
}
