//! Core prompt template implementation with variable substitution.
//!
//! Provides the `PromptTemplate` struct for creating reusable prompts with
//! variable placeholders that can be filled in dynamically.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// A part of a parsed template, either literal text or a variable placeholder
#[derive(Debug, Clone)]
enum TemplatePart<'a> {
    Literal(&'a str),
    Variable(&'a str),
}

#[derive(Error, Debug)]
pub enum TemplateError {
    #[error("Missing required variable: {0}")]
    MissingVariable(String),

    #[error("Invalid variable format in template")]
    InvalidFormat,

    #[error("Variable substitution error: {0}")]
    SubstitutionError(String),
}

/// A template for creating prompts with variable substitution.
///
/// Variables are specified using `{variable_name}` syntax in the template string.
///
/// # Examples
///
/// ```rust
/// use caliche::prompts::PromptTemplate;
/// use serde_json::json;
///
/// let template = PromptTemplate::new("Summarize this: {text}");
/// let prompt = template.format(&json!({"text": "Long document..."})).expect("Template formatting should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: HashSet<String>,
    partial_variables: HashMap<String, String>,
}

impl PromptTemplate {
    /// Create a new prompt template from a string.
    ///
    /// Variables are extracted automatically from `{variable}` placeholders.
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let variables = Self::extract_variables(&template);

        Self {
            template,
            variables,
            partial_variables: HashMap::new(),
        }
    }

    /// Add a partial variable that will be pre-filled in all format calls.
    pub fn with_partial(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.partial_variables.insert(key.into(), value.into());
        self
    }

    // ==================== Pre-built Templates ====================

    /// Create a summarization template.
    ///
    /// Variables: `{text}`
    ///
    /// # Example
    /// ```rust
    /// use caliche::prompts::PromptTemplate;
    ///
    /// let template = PromptTemplate::summarize();
    /// let prompt = template.format_string("Long article...").expect("Template formatting should succeed");
    /// ```
    pub fn summarize() -> Self {
        Self::new("Please provide a concise summary of the following text:\n\n{text}\n\nSummary:")
    }

    /// Create a detailed analysis template.
    ///
    /// Variables: `{text}`
    pub fn analyze() -> Self {
        Self::new("Please provide a detailed analysis of the following:\n\n{text}\n\nAnalysis:")
    }

    /// Create a translation template.
    ///
    /// Variables: `{text}`, `{target_language}`
    pub fn translate() -> Self {
        Self::new("Translate the following text to {target_language}:\n\n{text}\n\nTranslation:")
    }

    /// Create a code explanation template.
    ///
    /// Variables: `{code}`
    pub fn explain_code() -> Self {
        Self::new(
            "Explain what this code does in simple terms:\n\n```\n{code}\n```\n\nExplanation:",
        )
    }

    /// Create a question answering template.
    ///
    /// Variables: `{context}`, `{question}`
    pub fn question_answer() -> Self {
        Self::new(
            "Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        )
    }

    /// Create a text rewriting template.
    ///
    /// Variables: `{text}`, `{style}` (e.g., "formal", "casual", "professional")
    pub fn rewrite() -> Self {
        Self::new("Rewrite the following text in a {style} style:\n\n{text}\n\nRewritten version:")
    }

    /// Create a brainstorming template.
    ///
    /// Variables: `{topic}`, `{count}` (number of ideas)
    pub fn brainstorm() -> Self {
        Self::new("Generate {count} creative ideas for: {topic}\n\nIdeas:")
    }

    /// Create a data extraction template.
    ///
    /// Variables: `{text}`, `{fields}` (what to extract)
    pub fn extract() -> Self {
        Self::new(
            "Extract the following information from the text: {fields}\n\nText:\n{text}\n\nExtracted information (as JSON):",
        )
    }

    /// Create a comparison template.
    ///
    /// Variables: `{item1}`, `{item2}`
    pub fn compare() -> Self {
        Self::new("Compare and contrast the following:\n\nA: {item1}\n\nB: {item2}\n\nComparison:")
    }

    /// Create a sentiment analysis template.
    ///
    /// Variables: `{text}`
    pub fn sentiment() -> Self {
        Self::new(
            "Analyze the sentiment of the following text (positive, negative, or neutral):\n\n{text}\n\nSentiment:",
        )
    }

    // ==================== End Pre-built Templates ====================

    /// Get the list of variables required by this template.
    pub fn variables(&self) -> &HashSet<String> {
        &self.variables
    }

    /// Get the template string.
    pub fn template(&self) -> &str {
        &self.template
    }

    /// Get the partial variables map.
    pub fn partial_variables(&self) -> &HashMap<String, String> {
        &self.partial_variables
    }

    /// Format the template with a single string value.
    ///
    /// If the template has exactly one variable, you can pass just a string
    /// instead of a JSON object. This is more ergonomic for simple templates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use caliche::prompts::PromptTemplate;
    ///
    /// let template = PromptTemplate::new("Summarize: {text}");
    /// let result = template.format_string("Long document...").expect("Template formatting should succeed");
    /// ```
    pub fn format_string(&self, value: &str) -> Result<String, TemplateError> {
        if self.variables.len() == 1 {
            let var_name = self.variables.iter().next().ok_or_else(|| {
                TemplateError::SubstitutionError(
                    "Expected exactly one variable but found none".to_string(),
                )
            })?;
            self.format(&serde_json::json!({ var_name: value }))
        } else if self.variables.is_empty() {
            Ok(self.template.clone())
        } else {
            Err(TemplateError::SubstitutionError(format!(
                "Template has {} variables, use format() with a JSON object instead",
                self.variables.len()
            )))
        }
    }

    /// Format the template with the given variables.
    ///
    /// # Arguments
    ///
    /// * `values` - JSON object containing variable values
    ///
    /// # Errors
    ///
    /// Returns an error if required variables are missing.
    pub fn format(&self, values: &Value) -> Result<String, TemplateError> {
        // Parse template into parts
        let parts = Self::parse_template_parts(&self.template);

        // Pre-calculate approximate result size
        let mut estimated_size = self.template.len();
        if let Some(obj) = values.as_object() {
            for (_key, value) in obj {
                estimated_size += match value {
                    Value::String(s) => s.len(),
                    Value::Number(_) => 20, // Rough estimate for numbers
                    Value::Bool(_) => 5,
                    Value::Null => 0,
                    _ => 50, // Rough estimate for complex values
                };
            }
        }

        // Build result in single pass
        let mut result = String::with_capacity(estimated_size);
        let mut provided_vars = HashSet::with_capacity(self.variables.len());

        // Helper to get variable value with partial fallback
        let get_var_value = |var_name: &str| -> Option<String> {
            // Check provided values first
            if let Some(obj) = values.as_object()
                && let Some(value) = obj.get(var_name)
            {
                return Some(match value {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
                    Value::Null => String::new(),
                    _ => value.to_string(),
                });
            }

            // Fallback to partial variables
            self.partial_variables.get(var_name).cloned()
        };

        // Build result by iterating through parts
        for part in parts {
            match part {
                TemplatePart::Literal(text) => {
                    result.push_str(text);
                }
                TemplatePart::Variable(var_name) => {
                    if let Some(value) = get_var_value(var_name) {
                        result.push_str(&value);
                        provided_vars.insert(var_name);
                    } else {
                        // Variable not found, will be caught in validation
                        result.push('{');
                        result.push_str(var_name);
                        result.push('}');
                    }
                }
            }
        }

        // Track partial variables as provided
        for key in self.partial_variables.keys() {
            provided_vars.insert(key.as_str());
        }

        // Check for missing required variables
        for var in &self.variables {
            if !provided_vars.contains(var.as_str()) {
                return Err(TemplateError::MissingVariable(var.clone()));
            }
        }

        Ok(result)
    }

    /// Format the template using string key-value pairs.
    pub fn format_map(&self, values: &HashMap<String, String>) -> Result<String, TemplateError> {
        // Parse template into parts
        let parts = Self::parse_template_parts(&self.template);

        // Pre-calculate approximate result size
        let mut estimated_size = self.template.len();
        for value in values.values() {
            estimated_size += value.len();
        }

        // Build result in single pass
        let mut result = String::with_capacity(estimated_size);
        let mut provided_vars = HashSet::with_capacity(self.variables.len());

        // Helper to get variable value with partial fallback
        let get_var_value = |var_name: &str| -> Option<&str> {
            // Check provided values first
            if let Some(value) = values.get(var_name) {
                return Some(value.as_str());
            }

            // Fallback to partial variables
            self.partial_variables.get(var_name).map(|s| s.as_str())
        };

        // Build result by iterating through parts
        for part in parts {
            match part {
                TemplatePart::Literal(text) => {
                    result.push_str(text);
                }
                TemplatePart::Variable(var_name) => {
                    if let Some(value) = get_var_value(var_name) {
                        result.push_str(value);
                        provided_vars.insert(var_name);
                    } else {
                        // Variable not found, will be caught in validation
                        result.push('{');
                        result.push_str(var_name);
                        result.push('}');
                    }
                }
            }
        }

        // Track partial variables as provided
        for key in self.partial_variables.keys() {
            provided_vars.insert(key.as_str());
        }

        // Check for missing required variables
        for var in &self.variables {
            if !provided_vars.contains(var.as_str()) {
                return Err(TemplateError::MissingVariable(var.clone()));
            }
        }

        Ok(result)
    }

    /// Extract variables from a template string.
    fn extract_variables(template: &str) -> HashSet<String> {
        let mut variables = HashSet::new();
        let mut chars = template.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                let mut var_name = String::new();

                while let Some(&next_ch) = chars.peek() {
                    if next_ch == '}' {
                        chars.next(); // consume '}'
                        if !var_name.is_empty() {
                            variables.insert(var_name);
                        }
                        break;
                    } else if next_ch.is_alphanumeric() || next_ch == '_' {
                        var_name.push(next_ch);
                        chars.next();
                    } else {
                        // Invalid variable name, skip
                        break;
                    }
                }
            }
        }

        variables
    }

    /// Parse the template into literal and variable parts for single-pass formatting
    fn parse_template_parts(template: &str) -> Vec<TemplatePart<'_>> {
        let mut parts = Vec::new();
        let mut last_pos = 0;
        let bytes = template.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i] == b'{' {
                // Find the matching '}'
                let var_start = i + 1;
                let mut var_end = var_start;

                while var_end < bytes.len() && bytes[var_end] != b'}' {
                    var_end += 1;
                }

                // Check if we have a valid variable placeholder
                if var_end < bytes.len() && var_end > var_start {
                    let var_name = &template[var_start..var_end];
                    // Only treat as variable if it's a valid variable name (non-empty and alphanumeric/underscore)
                    if !var_name.is_empty()
                        && var_name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        // Add any literal text before this variable
                        if i > last_pos {
                            parts.push(TemplatePart::Literal(&template[last_pos..i]));
                        }

                        parts.push(TemplatePart::Variable(var_name));
                        i = var_end + 1;
                        last_pos = i;
                        continue;
                    }
                }
            }
            i += 1;
        }

        // Add any remaining literal text
        if last_pos < template.len() {
            parts.push(TemplatePart::Literal(&template[last_pos..]));
        }

        parts
    }

    /// Compose this template with another template.
    ///
    /// The output of this template becomes the input to the next template.
    pub fn then(self, next: PromptTemplate) -> ComposedTemplate {
        ComposedTemplate {
            templates: vec![self, next],
        }
    }
}

/// A composed template that chains multiple templates together.
#[derive(Debug, Clone)]
pub struct ComposedTemplate {
    templates: Vec<PromptTemplate>,
}

impl ComposedTemplate {
    /// Add another template to the chain.
    pub fn then(mut self, next: PromptTemplate) -> Self {
        self.templates.push(next);
        self
    }

    /// Format all templates in sequence.
    ///
    /// The output of each template is passed as the input to the next template
    /// under the variable name "text".
    pub fn format(&self, initial_values: &Value) -> Result<String, TemplateError> {
        let mut current_output = String::new();

        for (idx, template) in self.templates.iter().enumerate() {
            let values = if idx == 0 {
                // Use reference for first iteration (no clone)
                template.format(initial_values)?
            } else {
                // For subsequent iterations, wrap current output
                let values_obj = serde_json::json!({"text": current_output});
                template.format(&values_obj)?
            };

            current_output = values;
        }

        Ok(current_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_template() {
        let template = PromptTemplate::new("Hello {name}!");
        let result = template
            .format(&json!({"name": "World"}))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_multiple_variables() {
        let template = PromptTemplate::new("Hi {name}, you are {age} years old.");
        let result = template
            .format(&json!({
                "name": "Alice",
                "age": 30
            }))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Hi Alice, you are 30 years old.");
    }

    #[test]
    fn test_missing_variable() {
        let template = PromptTemplate::new("Hello {name}!");
        let result = template.format(&json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_variables() {
        let template = PromptTemplate::new("Hello {name}, welcome to {place}!")
            .with_partial("place", "Rust Land");

        let result = template
            .format(&json!({"name": "Alice"}))
            .expect("Template formatting should succeed");
        assert_eq!(result, "Hello Alice, welcome to Rust Land!");
    }

    #[test]
    fn test_variable_extraction() {
        let template = PromptTemplate::new("Test {var1} and {var2} and {var1} again");
        assert_eq!(template.variables().len(), 2);
        assert!(template.variables().contains("var1"));
        assert!(template.variables().contains("var2"));
    }

    #[test]
    fn test_composed_template() {
        let t1 = PromptTemplate::new("Analyze this: {input}");
        let t2 = PromptTemplate::new("Summarize: {text}");

        let composed = t1.then(t2);
        let result = composed
            .format(&json!({"input": "sample data"}))
            .expect("Composed template formatting should succeed");
        assert_eq!(result, "Summarize: Analyze this: sample data");
    }
}
