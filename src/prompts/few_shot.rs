//! Few-shot prompt templates for providing examples to the LLM.

use super::template::{PromptTemplate, TemplateError};
use serde_json::Value;

/// An example for few-shot prompting.
#[derive(Debug, Clone)]
pub struct Example {
    pub input: String,
    pub output: String,
}

impl Example {
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }
}

/// A template for few-shot prompting with examples.
///
/// Useful for providing the LLM with examples of the desired input/output format.
///
/// # Examples
///
/// ```rust
/// use caliche::prompts::{FewShotPromptTemplate, Example};
/// use serde_json::json;
///
/// let examples = vec![
///     Example::new("What is 2+2?", "4"),
///     Example::new("What is 3+3?", "6"),
/// ];
///
/// let template = FewShotPromptTemplate::new(
///     examples,
///     "Answer the following math question:",
///     "Question: {question}\nAnswer:",
/// );
///
/// let prompt = template.format(&json!({"question": "What is 5+5?"})).expect("Template formatting should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct FewShotPromptTemplate {
    examples: Vec<Example>,
    example_format: String,
    prefix: Option<String>,
    suffix: String,
    example_separator: String,
}

impl FewShotPromptTemplate {
    /// Create a new few-shot prompt template.
    ///
    /// # Arguments
    ///
    /// * `examples` - List of example input/output pairs
    /// * `prefix` - Optional text before the examples
    /// * `suffix` - Template for the final query (can include variables)
    pub fn new(
        examples: Vec<Example>,
        prefix: impl Into<String>,
        suffix: impl Into<String>,
    ) -> Self {
        Self {
            examples,
            example_format: "Input: {input}\nOutput: {output}".to_string(),
            prefix: Some(prefix.into()),
            suffix: suffix.into(),
            example_separator: "\n\n".to_string(),
        }
    }

    /// Create without a prefix.
    pub fn without_prefix(examples: Vec<Example>, suffix: impl Into<String>) -> Self {
        Self {
            examples,
            example_format: "Input: {input}\nOutput: {output}".to_string(),
            prefix: None,
            suffix: suffix.into(),
            example_separator: "\n\n".to_string(),
        }
    }

    /// Set the format for each example.
    ///
    /// Default is "Input: {input}\nOutput: {output}"
    pub fn with_example_format(mut self, format: impl Into<String>) -> Self {
        self.example_format = format.into();
        self
    }

    /// Set the separator between examples.
    ///
    /// Default is "\n\n"
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.example_separator = separator.into();
        self
    }

    /// Format the template with the given variables.
    pub fn format(&self, values: &Value) -> Result<String, TemplateError> {
        let mut parts = Vec::new();

        // Add prefix if present
        if let Some(prefix) = &self.prefix {
            parts.push(prefix.clone());
        }

        // Add formatted examples
        let example_template = PromptTemplate::new(&self.example_format);
        for example in &self.examples {
            let formatted = example_template.format(&serde_json::json!({
                "input": example.input,
                "output": example.output
            }))?;
            parts.push(formatted);
        }

        // Add suffix with variable substitution
        let suffix_template = PromptTemplate::new(&self.suffix);
        let formatted_suffix = suffix_template.format(values)?;
        parts.push(formatted_suffix);

        Ok(parts.join(&self.example_separator))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_few_shot_basic() {
        let examples = vec![Example::new("2+2", "4"), Example::new("3+3", "6")];

        let template = FewShotPromptTemplate::new(
            examples,
            "Solve the math problem:",
            "Question: {question}\nAnswer:",
        );

        let result = template
            .format(&json!({"question": "5+5"}))
            .expect("template formatting should succeed");
        assert!(result.contains("2+2"));
        assert!(result.contains("4"));
        assert!(result.contains("5+5"));
    }

    #[test]
    fn test_few_shot_custom_format() {
        let examples = vec![
            Example::new("hello", "greeting"),
            Example::new("goodbye", "farewell"),
        ];

        let template = FewShotPromptTemplate::new(examples, "Classify the word:", "{word} is a")
            .with_example_format("{input} -> {output}")
            .with_separator(" | ");

        let result = template
            .format(&json!({"word": "hi"}))
            .expect("template formatting should succeed");
        assert!(result.contains("hello -> greeting"));
        assert!(result.contains(" | "));
    }
}
