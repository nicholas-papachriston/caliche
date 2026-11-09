//! JavaScript bindings for Caliche Output Parsers

use caliche::output::{
    EnumOutputParser as RustEnumOutputParser, JsonOutputParser as RustJsonOutputParser,
    ListOutputParser as RustListOutputParser, OutputParser,
};
use napi_derive::napi;

/// JavaScript wrapper for `JsonOutputParser`
#[napi]
pub struct JsonOutputParser {
    inner: RustJsonOutputParser,
}

#[napi]
impl JsonOutputParser {
    /// Create a new `JsonOutputParser`
    #[napi(constructor, catch_unwind)]
    pub fn new(pretty_print: Option<bool>) -> Self {
        Self {
            inner: RustJsonOutputParser::new().with_pretty_print(pretty_print.unwrap_or(false)),
        }
    }

    /// Parse JSON output
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn parse(&self, text: String) -> napi::Result<serde_json::Value> {
        OutputParser::parse(&self.inner, &text)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to parse JSON: {e}")))
    }

    /// Get format instructions for JSON parser
    #[napi(catch_unwind)]
    pub fn get_format_instructions(&self) -> String {
        OutputParser::get_format_instructions(&self.inner)
    }

    /// Get name of JSON parser
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        OutputParser::name(&self.inner).to_string()
    }

    /// Set pretty print for JSON parser
    #[napi(catch_unwind)]
    pub fn with_pretty_print(&self, pretty: bool) -> Self {
        Self {
            inner: self.inner.clone().with_pretty_print(pretty),
        }
    }
}

/// JavaScript wrapper for `ListOutputParser`
#[napi]
pub struct ListOutputParser {
    inner: caliche::output::ListOutputParser,
}

#[napi]
impl ListOutputParser {
    /// Create a new `ListOutputParser`
    #[napi(constructor, catch_unwind)]
    pub fn new(separator: Option<String>, trim: Option<bool>) -> Self {
        Self {
            inner: RustListOutputParser::new()
                .with_separator(separator.unwrap_or_else(|| ",".to_string()))
                .with_trim(trim.unwrap_or(true)),
        }
    }

    /// Parse list output
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn parse(&self, text: String) -> napi::Result<Vec<String>> {
        OutputParser::parse(&self.inner, &text)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to parse list: {e}")))
    }

    /// Get format instructions for list parser
    #[napi(catch_unwind)]
    pub fn get_format_instructions(&self) -> String {
        OutputParser::get_format_instructions(&self.inner)
    }

    /// Get name of list parser
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        OutputParser::name(&self.inner).to_string()
    }

    /// Set separator for list parser
    #[napi(catch_unwind)]
    pub fn with_separator(&self, separator: String) -> Self {
        Self {
            inner: self.inner.clone().with_separator(separator),
        }
    }

    /// Set trim for list parser
    #[napi(catch_unwind)]
    pub fn with_trim(&self, trim: bool) -> Self {
        Self {
            inner: self.inner.clone().with_trim(trim),
        }
    }
}

/// JavaScript wrapper for `EnumOutputParser`
#[napi]
pub struct EnumOutputParser {
    inner: RustEnumOutputParser,
}

#[napi]
impl EnumOutputParser {
    /// Create a new `EnumOutputParser`
    #[napi(constructor, catch_unwind)]
    pub fn new(choices: Vec<String>, case_sensitive: Option<bool>) -> Self {
        Self {
            inner: RustEnumOutputParser::new(choices)
                .with_case_sensitive(case_sensitive.unwrap_or(false)),
        }
    }

    /// Parse enum output
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn parse(&self, text: String) -> napi::Result<String> {
        OutputParser::parse(&self.inner, &text)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to parse enum: {e}")))
    }

    /// Get format instructions for enum parser
    #[napi(catch_unwind)]
    pub fn get_format_instructions(&self) -> String {
        OutputParser::get_format_instructions(&self.inner)
    }

    /// Get name of enum parser
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        OutputParser::name(&self.inner).to_string()
    }
}
