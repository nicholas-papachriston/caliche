//! LSP type definitions and wrappers.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::path::PathBuf;

/// Position in a document (line and character).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Position {
    /// 0-based line number.
    pub line: u32,
    /// 0-based character offset.
    pub character: u32,
}

impl Position {
    pub fn new(line: u32, character: u32) -> Self {
        Self { line, character }
    }
}

/// Range in a document between two positions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Range {
    /// Start position (inclusive).
    pub start: Position,
    /// End position (exclusive).
    pub end: Position,
}

impl Range {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }
}

/// Diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    /// Error (highest priority).
    Error = 1,
    /// Warning.
    Warning = 2,
    /// Information.
    Information = 3,
    /// Hint (lowest priority).
    Hint = 4,
}

impl Serialize for DiagnosticSeverity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for DiagnosticSeverity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(match value {
            1 => Self::Error,
            2 => Self::Warning,
            3 => Self::Information,
            4 => Self::Hint,
            _ => Self::Information,
        })
    }
}

impl DiagnosticSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Information => "information",
            Self::Hint => "hint",
        }
    }
}

/// Related diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticRelatedInformation {
    /// Location of the related diagnostic.
    pub location: Location,
    /// Message of the related diagnostic.
    pub message: String,
}

/// Location in a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    /// Document URI.
    pub uri: String,
    /// Range in the document.
    pub range: Range,
}

/// A diagnostic message from an LSP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Range the diagnostic applies to.
    pub range: Range,
    /// Severity of the diagnostic.
    #[serde(default = "default_diagnostic_severity")]
    pub severity: DiagnosticSeverity,
    /// Optional diagnostic code.
    #[serde(default, deserialize_with = "deserialize_optional_code")]
    pub code: Option<String>,
    /// Source of the diagnostic (e.g., "rust-analyzer").
    pub source: Option<String>,
    /// Message text.
    pub message: String,
    /// Related diagnostic information.
    #[serde(default, rename = "relatedInformation")]
    pub related_info: Vec<DiagnosticRelatedInformation>,
}

fn default_diagnostic_severity() -> DiagnosticSeverity {
    DiagnosticSeverity::Information
}

fn deserialize_optional_code<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| match value {
        Value::String(s) => Some(s),
        Value::Number(n) => Some(n.to_string()),
        Value::Object(_) => Some(value.to_string()),
        _ => None,
    }))
}

impl Diagnostic {
    /// Format diagnostic for LLM consumption.
    pub fn format_for_llm(&self) -> String {
        let severity_str = self.severity.as_str();
        let code_part = self
            .code
            .as_ref()
            .map(|c| format!(" ({})", c))
            .unwrap_or_default();
        let source_part = self
            .source
            .as_ref()
            .map(|s| format!(" [{}]", s))
            .unwrap_or_default();
        format!(
            "{}:{}:{}: [{}]{}{}: {}",
            self.range.start.line + 1,
            self.range.start.character + 1,
            self.range.end.line + 1,
            severity_str,
            code_part,
            source_part,
            self.message
        )
    }
}

/// Symbol kind (from LSP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    File = 1,
    Module = 2,
    Namespace = 3,
    Package = 4,
    Class = 5,
    Method = 6,
    Property = 7,
    Field = 8,
    Constructor = 9,
    Enum = 10,
    Interface = 11,
    Function = 12,
    Variable = 13,
    Constant = 14,
    String = 15,
    Number = 16,
    Boolean = 17,
    Array = 18,
    Object = 19,
    Key = 20,
    Null = 21,
    EnumMember = 22,
    Struct = 23,
    Event = 24,
    Operator = 25,
    TypeParameter = 26,
}

impl Serialize for SymbolKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for SymbolKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::from_lsp_value(u64::deserialize(deserializer)?))
    }
}

impl SymbolKind {
    pub fn from_lsp_value(value: u64) -> Self {
        if let Some(kind) = Self::from_lsp_low(value) {
            return kind;
        }
        Self::from_lsp_high(value)
    }

    fn from_lsp_low(value: u64) -> Option<Self> {
        Some(match value {
            1 => Self::File,
            2 => Self::Module,
            3 => Self::Namespace,
            4 => Self::Package,
            5 => Self::Class,
            6 => Self::Method,
            7 => Self::Property,
            8 => Self::Field,
            9 => Self::Constructor,
            10 => Self::Enum,
            11 => Self::Interface,
            12 => Self::Function,
            13 => Self::Variable,
            _ => return None,
        })
    }

    fn from_lsp_high(value: u64) -> Self {
        match value {
            14 => Self::Constant,
            15 => Self::String,
            16 => Self::Number,
            17 => Self::Boolean,
            18 => Self::Array,
            19 => Self::Object,
            20 => Self::Key,
            21 => Self::Null,
            22 => Self::EnumMember,
            23 => Self::Struct,
            24 => Self::Event,
            25 => Self::Operator,
            26 => Self::TypeParameter,
            _ => Self::Variable,
        }
    }

    pub fn as_str(&self) -> &'static str {
        if let Some(name) = self.low_name() {
            return name;
        }
        self.high_name()
    }

    fn low_name(&self) -> Option<&'static str> {
        Some(match self {
            Self::File => "File",
            Self::Module => "Module",
            Self::Namespace => "Namespace",
            Self::Package => "Package",
            Self::Class => "Class",
            Self::Method => "Method",
            Self::Property => "Property",
            Self::Field => "Field",
            Self::Constructor => "Constructor",
            Self::Enum => "Enum",
            Self::Interface => "Interface",
            Self::Function => "Function",
            Self::Variable => "Variable",
            _ => return None,
        })
    }

    fn high_name(&self) -> &'static str {
        match self {
            Self::Constant => "Constant",
            Self::String => "String",
            Self::Number => "Number",
            Self::Boolean => "Boolean",
            Self::Array => "Array",
            Self::Object => "Object",
            Self::Key => "Key",
            Self::Null => "Null",
            Self::EnumMember => "EnumMember",
            Self::Struct => "Struct",
            Self::Event => "Event",
            Self::Operator => "Operator",
            Self::TypeParameter => "TypeParameter",
            // Kinds 1 through 13 are named by `low_name`.
            _ => "File",
        }
    }
}

/// Symbol information from workspace/symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol name.
    pub name: String,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Location of the symbol.
    pub location: Location,
    /// Optional container name.
    pub container_name: Option<String>,
}

/// Hover information returned by `textDocument/hover`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverInfo {
    /// Markdown/plaintext contents.
    pub contents: String,
    /// Optional range this hover applies to.
    pub range: Option<Range>,
}

/// A location formatted for human-facing tool output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileLocation {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        let pos = Position::new(10, 5);
        assert_eq!(pos.line, 10);
        assert_eq!(pos.character, 5);
    }

    #[test]
    fn test_diagnostic_format() {
        let diag = Diagnostic {
            range: Range::new(Position::new(9, 4), Position::new(9, 10)),
            severity: DiagnosticSeverity::Error,
            code: Some("E0425".to_string()),
            source: Some("rust-analyzer".to_string()),
            message: "cannot find value `x` in this scope".to_string(),
            related_info: vec![],
        };
        let formatted = diag.format_for_llm();
        assert!(formatted.contains("10:5"));
        assert!(formatted.contains("error"));
        assert!(formatted.contains("E0425"));
        assert!(formatted.contains("rust-analyzer"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(DiagnosticSeverity::Error < DiagnosticSeverity::Warning);
        assert!(DiagnosticSeverity::Warning < DiagnosticSeverity::Information);
        assert!(DiagnosticSeverity::Information < DiagnosticSeverity::Hint);
    }
}
