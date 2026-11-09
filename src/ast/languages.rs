//! Language detection and grammar registry for tree-sitter.

use crate::ast::types::Language;
use std::path::Path;
use tree_sitter::Language as TsLanguage;

/// Detect programming language from file path (extension).
pub fn detect_language(path: &Path) -> Language {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "rs" => Language::Rust,
        "py" | "pyi" => Language::Python,
        "js" | "mjs" | "cjs" => Language::JavaScript,
        "ts" | "tsx" | "mts" | "cts" => Language::TypeScript,
        "go" => Language::Go,
        "java" => Language::Java,
        "c" => Language::C,
        "cc" | "cpp" | "cxx" | "h" | "hpp" | "hxx" => Language::Cpp,
        "rb" => Language::Ruby,
        "toml" => Language::Toml,
        "json" | "jsonc" | "json5" => Language::Json,
        "yaml" | "yml" => Language::Yaml,
        "html" | "htm" => Language::Html,
        "csv" | "tsv" | "psv" => Language::Csv,
        "md" | "mdx" | "mkd" | "markdown" => Language::Markdown,
        _ => Language::Unknown,
    }
}

/// Registry for lazily loading tree-sitter language grammars.
pub struct LanguageRegistry;

impl LanguageRegistry {
    /// Get tree-sitter Language for the given Language enum.
    /// Returns None if the grammar is not available (feature not enabled).
    pub fn get_language(lang: Language) -> Option<TsLanguage> {
        match lang {
            #[cfg(feature = "ast-rust")]
            Language::Rust => Self::rust(),
            #[cfg(feature = "ast-python")]
            Language::Python => Self::python(),
            #[cfg(feature = "ast-javascript")]
            Language::JavaScript => Self::javascript(),
            #[cfg(feature = "ast-typescript")]
            Language::TypeScript => Self::typescript(),
            #[cfg(feature = "ast-go")]
            Language::Go => Self::go(),
            #[cfg(feature = "ast-java")]
            Language::Java => Self::java(),
            #[cfg(feature = "ast-c")]
            Language::C => Self::c(),
            #[cfg(feature = "ast-cpp")]
            Language::Cpp => Self::cpp(),
            #[cfg(feature = "ast-json")]
            Language::Json => Self::json(),
            #[cfg(feature = "ast-yaml")]
            Language::Yaml => Self::yaml(),
            #[cfg(feature = "ast-html")]
            Language::Html => Self::html(),
            #[cfg(feature = "ast-markdown")]
            Language::Markdown => Self::markdown(),
            #[cfg(feature = "ast-csv")]
            Language::Csv => Self::csv(),
            _ => None,
        }
    }

    #[cfg(feature = "ast-rust")]
    fn rust() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_rust::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-python")]
    fn python() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_python::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-javascript")]
    fn javascript() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_javascript::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-typescript")]
    fn typescript() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into();
        Some(lang)
    }

    #[cfg(feature = "ast-go")]
    fn go() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_go::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-java")]
    fn java() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_java::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-c")]
    fn c() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_c::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-cpp")]
    fn cpp() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_cpp::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-json")]
    fn json() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_json::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-yaml")]
    fn yaml() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_yaml::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-html")]
    fn html() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_html::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-markdown")]
    fn markdown() -> Option<TsLanguage> {
        let lang: TsLanguage = tree_sitter_md::LANGUAGE.into();
        Some(lang)
    }

    #[cfg(feature = "ast-csv")]
    fn csv() -> Option<TsLanguage> {
        None
    }

    #[cfg(not(feature = "ast-rust"))]
    fn rust() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-python"))]
    fn python() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-javascript"))]
    fn javascript() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-typescript"))]
    fn typescript() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-go"))]
    fn go() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-java"))]
    fn java() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-c"))]
    fn c() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-cpp"))]
    fn cpp() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-json"))]
    fn json() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-yaml"))]
    fn yaml() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-html"))]
    fn html() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-markdown"))]
    fn markdown() -> Option<TsLanguage> {
        None
    }
    #[cfg(not(feature = "ast-csv"))]
    fn csv() -> Option<TsLanguage> {
        None
    }
}
