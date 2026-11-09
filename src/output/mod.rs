//! Output parsers for extracting structured data from LLM responses.
//!
//! This module provides parsers to convert raw LLM text outputs into structured
//! data types, with support for JSON schema validation and typed deserialization.
//!
//! # Examples
//!
//! JSON parsing:
//!
//! ```rust
//! use caliche::output::{JsonOutputParser, OutputParser};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let parser = JsonOutputParser::new();
//! let value = parser.parse(r#"{"name": "Alice"}"#).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Typed structured output:
//!
//! ```rust
//! use caliche::output::{StructuredOutputParser, OutputParser};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Person {
//!     name: String,
//!     age: u32,
//! }
//!
//! # async fn example() -> anyhow::Result<()> {
//! let parser = StructuredOutputParser::<Person>::new();
//! let person = parser.parse(r#"{"name": "Bob", "age": 25}"#).await?;
//! # Ok(())
//! # }
//! ```
//!
//! List parsing:
//!
//! ```rust
//! use caliche::output::{ListOutputParser, OutputParser};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let parser = ListOutputParser::new();
//! let items = parser.parse("apple, banana, cherry").await?;
//! # Ok(())
//! # }
//! ```

pub mod enforcer;
#[cfg(feature = "jsonschema")]
pub mod json_parser;
pub mod parser;
pub mod schema;
pub mod structured;

pub use enforcer::StructuredOutputEnforcer;
#[cfg(feature = "jsonschema")]
pub use json_parser::JsonOutputParser;
pub use parser::{EnumOutputParser, ListOutputParser, OutputParser, ParseError, PassthroughParser};
pub use schema::{JsonSchemaDefinition, ResponseFormat, StructuredOutputConfig};
pub use structured::StructuredOutputParser;
