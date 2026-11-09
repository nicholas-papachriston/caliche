//! Prompt templates for variable substitution and reusable prompts.
//!
//! This module provides prompt template functionality, allowing
//! you to create reusable prompts with variable placeholders.
//!
//! # Note: Difference from Agent Templates
//!
//! This module (`prompts`) provides **text templating** with variable substitution
//! for LLM input formatting. For full **agent configuration templates** (YAML/JSON
//! files that define complete agent setups), see [`crate::templates`].
//!
//! # Examples
//!
//! Basic template:
//!
//! ```rust
//! use caliche::prompts::PromptTemplate;
//! use serde_json::json;
//!
//! let template = PromptTemplate::new("Summarize this: {text}");
//! let prompt = template.format(&json!({"text": "Long document..."})).expect("Template formatting should succeed");
//! ```
//!
//! Few-shot prompting:
//!
//! ```rust
//! use caliche::prompts::{FewShotPromptTemplate, Example};
//! use serde_json::json;
//!
//! let examples = vec![
//!     Example::new("What is 2+2?", "4"),
//!     Example::new("What is 3+3?", "6"),
//! ];
//!
//! let template = FewShotPromptTemplate::new(
//!     examples,
//!     "Answer the math question:",
//!     "Question: {question}\nAnswer:",
//! );
//!
//! let prompt = template.format(&json!({"question": "What is 5+5?"})).expect("Template formatting should succeed");
//! ```
//!
//! Chat templates:
//!
//! ```rust
//! use caliche::prompts::{ChatPromptTemplate, ChatMessage};
//! use serde_json::json;
//!
//! let template = ChatPromptTemplate::with_system("You are a {role}.")
//!     .add_user("Help me with: {task}");
//!
//! let messages = template.format(&json!({
//!     "role": "coding assistant",
//!     "task": "writing Rust code"
//! })).expect("Template formatting should succeed");
//! ```

pub mod chat_template;
pub mod few_shot;
#[cfg(feature = "s3")]
pub mod storage;
pub mod template;

pub use chat_template::{ChatMessage, ChatPromptTemplate, Role};
pub use few_shot::{Example, FewShotPromptTemplate};
#[cfg(feature = "s3")]
pub use storage::{PromptTemplateMetadata, PromptTemplateStore};
pub use template::{ComposedTemplate, PromptTemplate, TemplateError};
