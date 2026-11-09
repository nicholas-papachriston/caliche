use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Shell type for completion script generation
#[derive(Clone, Debug, ValueEnum)]
pub enum CompletionShell {
    /// Bash shell
    Bash,
    /// Zsh shell
    Zsh,
    /// Fish shell
    Fish,
    /// PowerShell
    #[value(name = "powershell")]
    PowerShell,
    /// Elvish shell
    Elvish,
}

/// Output format for CLI responses
#[derive(Clone, Debug, ValueEnum, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    /// Plain text output (default)
    #[default]
    Text,
    /// Full response as JSON
    Json,
    /// Line-delimited JSON for streaming
    Jsonl,
    /// SSE-style streaming output
    Stream,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Text => write!(f, "text"),
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Jsonl => write!(f, "jsonl"),
            OutputFormat::Stream => write!(f, "stream"),
        }
    }
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "jsonl" => Ok(OutputFormat::Jsonl),
            "stream" => Ok(OutputFormat::Stream),
            _ => Err(format!("Unknown output format: {}", s)),
        }
    }
}

/// Supported chain types for CLI execution.
#[derive(Clone, Debug, ValueEnum)]
pub enum ChainType {
    /// Single prompt + model execution chain
    Llm,
    /// Config-driven sequential chain
    Sequential,
    /// Map-reduce style chain
    #[value(name = "map-reduce")]
    MapReduce,
}

impl std::fmt::Display for ChainType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainType::Llm => write!(f, "llm"),
            ChainType::Sequential => write!(f, "sequential"),
            ChainType::MapReduce => write!(f, "map-reduce"),
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
pub enum RagCliSearchMode {
    Dense,
    Hybrid,
    #[value(name = "full_text")]
    FullText,
    #[value(name = "hybrid_full_text")]
    HybridFullText,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum RagCliFusionMode {
    Weighted,
    #[value(name = "rrf")]
    Rrf,
}
