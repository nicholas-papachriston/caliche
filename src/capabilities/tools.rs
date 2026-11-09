//! Tool capability detection and availability functions.

/// Check if a tool is available at compile time.
///
/// Returns `true` if the tool feature is enabled, `false` otherwise.
pub fn is_tool_available(tool_name: &str) -> bool {
    match tool_name {
        "calculator" => cfg!(feature = "tools-calculator"),
        "fs" => cfg!(feature = "tools-file"),
        "data_manipulation" | "data-manipulation" => cfg!(feature = "tools-data"),
        "http" | "web_search" | "web_fetch" => {
            cfg!(feature = "tools-http") || cfg!(feature = "reqwest")
        }
        "process_document" | "process-document" => cfg!(feature = "tools-document"),
        "execute_python" | "execute-python" => true, // Always available
        _ => false,
    }
}

/// Get a list of all available tool names as strings.
///
/// Returns a static slice of tool names that are compiled into the binary.
pub fn available_tools() -> &'static [&'static str] {
    &[
        #[cfg(feature = "tools-calculator")]
        "calculator",
        #[cfg(feature = "tools-file")]
        "fs",
        #[cfg(feature = "tools-data")]
        "data_manipulation",
        #[cfg(feature = "tools-http")]
        "http",
        #[cfg(any(feature = "tools-http", feature = "reqwest"))]
        "web_search",
        #[cfg(any(feature = "tools-http", feature = "reqwest"))]
        "web_fetch",
        #[cfg(feature = "tools-document")]
        "process_document",
        "execute_python",
    ]
}

/// Generate a helpful error message for an unavailable tool.
pub fn tool_not_available_error(tool_name: &str) -> anyhow::Error {
    let available = available_tools();
    anyhow::anyhow!(
        "Tool '{}' requires a feature flag to be enabled. Available tools: {}. \
         To enable this tool, add the corresponding feature flag to your Cargo.toml.",
        tool_name,
        available.join(", ")
    )
}

/// Check if RAG tools are available.
pub fn rag_tools_available() -> bool {
    cfg!(feature = "tools-rag")
}

/// Check if vector memory is available (required for some tools).
pub fn vector_memory_available() -> bool {
    cfg!(feature = "vector-memory")
}
