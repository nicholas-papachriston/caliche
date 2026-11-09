//! Pre-configured agent presets for common use cases.
//!
//! This module provides ready-to-use agent configurations for common scenarios,
//! making it easier to get started without manually configuring every option.

use crate::agent::builder::AgentBuilder;

/// Pre-configured agent presets for common use cases.
///
/// # Examples
///
/// ```rust,no_run
/// use caliche::presets::AgentPresets;
///
/// # async fn example() -> anyhow::Result<()> {
/// // Create a conversational agent with default settings
/// let mut agent = AgentPresets::conversational().build().await?;
///
/// // Create a code assistant
/// let mut agent = AgentPresets::code_assistant().build().await?;
///
/// // Create a data analyst
/// let mut agent = AgentPresets::data_analyst().build().await?;
/// # Ok(())
/// # }
/// ```
pub struct AgentPresets;

impl AgentPresets {
    /// Create a conversational agent preset.
    ///
    /// Configured for natural dialogue with:
    /// - Friendly, helpful personality
    /// - Calculator tool
    /// - Moderate temperature for natural responses
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::conversational().build().await?;
    /// let response = agent.chat("Hello! How are you?").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn conversational() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are a friendly and helpful AI assistant. \
                 You provide clear, concise responses and ask clarifying questions when needed. \
                 You maintain a warm, conversational tone while being informative."
                    .to_string(),
            )
            .enable_tool("calculator")
    }

    /// Create a code assistant preset.
    ///
    /// Configured for software development with:
    /// - Technical, precise communication style
    /// - Code execution, file reading, and calculator tools
    /// - Lower temperature for more deterministic outputs
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::code_assistant().build().await?;
    /// let response = agent.chat("Explain this Python function").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn code_assistant() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are an expert software engineer and coding assistant. \
                 You help with code review, debugging, explanations, and writing clean, efficient code. \
                 You provide detailed technical explanations and follow best practices. \
                 When writing code, you include comments and explain your reasoning."
                    .to_string(),
            )
            .enable_tool("execute_python")
            .enable_tool("fs")
            .enable_tool("calculator")
            .enable_tool("http")
    }

    /// Create a data analyst preset.
    ///
    /// Configured for data analysis and manipulation with:
    /// - Analytical, data-focused communication
    /// - Data manipulation, calculator, and file tools
    /// - Document processing capabilities
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::data_analyst().build().await?;
    /// let response = agent.chat("Analyze this dataset").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn data_analyst() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are a skilled data analyst. \
                 You help analyze datasets, identify patterns, create visualizations, and derive insights. \
                 You explain statistical concepts clearly and provide actionable recommendations. \
                 You're proficient with data manipulation and can work with various data formats."
                    .to_string(),
            )
            .enable_tool("calculator")
            .enable_tool("data_manipulation")
            .enable_tool("fs")
            .enable_tool("process_document")
    }

    /// Create a research assistant preset.
    ///
    /// Configured for research and information gathering with:
    /// - Academic, thorough communication style
    /// - Web fetching, search, and document processing tools
    /// - File reading capabilities
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::research_assistant().build().await?;
    /// let response = agent.chat("Research quantum computing").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn research_assistant() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are a thorough research assistant. \
                 You help gather information, synthesize findings from multiple sources, \
                 and present well-organized research summaries. \
                 You cite sources when available and distinguish between facts and opinions. \
                 You're skilled at breaking down complex topics into understandable explanations."
                    .to_string(),
            )
            .enable_tool("http")
            .enable_tool("web_search")
            .enable_tool("web_fetch")
            .enable_tool("perplexity_search")
            .enable_tool("fs")
            .enable_tool("process_document")
    }

    /// Create a system administrator preset.
    ///
    /// Configured for system administration tasks with:
    /// - Technical, security-conscious communication
    /// - File system tools (read, list directory)
    /// - Code execution for scripts
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::sysadmin().build().await?;
    /// let response = agent.chat("Check system logs").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn sysadmin() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are an experienced system administrator. \
                 You help with server management, troubleshooting, automation, and system monitoring. \
                 You prioritize security, reliability, and best practices. \
                 You provide clear explanations of system behavior and potential issues."
                    .to_string(),
            )
            .enable_tool("execute_python")
            .enable_tool("fs")
            .enable_tool("fs")
            .enable_tool("calculator")
    }

    /// Create a creative writer preset.
    ///
    /// Configured for creative writing with:
    /// - Creative, expressive communication style
    /// - Minimal tools (no tools by default)
    /// - Higher temperature for more creative outputs
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::creative_writer().build().await?;
    /// let response = agent.chat("Write a short story about a robot").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn creative_writer() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are a creative writer with a vivid imagination. \
                 You craft engaging narratives, develop interesting characters, and create compelling dialogue. \
                 You're skilled in various writing styles and genres. \
                 You help brainstorm ideas, develop plots, and refine creative works."
                    .to_string(),
            )
    }

    /// Create a minimal agent with no tools.
    ///
    /// A basic agent with no pre-configured tools or personality.
    /// Useful as a starting point for custom configurations.
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::minimal()
    ///     .system_prompt("Custom instructions".to_string())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn minimal() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Create a customer support agent preset.
    ///
    /// Configured for customer service with:
    /// - Empathetic, professional communication
    /// - Search and information retrieval tools
    /// - Document processing for knowledge bases
    ///
    /// # Example
    /// ```rust,no_run
    /// use caliche::presets::AgentPresets;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentPresets::customer_support().build().await?;
    /// let response = agent.chat("I need help with my order").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn customer_support() -> AgentBuilder {
        AgentBuilder::new()
            .system_prompt(
                "You are a helpful customer support agent. \
                 You're patient, empathetic, and focused on resolving customer issues. \
                 You ask clarifying questions to understand problems fully. \
                 You provide step-by-step solutions and follow up to ensure satisfaction. \
                 You maintain a professional yet friendly tone."
                    .to_string(),
            )
            .enable_tool("perplexity_search")
            .enable_tool("web_search")
            .enable_tool("web_fetch")
            .enable_tool("http")
            .enable_tool("process_document")
    }
}
