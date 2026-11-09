//! Agent chat response returned by high-level chat APIs.

use crate::messages::{PerformanceMetrics, TokenUsage};

/// Response from a chat interaction with the agent.
///
/// Provides convenient access to the response text, token usage, and tool execution information.
/// Implements `Display` and `Deref<Target=str>` for ergonomic usage.
///
/// # Examples
///
/// ```rust,no_run
/// use caliche::AgentService;
///
/// # async fn example() -> anyhow::Result<()> {
/// let mut agent = AgentService::new().await?;
///
/// // Simple usage - just print the response
/// let response = agent.chat("Hello!").await?;
/// println!("{}", response);  // impl Display
///
/// // Access text directly
/// let text = response.text();
///
/// // Access token usage
/// if let Some(total) = response.usage.total_tokens {
///     println!("Used {} tokens", total);
/// }
///
/// // Check if tools were used
/// if response.used_tools() {
///     println!("Tools were invoked");
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The response text from the agent
    pub text: String,
    /// Token usage information
    pub usage: TokenUsage,
    /// Whether any tools were executed during this chat interaction
    pub tools_used: bool,
    /// List of tool names that were executed during this chat interaction
    pub tools_executed: Vec<String>,
    /// Whether the response was cut short due to iteration limit
    pub limit_reached: bool,
    /// Optional performance metrics (timing breakdown)
    pub metrics: Option<PerformanceMetrics>,
    /// Provider chain-of-thought for this turn. Empty when the model sent none.
    pub reasoning_content: String,
}

impl ChatResponse {
    /// Create a new ChatResponse
    pub fn new(text: impl Into<String>, usage: TokenUsage, tools_used: bool) -> Self {
        Self {
            text: text.into(),
            usage,
            tools_used,
            tools_executed: Vec::new(),
            limit_reached: false,
            metrics: None,
            reasoning_content: String::new(),
        }
    }

    /// Attach provider chain-of-thought. Empty text is stored as empty.
    #[must_use]
    pub fn with_reasoning_content(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = reasoning.into();
        self
    }

    /// Provider chain-of-thought for this turn, if any.
    #[must_use]
    pub fn reasoning_content(&self) -> &str {
        &self.reasoning_content
    }

    /// Create a new ChatResponse with tool execution list
    pub fn with_tools_executed(mut self, tools_executed: Vec<String>) -> Self {
        self.tools_executed = tools_executed;
        self
    }

    /// Attach performance metrics to the response
    pub fn with_metrics(mut self, metrics: PerformanceMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Get the response text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Check if tools were used during this interaction
    pub fn used_tools(&self) -> bool {
        self.tools_used
    }

    /// Get token usage information
    pub fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    /// Check if the iteration limit was reached
    pub fn limit_reached(&self) -> bool {
        self.limit_reached
    }

    /// Set the limit_reached flag (builder pattern)
    pub fn with_limit_reached(mut self) -> Self {
        self.limit_reached = true;
        self
    }
}

impl std::fmt::Display for ChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

impl std::ops::Deref for ChatResponse {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.text
    }
}
