//! Hook matchers for targeting specific tools or events.

use beef::lean::Cow;

/// Matcher for determining which tools a hook applies to
#[derive(Debug, Clone)]
pub enum HookMatcher {
    /// Match a specific tool by name
    Tool {
        /// Tool name to match
        name: Cow<'static, str>,
    },
    /// Match all tools
    All,
}

impl HookMatcher {
    /// Create a matcher for a specific tool
    pub fn tool(name: impl Into<Cow<'static, str>>) -> Self {
        Self::Tool { name: name.into() }
    }

    /// Create a matcher that matches all tools
    #[must_use]
    pub const fn all() -> Self {
        Self::All
    }
}
