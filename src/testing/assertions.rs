//! Assertion helpers for testing

use anyhow::Result;

/// Assert that response contains text
pub fn assert_contains(response: &str, expected: &str) -> Result<()> {
    if response.contains(expected) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Response does not contain '{}'. Response: {}",
            expected,
            response
        ))
    }
}

/// Assert that response matches regex pattern
#[cfg(feature = "tools-data")]
pub fn assert_matches(response: &str, pattern: &str) -> Result<()> {
    use regex::Regex;
    let re = Regex::new(pattern)?;
    if re.is_match(response) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Response does not match pattern '{}'. Response: {}",
            pattern,
            response
        ))
    }
}

/// Assert token usage is within expected range
pub fn assert_token_usage(
    usage: &crate::messages::TokenUsage,
    min_total: Option<u32>,
    max_total: Option<u32>,
) -> Result<()> {
    if let Some(total) = usage.total_tokens {
        if let Some(min) = min_total
            && total < min
        {
            return Err(anyhow::anyhow!(
                "Token usage {} is below minimum {}",
                total,
                min
            ));
        }
        if let Some(max) = max_total
            && total > max
        {
            return Err(anyhow::anyhow!(
                "Token usage {} exceeds maximum {}",
                total,
                max
            ));
        }
    }
    Ok(())
}

/// Assert that a tool was called
pub fn assert_tool_called(tool_calls: &[String], expected_tool: &str) -> Result<()> {
    if tool_calls.contains(&expected_tool.to_string()) {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Tool '{}' was not called. Called tools: {:?}",
            expected_tool,
            tool_calls
        ))
    }
}
