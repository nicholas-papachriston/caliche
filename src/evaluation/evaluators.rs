//! Evaluator implementations for scoring agent outputs

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evaluation result for a single test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub score: f64,
    pub passed: bool,
    pub details: HashMap<String, serde_json::Value>,
    pub error: Option<String>,
}

impl EvaluationResult {
    pub fn new(score: f64, passed: bool) -> Self {
        Self {
            score,
            passed,
            details: HashMap::new(),
            error: None,
        }
    }

    pub fn with_details(mut self, details: HashMap<String, serde_json::Value>) -> Self {
        self.details = details;
        self
    }

    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self.passed = false;
        self
    }
}

/// Trait for evaluating agent outputs
#[async_trait]
pub trait Evaluator: Send + Sync + std::fmt::Debug {
    /// Evaluate an agent output against expected output
    async fn evaluate(
        &self,
        output: &str,
        expected: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<EvaluationResult>;

    /// Get the name of this evaluator
    fn name(&self) -> &str;
}

/// Exact string match evaluator
#[derive(Debug)]
pub struct ExactMatchEvaluator {
    case_sensitive: bool,
}

impl ExactMatchEvaluator {
    pub fn new(case_sensitive: bool) -> Self {
        Self { case_sensitive }
    }

    pub fn case_insensitive() -> Self {
        Self {
            case_sensitive: false,
        }
    }
}

#[async_trait]
impl Evaluator for ExactMatchEvaluator {
    async fn evaluate(
        &self,
        output: &str,
        expected: &str,
        _context: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<EvaluationResult> {
        let passed = if self.case_sensitive {
            output == expected
        } else {
            output.to_lowercase() == expected.to_lowercase()
        };

        let score = if passed { 1.0 } else { 0.0 };

        let mut details = HashMap::new();
        details.insert("output".to_owned(), serde_json::json!(output));
        details.insert("expected".to_owned(), serde_json::json!(expected));
        details.insert(
            "case_sensitive".to_owned(),
            serde_json::json!(self.case_sensitive),
        );

        Ok(EvaluationResult::new(score, passed).with_details(details))
    }

    fn name(&self) -> &str {
        "exact_match"
    }
}

/// Regex pattern evaluator.
#[derive(Debug)]
pub struct RegexEvaluator {
    pattern: Regex,
}

impl RegexEvaluator {
    pub fn new(pattern: &str) -> Result<Self> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
        })
    }
}

#[async_trait]
impl Evaluator for RegexEvaluator {
    async fn evaluate(
        &self,
        output: &str,
        _expected: &str,
        _context: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<EvaluationResult> {
        let passed = self.pattern.is_match(output);
        let score = if passed { 1.0 } else { 0.0 };

        let mut details = HashMap::new();
        details.insert("output".to_owned(), serde_json::json!(output));
        details.insert(
            "pattern".to_owned(),
            serde_json::json!(self.pattern.as_str()),
        );
        details.insert("matched".to_owned(), serde_json::json!(passed));

        Ok(EvaluationResult::new(score, passed).with_details(details))
    }

    fn name(&self) -> &str {
        "regex"
    }
}

/// Error from [`LLMEvaluator::evaluate`] when no provider was set.
pub const LLM_EVALUATOR_NEEDS_PROVIDER: &str =
    "LLMEvaluator needs a provider and does not pick one from the environment";

/// LLM-based evaluator (uses an LLM to grade the response)
///
/// It has no default provider: without one, `evaluate` fails instead of
/// falling back to a cloud provider from the environment.
#[derive(Debug)]
pub struct LLMEvaluator {
    provider: Option<crate::llm::ProviderType>,
    model: Option<String>,
    criteria: String,
}

impl LLMEvaluator {
    /// Create an evaluator for `criteria`.
    ///
    /// Set a provider with [`Self::with_provider`] before calling `evaluate`;
    /// without one, `evaluate` fails with [`LLM_EVALUATOR_NEEDS_PROVIDER`].
    pub fn new(criteria: impl Into<String>) -> Self {
        Self {
            provider: None,
            model: None,
            criteria: criteria.into(),
        }
    }

    pub fn with_provider(mut self, provider: crate::llm::ProviderType) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[async_trait]
impl Evaluator for LLMEvaluator {
    async fn evaluate(
        &self,
        output: &str,
        expected: &str,
        context: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<EvaluationResult> {
        let Some(provider) = self.provider else {
            anyhow::bail!(LLM_EVALUATOR_NEEDS_PROVIDER);
        };

        // Build evaluation prompt
        let mut prompt = format!(
            "Evaluate the following agent output against the expected output.\n\n\
            Evaluation Criteria: {}\n\n\
            Expected Output: {}\n\n\
            Agent Output: {}\n\n",
            self.criteria, expected, output
        );

        if let Some(ctx) = context {
            prompt.push_str(&format!(
                "\nContext: {}\n\n",
                serde_json::to_string_pretty(ctx)?
            ));
        }

        prompt.push_str(
            "Respond with a JSON object containing:\n\
            - score: float between 0.0 and 1.0\n\
            - passed: boolean\n\
            - reasoning: string explaining your evaluation\n\n\
            JSON Response:",
        );

        // Create agent service
        use crate::agent::service::AgentService;
        let mut agent = AgentService::from_config(
            crate::AgentServiceConfig::new()
                .with_provider(Some(provider), self.model.clone())
                .with_all_default_tools(),
        )
        .await?;

        // Get LLM evaluation
        let chat_response = agent.chat(&prompt).await?;

        // Parse JSON response
        let eval_json: serde_json::Value = match serde_json::from_str(chat_response.text()) {
            Ok(val) => val,
            Err(_) => {
                // Try to extract JSON from markdown code blocks
                if let Some(json_str) = caliche_contracts::text::extract_json(chat_response.text())
                {
                    serde_json::from_str(json_str)?
                } else {
                    anyhow::bail!("No JSON found in LLM response");
                }
            }
        };

        let score = eval_json
            .get("score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        let passed = eval_json
            .get("passed")
            .and_then(|v| v.as_bool())
            .unwrap_or(score >= 0.7);

        let reasoning = eval_json
            .get("reasoning")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut details = HashMap::new();
        details.insert("output".to_owned(), serde_json::json!(output));
        details.insert("expected".to_owned(), serde_json::json!(expected));
        details.insert("reasoning".to_owned(), serde_json::json!(reasoning));
        details.insert("criteria".to_owned(), serde_json::json!(self.criteria));

        Ok(EvaluationResult::new(score, passed).with_details(details))
    }

    fn name(&self) -> &str {
        "llm"
    }
}

/// Contains evaluator - checks if output contains expected substring
#[derive(Debug)]
pub struct ContainsEvaluator {
    case_sensitive: bool,
}

impl ContainsEvaluator {
    pub fn new(case_sensitive: bool) -> Self {
        Self { case_sensitive }
    }

    pub fn case_insensitive() -> Self {
        Self {
            case_sensitive: false,
        }
    }
}

#[async_trait]
impl Evaluator for ContainsEvaluator {
    async fn evaluate(
        &self,
        output: &str,
        expected: &str,
        _context: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<EvaluationResult> {
        let passed = if self.case_sensitive {
            output.contains(expected)
        } else {
            output.to_lowercase().contains(&expected.to_lowercase())
        };

        let score = if passed { 1.0 } else { 0.0 };

        let mut details = HashMap::new();
        details.insert("output".to_owned(), serde_json::json!(output));
        details.insert("expected".to_owned(), serde_json::json!(expected));
        details.insert(
            "case_sensitive".to_owned(),
            serde_json::json!(self.case_sensitive),
        );

        Ok(EvaluationResult::new(score, passed).with_details(details))
    }

    fn name(&self) -> &str {
        "contains"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serves `reply` as the answer to every chat completion request, in
    /// streamed or plain form, like a local OpenAI-compatible server.
    fn fake_chat_server(reply: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake chat server");
        let addr = listener.local_addr().expect("fake chat server addr");
        let reply = reply.to_owned();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let request = read_request(&mut stream);
                let (content_type, body) = if request.contains("\"stream\":true") {
                    let chunk = serde_json::json!({
                        "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]
                    });
                    (
                        "text/event-stream",
                        format!("data: {chunk}\n\ndata: [DONE]\n\n"),
                    )
                } else {
                    let body = serde_json::json!({
                        "id": "c", "object": "chat.completion", "created": 0, "model": "fake",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    });
                    ("application/json", body.to_string())
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        format!("http://{addr}/v1")
    }

    /// Reads one HTTP request: the headers, then `Content-Length` bytes of body.
    fn read_request(stream: &mut std::net::TcpStream) -> String {
        let mut data = Vec::new();
        let mut buf = [0_u8; 4096];
        loop {
            let Ok(n) = stream.read(&mut buf) else { break };
            if n == 0 {
                break;
            }
            data.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&data);
            if let Some(end) = text.find("\r\n\r\n") {
                let length = text[..end]
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if data.len() >= end + 4 + length {
                    break;
                }
            }
        }
        String::from_utf8_lossy(&data).into_owned()
    }

    #[expect(
        unsafe_code,
        clippy::disallowed_methods,
        reason = "the test points the LM Studio provider at the fake server"
    )]
    fn set_lmstudio_base_url(url: Option<&str>) {
        // SAFETY: the tests that call this are serial, so nothing reads the environment meanwhile.
        unsafe {
            match url {
                Some(url) => std::env::set_var("LMSTUDIO_BASE_URL", url),
                None => std::env::remove_var("LMSTUDIO_BASE_URL"),
            }
        }
    }

    /// Grades through a fake LM Studio server that answers with `reply`.
    async fn grade(reply: &str) -> Result<(f64, bool), String> {
        set_lmstudio_base_url(Some(&fake_chat_server(reply)));
        let result = LLMEvaluator::new("Check if output is correct")
            .with_provider(crate::llm::ProviderType::LMStudio)
            .with_model("fake")
            .evaluate("Hello", "Hello", None)
            .await;
        set_lmstudio_base_url(None);
        result
            .map(|r| (r.score, r.passed))
            .map_err(|e| e.to_string())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn multi_line_fenced_verdict_is_read() {
        let reply = "Verdict:\n```json\n{\n  \"score\": 0.9,\n  \"passed\": true,\n  \"reasoning\": \"ok\"\n}\n```";
        assert_eq!(grade(reply).await, Ok((0.9, true)));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn unfenced_verdict_in_text_is_read() {
        let reply = r#"Verdict: {"score": 0.2, "passed": false, "reasoning": "no"} end"#;
        assert_eq!(grade(reply).await, Ok((0.2, false)));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn reply_without_json_is_refused() {
        assert_eq!(
            grade("I cannot grade this.").await,
            Err("No JSON found in LLM response".to_string())
        );
    }

    #[tokio::test]
    async fn llm_evaluator_without_a_provider_refuses() {
        let err = LLMEvaluator::new("Check if output is correct")
            .evaluate("Hello", "Hello", None)
            .await
            .expect_err("no provider must not fall back to one from the environment");
        assert_eq!(err.to_string(), LLM_EVALUATOR_NEEDS_PROVIDER);
    }
}
