//! JavaScript bindings for Caliche Reasoning Patterns
#![expect(
    clippy::trailing_empty_array,
    reason = "NAPI callback code generation emits the zero-sized ABI container"
)]

use caliche::reasoning::plan_execute::PlanExecutePattern as RustPlanExecutePattern;
use caliche::reasoning::react::ReActPattern as RustReActPattern;
use caliche::reasoning::reflexion::ReflexionPattern as RustReflexionPattern;
use caliche::reasoning::self_ask::SelfAskPattern as RustSelfAskPattern;
use caliche::reasoning::tree_of_thoughts::TreeOfThoughtsPattern as RustTreeOfThoughtsPattern;
use caliche::reasoning::{ReasoningPattern, ReasoningResult, StepType};
use napi_derive::napi;
use std::sync::Arc;

async fn build_default_agent_service(
    provider: caliche::ProviderType,
    model: String,
) -> napi::Result<caliche::AgentService> {
    caliche::AgentService::from_config(
        caliche::AgentServiceConfig::new()
            .with_provider(Some(provider), Some(model))
            .with_all_default_tools(),
    )
    .await
    .map_err(|e| napi::Error::from_reason(format!("Failed to create agent service: {e}")))
}

/// JavaScript wrapper for `ReActPattern`
#[napi]
pub struct ReActPattern {
    inner: Arc<RustReActPattern>,
}

#[napi]
impl ReActPattern {
    /// Create a new `ReActPattern`
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(provider: String, model: String, max_iterations: u32) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let pattern = RustReActPattern::new(agent_service, max_iterations as usize);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute a `ReActPattern`
    #[napi]
    pub async fn execute(&self, query: String) -> napi::Result<ReasoningResultJs> {
        let result = self.inner.execute(&query).await;
        match result {
            Ok(reasoning_result) => Ok(reasoning_result.into()),
            Err(e) => Err(napi::Error::from_reason(format!(
                "ReAct pattern execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `PlanExecutePattern`
#[napi]
pub struct PlanExecutePattern {
    inner: Arc<RustPlanExecutePattern>,
}

#[napi]
impl PlanExecutePattern {
    /// Create a new `PlanExecutePattern`
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(provider: String, model: String, max_iterations: u32) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let pattern = RustPlanExecutePattern::new(agent_service, max_iterations as usize);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute a `PlanExecutePattern`
    #[napi]
    pub async fn execute(&self, query: String) -> napi::Result<ReasoningResultJs> {
        let result = self.inner.execute(&query).await;
        match result {
            Ok(reasoning_result) => Ok(reasoning_result.into()),
            Err(e) => Err(napi::Error::from_reason(format!(
                "PlanExecute pattern execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `ReflexionPattern`
#[napi]
pub struct ReflexionPattern {
    inner: Arc<RustReflexionPattern>,
}

#[napi]
impl ReflexionPattern {
    /// Create a new `ReflexionPattern`
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(provider: String, model: String, max_iterations: u32) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let pattern = RustReflexionPattern::new(agent_service, max_iterations as usize);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute a `ReflexionPattern`
    #[napi]
    pub async fn execute(&self, query: String) -> napi::Result<ReasoningResultJs> {
        let result = self.inner.execute(&query).await;
        match result {
            Ok(reasoning_result) => Ok(reasoning_result.into()),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Reflexion pattern execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `SelfAskPattern`
#[napi]
pub struct SelfAskPattern {
    inner: Arc<RustSelfAskPattern>,
}

#[napi]
impl SelfAskPattern {
    /// Create a new `SelfAskPattern`
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(
        provider: String,
        model: String,
        max_iterations: Option<u32>,
    ) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let pattern = RustSelfAskPattern::new(agent_service, max_iterations.unwrap_or(10) as usize);
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute a `SelfAskPattern`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self, query: String) -> napi::Result<ReasoningResultJs> {
        let result = self.inner.execute(&query).await;
        match result {
            Ok(reasoning_result) => Ok(reasoning_result.into()),
            Err(e) => Err(napi::Error::from_reason(format!(
                "SelfAsk pattern execution failed: {e}"
            ))),
        }
    }
}

/// JavaScript wrapper for `TreeOfThoughtsPattern`
#[napi]
pub struct TreeOfThoughtsPattern {
    inner: Arc<RustTreeOfThoughtsPattern>,
}

#[napi]
impl TreeOfThoughtsPattern {
    /// Create a new `TreeOfThoughtsPattern`
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(
        provider: String,
        model: String,
        max_iterations: u32,
        breadth: u32,
    ) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let agent_service = build_default_agent_service(provider_type, model).await?;

        let pattern = RustTreeOfThoughtsPattern::new(
            agent_service,
            max_iterations as usize,
            breadth as usize,
        );
        Ok(Self {
            inner: Arc::new(pattern),
        })
    }

    /// Execute a `TreeOfThoughtsPattern`
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn execute(&self, query: String) -> napi::Result<ReasoningResultJs> {
        let result = self.inner.execute(&query).await;
        match result {
            Ok(reasoning_result) => Ok(reasoning_result.into()),
            Err(e) => Err(napi::Error::from_reason(format!(
                "TreeOfThoughts pattern execution failed: {e}"
            ))),
        }
    }
}

#[napi(object)]
pub struct ReasoningResultJs {
    pub final_answer: String,
    pub success: bool,
    pub error: Option<String>,
    pub steps: Vec<ReasoningStepJs>,
    pub token_usage: Option<crate::TokenUsageJs>,
}

#[napi(object)]
pub struct ReasoningStepJs {
    pub step_type: String,
    pub content: String,
    pub metadata: Option<serde_json::Value>,
}

impl From<ReasoningResult> for ReasoningResultJs {
    fn from(result: ReasoningResult) -> Self {
        Self {
            final_answer: result.final_answer,
            success: result.success,
            error: result.error,
            steps: result.steps.into_iter().map(Into::into).collect(),
            token_usage: result.token_usage.map(Into::into),
        }
    }
}

impl From<caliche::reasoning::ReasoningStep> for ReasoningStepJs {
    fn from(step: caliche::reasoning::ReasoningStep) -> Self {
        Self {
            step_type: match step.step_type {
                StepType::Thought => "thought".to_string(),
                StepType::Action => "action".to_string(),
                StepType::Observation => "observation".to_string(),
                StepType::Plan => "plan".to_string(),
                StepType::Reflection => "reflection".to_string(),
                StepType::Answer => "answer".to_string(),
            },
            content: step.content,
            metadata: step.metadata,
        }
    }
}
