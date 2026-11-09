//! Integration tests for reasoning patterns

#[cfg(feature = "reasoning-patterns")]
mod reasoning_integration {
    use caliche::reasoning::{ReasoningResult, ReasoningStep, StepType};
    use serde_json::json;

    #[test]
    fn test_reasoning_result_serialization() {
        let steps = vec![
            ReasoningStep {
                step_type: StepType::Thought,
                content: "Analyzing the problem".to_string(),
                metadata: Some(json!({"iteration": 1})),
            },
            ReasoningStep {
                step_type: StepType::Answer,
                content: "The answer is 42".to_string(),
                metadata: None,
            },
        ];

        let result = ReasoningResult::success("The answer is 42".to_string(), steps);

        // Test serialization
        let json_str = serde_json::to_string(&result).unwrap();
        assert!(json_str.contains("The answer is 42"));
        assert!(json_str.contains("success"));

        // Test deserialization
        let deserialized: ReasoningResult = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.final_answer, result.final_answer);
        assert_eq!(deserialized.success, result.success);
    }

    #[test]
    fn test_reasoning_step_metadata() {
        let step = ReasoningStep {
            step_type: StepType::Thought,
            content: "Complex reasoning step".to_string(),
            metadata: Some(json!({
                "iteration": 1,
                "confidence": 0.95,
                "tool_used": "calculator"
            })),
        };

        assert_eq!(step.step_type, StepType::Thought);
        assert!(step.metadata.is_some());

        let metadata = step.metadata.unwrap();
        assert_eq!(metadata["iteration"], 1);
        assert_eq!(metadata["confidence"], 0.95);
        assert_eq!(metadata["tool_used"], "calculator");
    }

    #[test]
    fn test_step_type_equality() {
        assert_eq!(StepType::Thought, StepType::Thought);
        assert_eq!(StepType::Action, StepType::Action);
        assert_eq!(StepType::Observation, StepType::Observation);
        assert_eq!(StepType::Plan, StepType::Plan);
        assert_eq!(StepType::Reflection, StepType::Reflection);
        assert_eq!(StepType::Answer, StepType::Answer);

        assert_ne!(StepType::Thought, StepType::Action);
        assert_ne!(StepType::Action, StepType::Answer);
    }

    #[test]
    fn test_reasoning_result_failure_case() {
        let steps = vec![ReasoningStep {
            step_type: StepType::Thought,
            content: "Starting to solve".to_string(),
            metadata: None,
        }];

        let result =
            ReasoningResult::failure("Timeout: Maximum iterations exceeded".to_string(), steps);

        assert!(!result.success);
        assert_eq!(result.final_answer, "");
        assert!(result.error.is_some());
        assert_eq!(
            result.error.unwrap(),
            "Timeout: Maximum iterations exceeded"
        );
    }

    #[test]
    fn test_multiple_step_types_in_sequence() {
        let steps = vec![
            ReasoningStep {
                step_type: StepType::Thought,
                content: "I need to calculate the sum".to_string(),
                metadata: None,
            },
            ReasoningStep {
                step_type: StepType::Action,
                content: "calculator(10, 20)".to_string(),
                metadata: Some(json!({"tool": "calculator"})),
            },
            ReasoningStep {
                step_type: StepType::Observation,
                content: "Result: 30".to_string(),
                metadata: None,
            },
            ReasoningStep {
                step_type: StepType::Answer,
                content: "The sum is 30".to_string(),
                metadata: None,
            },
        ];

        assert_eq!(steps.len(), 4);
        assert_eq!(steps[0].step_type, StepType::Thought);
        assert_eq!(steps[1].step_type, StepType::Action);
        assert_eq!(steps[2].step_type, StepType::Observation);
        assert_eq!(steps[3].step_type, StepType::Answer);
    }
}
