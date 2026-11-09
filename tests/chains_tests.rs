/// Tests for chain patterns
#[cfg(all(feature = "chains", feature = "prompts"))]
mod chain_tests {
    use async_trait::async_trait;
    use caliche::chains::{
        Chain, ChainContext, ChainError, ChainResult, ComposableChain, MapReduceChain,
    };
    use serde_json::{Value, json};
    use std::sync::Arc;

    // Simple mock chain for testing
    #[derive(Debug)]
    struct MockChain {
        name: String,
        output: String,
    }

    impl MockChain {
        fn new(name: &str, output: &str) -> Self {
            Self {
                name: name.to_string(),
                output: output.to_string(),
            }
        }
    }

    #[async_trait]
    impl Chain for MockChain {
        async fn run(&self, _input: &Value) -> Result<ChainResult, ChainError> {
            Ok(ChainResult::new(Value::String(self.output.clone())))
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    // Error chain for testing error handling
    #[derive(Debug)]
    struct ErrorChain;

    #[async_trait]
    impl Chain for ErrorChain {
        async fn run(&self, _input: &Value) -> Result<ChainResult, ChainError> {
            Err(ChainError::ExecutionFailed("Mock error".to_string()))
        }

        fn name(&self) -> &str {
            "error_chain"
        }
    }

    #[tokio::test]
    async fn test_sequential_chain_basic() {
        let chain1 = MockChain::new("chain1", "Result 1");
        let chain2 = MockChain::new("chain2", "Result 2");

        let seq_chain = ComposableChain::new(chain1) >> ComposableChain::new(chain2);

        let result = seq_chain
            .run(&json!({"input": "test"}))
            .await
            .expect("Chain execution should succeed");
        assert!(result.output.is_string());
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Result 2"
        );
    }

    #[tokio::test]
    async fn test_sequential_chain_name() {
        let chain1 = MockChain::new("chain1", "Result 1");
        let chain2 = MockChain::new("chain2", "Result 2");
        let chain = ComposableChain::new(chain1) >> ComposableChain::new(chain2);
        assert_eq!(chain.name(), "sequential_chain");
    }

    #[tokio::test]
    async fn test_sequential_chain_single() {
        // SequentialChain requires 2 chains, so we'll use a single chain wrapped in ComposableChain
        let chain1 = MockChain::new("process", "Processed");
        let chain = ComposableChain::new(chain1);

        let result = chain
            .run(&json!({"input": "test"}))
            .await
            .expect("Chain execution should succeed");
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Processed"
        );
    }

    #[tokio::test]
    async fn test_sequential_chain_multiple() {
        let chain1 = MockChain::new("step1", "Step 1 done");
        let chain2 = MockChain::new("step2", "Step 2 done");
        let chain3 = MockChain::new("step3", "Step 3 done");

        // Chain multiple chains using >> operator
        let seq_chain = ComposableChain::new(chain1)
            >> ComposableChain::new(chain2)
            >> ComposableChain::new(chain3);

        let result = seq_chain
            .run(&json!({"input": "start"}))
            .await
            .expect("Chain execution should succeed");
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Step 3 done"
        );
    }

    #[tokio::test]
    async fn test_map_reduce_chain_basic() {
        let map_chain = Arc::new(MockChain::new("map", "Mapped"));
        let reduce_chain = Arc::new(MockChain::new("reduce", "Reduced"));

        let mr_chain = MapReduceChain::new(map_chain, reduce_chain);

        let items = json!(["item1", "item2", "item3"]);
        let result = mr_chain
            .run(&items)
            .await
            .expect("MapReduce chain execution should succeed");

        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Reduced"
        );
    }

    #[tokio::test]
    async fn test_map_reduce_chain_name() {
        let map_chain = Arc::new(MockChain::new("map", "Mapped"));
        let reduce_chain = Arc::new(MockChain::new("reduce", "Reduced"));

        let chain = MapReduceChain::new(map_chain, reduce_chain);
        assert_eq!(chain.name(), "map_reduce_chain");
    }

    #[tokio::test]
    async fn test_map_reduce_chain_empty_items() {
        let map_chain = Arc::new(MockChain::new("map", "Mapped"));
        let reduce_chain = Arc::new(MockChain::new("reduce", "Empty result"));

        let mr_chain = MapReduceChain::new(map_chain, reduce_chain);

        let items = json!([]);
        let result = mr_chain
            .run(&items)
            .await
            .expect("MapReduce chain execution should succeed");
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Empty result"
        );
    }

    #[tokio::test]
    async fn test_map_reduce_chain_single_item() {
        let map_chain = Arc::new(MockChain::new("map", "Mapped result"));
        let reduce_chain = Arc::new(MockChain::new("reduce", "Reduced result"));

        let mr_chain = MapReduceChain::new(map_chain, reduce_chain);

        let items = json!(["single_item"]);
        let result = mr_chain
            .run(&items)
            .await
            .expect("MapReduce chain execution should succeed");
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Reduced result"
        );
    }

    #[tokio::test]
    async fn test_chain_context_basic() {
        let mut context = ChainContext::new();
        context.add_step("key", json!("value"));

        assert_eq!(context.get_step("key"), Some(&json!("value")));
    }

    #[tokio::test]
    async fn test_chain_context_overwrite() {
        let mut context = ChainContext::new();
        context.add_step("key", json!("value1"));
        context.add_step("key", json!("value2"));

        // Both steps are stored, get_step returns the first match
        assert_eq!(context.get_step("key"), Some(&json!("value1")));
    }

    #[tokio::test]
    async fn test_chain_context_missing_key() {
        let context = ChainContext::new();
        assert_eq!(context.get_step("nonexistent"), None);
    }

    #[tokio::test]
    async fn test_chain_context_multiple_keys() {
        let mut context = ChainContext::new();
        context.add_step("key1", json!("value1"));
        context.add_step("key2", json!(42));
        context.add_step("key3", json!({"nested": "object"}));

        assert_eq!(context.get_step("key1"), Some(&json!("value1")));
        assert_eq!(context.get_step("key2"), Some(&json!(42)));
        assert_eq!(context.get_step("key3"), Some(&json!({"nested": "object"})));
    }

    #[tokio::test]
    async fn test_chain_with_context() {
        let chain = Arc::new(MockChain::new("test", "Response"));

        let mut context = ChainContext::new();
        context.add_step("shared", json!("data"));

        let result = chain
            .run_with_context(&json!({"input": "test"}), context)
            .await
            .expect("Chain execution with context should succeed");

        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Response"
        );
    }

    #[tokio::test]
    async fn test_sequential_chain_with_context() {
        let chain1 = MockChain::new("process", "Result");
        let chain2 = MockChain::new("process2", "Result2");

        let seq_chain = ComposableChain::new(chain1) >> ComposableChain::new(chain2);

        let context = ChainContext::new().with_metadata("config", "value");

        let result = seq_chain
            .run_with_context(&json!({"input": "test"}), context)
            .await
            .expect("Sequential chain execution with context should succeed");

        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Result2"
        );
    }

    #[tokio::test]
    async fn test_chain_result_structure() {
        let chain = MockChain::new("test", "Test response");
        let result = chain
            .run(&json!({}))
            .await
            .expect("Chain execution should succeed");

        // Should have output and context
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Test response"
        );
        assert!(result.context.intermediate_steps.is_empty());
    }

    #[tokio::test]
    async fn test_chain_error_handling() {
        let chain = ErrorChain;
        let result = chain.run(&json!({})).await;

        assert!(result.is_err());
        match result {
            Err(ChainError::ExecutionFailed(msg)) => {
                assert_eq!(msg, "Mock error");
            }
            _ => panic!("Expected ExecutionFailed"),
        }
    }

    #[tokio::test]
    async fn test_sequential_chain_error_propagation() {
        let good_chain = MockChain::new("good", "OK");
        let bad_chain = ErrorChain;

        let seq_chain = ComposableChain::new(good_chain) >> ComposableChain::new(bad_chain);

        let result = seq_chain.run(&json!({"input": "test"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_map_reduce_with_batch_size() {
        let map_chain = Arc::new(MockChain::new("map", "Mapped"));
        let reduce_chain = Arc::new(MockChain::new("reduce", "Reduced"));

        let mr_chain = MapReduceChain::new(map_chain, reduce_chain).with_batch_size(5);

        let items = json!(vec!["a"; 10]);
        let result = mr_chain
            .run(&items)
            .await
            .expect("MapReduce chain execution should succeed");
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "Reduced"
        );
    }

    #[tokio::test]
    async fn test_chain_output() {
        let chain = MockChain::new("test", "output");
        let result = chain
            .run(&json!({"input": "test"}))
            .await
            .expect("Chain execution should succeed");

        assert!(result.output.is_string());
        assert_eq!(result.output, json!("output"));
        assert!(result.context.intermediate_steps.is_empty());
    }

    #[tokio::test]
    async fn test_sequential_chain_builder() {
        // Test sequential composition using >> operator
        let chain1 = MockChain::new("1", "first");
        let chain2 = MockChain::new("2", "second");
        let chain3 = MockChain::new("3", "third");

        let seq = ComposableChain::new(chain1)
            >> ComposableChain::new(chain2)
            >> ComposableChain::new(chain3);

        let result = seq
            .run(&json!({}))
            .await
            .expect("Sequential chain execution should succeed");
        assert_eq!(
            result.output.as_str().expect("output should be a string"),
            "third"
        );
    }
}
