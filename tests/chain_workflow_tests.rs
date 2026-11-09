//! Tests for chain operation workflows

#[cfg(feature = "chains")]
mod tests {
    use caliche::AgentBuilder;
    use caliche::chains::{Chain, ComposableChain, LLMChain};
    use caliche::prelude::ProviderType;
    use caliche::prompts::PromptTemplate;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn test_llm_chain_workflow() {
        let agent = AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4o".to_string())
            .build()
            .await
            .expect("Failed to build agent");

        let agent_arc = Arc::new(Mutex::new(agent));
        let prompt = PromptTemplate::new("Summarize this text: {text}");
        let chain = LLMChain::new(agent_arc, prompt);

        let result = chain
            .run(&serde_json::json!({
                "text": "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety."
            }))
            .await
            .expect("Chain execution failed");

        let output_str = result
            .output
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(!output_str.is_empty());
        assert!(
            output_str.to_lowercase().contains("rust")
                || output_str.to_lowercase().contains("programming")
        );
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn test_sequential_chain_workflow() {
        let agent = AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4o".to_string())
            .build()
            .await
            .expect("Failed to build agent");

        let agent_arc = Arc::new(Mutex::new(agent));

        let summarize_chain =
            LLMChain::new(agent_arc.clone(), PromptTemplate::new("Summarize: {input}"));

        let analyze_chain = LLMChain::new(
            agent_arc.clone(),
            PromptTemplate::new("Analyze this summary: {output}"),
        );

        // Use >> operator for sequential composition
        let sequential =
            ComposableChain::new(summarize_chain) >> ComposableChain::new(analyze_chain);

        let result = sequential
            .run(&serde_json::json!({
                "input": "Long text about artificial intelligence and machine learning..."
            }))
            .await
            .expect("Sequential chain failed");

        // Sequential chain output should contain the final result
        assert!(result.output.is_object());
        // The output should have content from the analysis chain
        let output_str = result.output.to_string();
        assert!(!output_str.is_empty());
    }
}
