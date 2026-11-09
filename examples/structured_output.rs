//! Example demonstrating structured output parsing and native schema enforcement.
//!
//! Run with:
//! ```bash
//! cargo run --example structured_output --features output-parsing
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use caliche::AgentBuilder;
#[cfg(feature = "output-parsing")]
use caliche::output::{
    EnumOutputParser, JsonOutputParser, ListOutputParser, OutputParser, StructuredOutputParser,
};
use caliche::prelude::ProviderType;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Person {
    name: String,
    age: u32,
    city: String,
}

#[derive(Debug, Deserialize)]
struct Analysis {
    sentiment: String,
    confidence: f32,
    key_points: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Structured Output Parsing Examples ===\n");

    // 1. Basic JSON parsing
    println!("1. JSON Output Parser:");
    let json_parser = JsonOutputParser::new();

    let llm_output = r#"{"name": "Alice", "age": 30, "city": "NYC"}"#;
    let value = json_parser.parse(llm_output).await?;
    println!("   Parsed JSON: {:?}\n", value);

    // 2. JSON from markdown code block
    println!("2. JSON from Markdown:");
    let markdown_output = r#"```json
{
  "name": "Bob",
  "age": 25,
  "city": "SF"
}
```"#;

    let value = json_parser.parse(markdown_output).await?;
    println!("   Parsed JSON: {:?}\n", value);

    // 3. Structured output with type safety
    println!("3. Structured Output Parser:");
    let struct_parser = StructuredOutputParser::<Person>::new();

    let llm_output = r#"{"name": "Charlie", "age": 35, "city": "LA"}"#;
    let person = struct_parser.parse(llm_output).await?;
    println!(
        "   Parsed Person: {} is {} years old from {}\n",
        person.name, person.age, person.city
    );

    // 4. Complex structure
    println!("4. Complex Structured Output:");
    let analysis_parser = StructuredOutputParser::<Analysis>::new();

    let llm_output = r#"{
        "sentiment": "positive",
        "confidence": 0.89,
        "key_points": ["Great product", "Fast shipping", "Good value"]
    }"#;

    let analysis = analysis_parser.parse(llm_output).await?;
    println!(
        "   Sentiment: {} (confidence: {:.2})",
        analysis.sentiment, analysis.confidence
    );
    println!("   Key points: {:?}\n", analysis.key_points);

    // 5. List output parser
    println!("5. List Output Parser:");
    let list_parser = ListOutputParser::new();

    let llm_output = "apple, banana, cherry, date, elderberry";
    let items = list_parser.parse(llm_output).await?;
    println!("   Parsed list: {:?}\n", items);

    // 6. Enum/choice parser
    println!("6. Enum Output Parser:");
    let enum_parser = EnumOutputParser::new(vec![
        "positive".to_string(),
        "negative".to_string(),
        "neutral".to_string(),
    ]);

    let llm_output = "positive";
    let sentiment = enum_parser.parse(llm_output).await?;
    println!("   Parsed sentiment: {}\n", sentiment);

    // 7. Format instructions
    println!("7. Format Instructions:");
    println!(
        "   JSON Parser instructions:\n   {}\n",
        json_parser.get_format_instructions()
    );
    println!(
        "   List Parser instructions:\n   {}\n",
        list_parser.get_format_instructions()
    );
    println!(
        "   Enum Parser instructions:\n   {}\n",
        enum_parser.get_format_instructions()
    );

    println!("All structured output examples completed!");

    // 8. Native structured output with schema enforcement (requires API key)
    println!("8. Native Structured Output with Schema Enforcement:");

    // Check if we have an API key (optional - skip if not available)
    if std::env::var("OPENAI_API_KEY").is_ok() || std::env::var("AGENT_PROVIDER").is_ok() {
        println!("   Testing native schema enforcement...");

        match AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4o")
            .build()
            .await
        {
            Ok(mut agent) => {
                let schema = json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "city": {"type": "string"}
                    },
                    "required": ["name", "age", "city"]
                });

                match agent
                    .chat_structured::<Person>(
                        "Extract person info: Alice is 30 years old and lives in NYC",
                        schema,
                    )
                    .await
                {
                    Ok((person, usage)) => {
                        println!(
                            "   Parsed Person: {} is {} years old from {}",
                            person.name, person.age, person.city
                        );
                        if let Some(usage) = usage {
                            println!("   Tokens used: {:?}", usage);
                        }
                    }
                    Err(e) => {
                        println!("   Warning: Failed to get structured output: {}", e);
                        println!(
                            "   (This is expected if API key is invalid or model doesn't support JSON schema)"
                        );
                    }
                }
            }
            Err(e) => {
                println!("   Warning: Failed to create agent: {}", e);
                println!("   (Set OPENAI_API_KEY or AGENT_PROVIDER to test this feature)");
            }
        }
    } else {
        println!("   (Skipping - requires OPENAI_API_KEY or AGENT_PROVIDER env var)");
        println!("\n   Example usage:");
        println!("   ```rust");
        println!("   let mut agent = AgentBuilder::new()");
        println!("       .provider(ProviderType::OpenAI)");
        println!("       .model(\"gpt-4o\")");
        println!("       .json_schema(\"Person\", schema)");
        println!("       .build().await?;");
        println!("   ");
        println!("   let (person, usage) = agent.chat_structured::<Person>(");
        println!("       \"Extract person info: Alice is 30\",");
        println!("       schema,");
        println!("   ).await?;");
        println!("   ```");
    }

    println!("\nAll examples completed!");
    println!("\nNote: For native structured output with schema enforcement,");
    println!("      use AgentBuilder.json_schema() or AgentService.chat_structured()");

    Ok(())
}
