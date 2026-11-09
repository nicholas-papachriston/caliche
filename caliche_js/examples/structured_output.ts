/**
 * Example demonstrating native structured output with schema enforcement
 *
 * Run with:
 * ```bash
 * bun run examples/structured_output.ts
 * ```
 */

import { AgentBuilder } from "../src/index";

interface Person {
  name: string;
  age: number;
  email: string;
}

async function main() {
  console.log("=== Native Structured Output Examples ===\n");

  // Example 1: Using json_mode() for any valid JSON
  console.log("1. JSON Mode (any valid JSON):");
  try {
    const agent1 = await new AgentBuilder().provider("openai").model("gpt-4o").jsonMode().build();

    const response1 = await agent1.chat("Return a JSON object with name and age");
    console.log("   Response:", response1.response);

    // Extract JSON from markdown code blocks if present
    let jsonText = response1.response;
    const jsonMatch = response1.response.match(/```(?:json)?\s*(\{.*?\})\s*```/s);
    if (jsonMatch?.[1]) {
      jsonText = jsonMatch[1];
    }

    try {
      const parsed = JSON.parse(jsonText);
      console.log("   Parsed:", parsed);
    } catch (error) {
      console.log("   Warning: Failed to parse JSON:", (error as Error).message);
    }
  } catch (error) {
    console.log("   Error:", (error as Error).message);
    console.log("   (Set OPENAI_API_KEY to test this feature)");
  }

  console.log("");

  // Example 2: Using json_schema() for schema-enforced output
  console.log("2. JSON Schema Mode (schema-enforced):");
  try {
    const personSchema = {
      type: "object",
      properties: {
        name: { type: "string" },
        age: { type: "integer" },
        email: { type: "string", format: "email" },
      },
      required: ["name", "age", "email"],
    };

    const agent2 = await new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .jsonSchema("Person", personSchema)
      .build();

    const response2 = await agent2.chat(
      "Extract person info: John Doe is 30 years old, email john@example.com"
    );

    console.log("   Response:", response2.response);

    // Extract JSON from markdown code blocks if present
    let jsonText = response2.response;
    const jsonMatch = response2.response.match(/```(?:json)?\s*(\{.*?\})\s*```/s);
    if (jsonMatch?.[1]) {
      jsonText = jsonMatch[1];
    }

    try {
      const person: Person = JSON.parse(jsonText);
      console.log("   Parsed Person:", person);
      console.log(`   Name: ${person.name}, Age: ${person.age}, Email: ${person.email}`);
    } catch (error) {
      console.log("   Warning: Failed to parse JSON:", (error as Error).message);
      console.log("   (The model may not have followed the JSON schema format)");
    }
  } catch (error) {
    console.log("   Error:", (error as Error).message);
    console.log("   (Set OPENAI_API_KEY to test this feature)");
  }

  console.log("\nExamples completed!");
  console.log("\nNote: The agent will automatically retry if the response");
  console.log("      does not match the schema (up to 2 retries by default)");
}

main().catch(console.error);
