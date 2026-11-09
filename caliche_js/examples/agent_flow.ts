/**
 * Agent Flow Example for Caliche TypeScript SDK
 *
 * This example demonstrates:
 * - Creating multiple specialized agents
 * - Building agent flows with FlowBuilder
 * - Executing multi-agent workflows with handoffs
 * - Generating Mermaid flow diagrams
 *
 * bun run examples/agent-flow.ts
 *
 * Note: This example requires provider credentials to be set in environment variables.
 */

import { type Agent, AgentBuilder, FlowBuilder, type FlowExecutionResult } from "../src/index";

async function main(): Promise<void> {
  console.log("=== Agent Flow Example ===\n");

  // Check if provider is configured
  if (!process.env.AGENT_PROVIDER && !process.env.AGENT_MODEL) {
    console.log("Note: Set AGENT_PROVIDER and AGENT_MODEL environment variables");
    console.log("      to run this example with actual agents.\n");
    console.log("Example:");
    console.log("  export AGENT_PROVIDER=bedrock");
    console.log("  export AGENT_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0\n");
    return;
  }

  try {
    // Create specialized agents
    console.log("1. Creating specialized agents...");

    // Research agent - specializes in research tasks
    const researchAgent: Agent = await new AgentBuilder()
      .systemPrompt(
        "You are a research specialist. When you need to write content, " +
          "transfer to the writer agent using transfer_to tool."
      )
      .enableTool("transfer_to")
      .build();

    console.log("   Research agent created");

    // Writer agent - specializes in writing
    const writerAgent: Agent = await new AgentBuilder()
      .systemPrompt(
        "You are a writing specialist. Focus on creating clear, " + "well-structured content."
      )
      .build();

    console.log("   Writer agent created\n");

    // Build the flow
    console.log("2. Building agent flow...");
    const flow = new FlowBuilder()
      .addNode("researcher", researchAgent)
      .addNode("writer", writerAgent)
      .setEntryPoint("researcher")
      .build();

    console.log(`   Flow entry point: ${flow.entryPoint}`);
    console.log(`   Available nodes: ${flow.nodeNames.join(", ")}\n`);

    // Generate Mermaid diagram
    console.log("3. Flow visualization (Mermaid):");
    const mermaid = flow.toMermaid();
    console.log(mermaid);
    console.log();

    // Execute the flow
    console.log("4. Executing flow...");
    const result: FlowExecutionResult = await flow.run(
      "Research the benefits of Rust programming language and write a summary"
    );

    console.log("\n=== Flow Execution Result ===");
    if (result.final_response) {
      console.log(`Final Response:\n${result.final_response}\n`);
    }

    console.log("Execution Log:");
    result.execution_log.forEach((step, i) => {
      console.log(`\nStep ${i + 1}: Agent '${step.agent_name}'`);
      console.log(`  Input: ${step.input}`);
      if (step.response) {
        console.log(`  Response: ${step.response}`);
      }
      if (step.handoff) {
        console.log(`  Handoff to: ${step.handoff.target_agent} (${step.handoff.reason})`);
      }
      if (step.usage.total_tokens) {
        console.log(`  Tokens: ${step.usage.total_tokens}`);
      }
    });
  } catch (error) {
    if (error instanceof Error) {
      console.error(`Error: ${error.message}`);
    } else {
      console.error("Unknown error:", error);
    }
    console.log("\nNote: This example requires valid provider credentials.");
  }

  console.log("\n=== Example Complete ===");
  console.log("\nTips:");
  console.log("- Use FlowBuilder to chain multiple agents together");
  console.log("- Agents can hand off to each other using transfer_to tool");
  console.log("- Use toMermaid() to visualize your agent flows");
}

main().catch(console.error);
