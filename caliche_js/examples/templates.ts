/**
 * Templates Example
 *
 * This example demonstrates:
 * - Basic prompt template usage
 * - Template composition and chaining (when available)
 * - Template variables and instantiation
 *
 * Best practices shown:
 * - Composing complex prompts from simple templates
 * - Parameterized template usage
 *
 * Run with:
 * ```bash
 * bun run examples/templates.ts
 * */

import { PromptTemplate } from "../src/index";

function basicTemplates(): void {
  console.log("=== Example 1: Basic Prompt Templates ===\n");

  try {
    // Basic template
    const template = new PromptTemplate("Analyze this data: {data}");
    const prompt = template.format({
      data: "Sales figures Q1-Q4: $1M, $1.2M, $1.5M, $1.8M",
    });

    console.log("Template:");
    console.log(`${prompt}\n`);

    // Template with multiple variables
    const multiTemplate = new PromptTemplate("You are a {role}. {instruction}");
    const prompt2 = multiTemplate.format({
      role: "helpful assistant",
      instruction: "Explain quantum computing",
    });

    console.log("Multi-variable Template:");
    console.log(`${prompt2}\n`);
  } catch (error) {
    console.log(`(Skipped: ${error})\n`);
  }
}

function templateReuse(): void {
  console.log("=== Example 2: Template Reuse ===\n");

  try {
    const base = new PromptTemplate("You are a {role}. {instruction}");

    const prompt1 = base.format({
      role: "helpful assistant",
      instruction: "Explain quantum computing",
    });

    const prompt2 = base.format({
      role: "helpful assistant",
      instruction: "Write a haiku",
    });

    console.log("Template Reuse:");
    console.log(`1. ${prompt1}`);
    console.log(`2. ${prompt2}\n`);
  } catch (error) {
    console.log(`(Skipped: ${error})\n`);
  }
}

function templateExamples(): void {
  console.log("=== Example 3: Template Examples ===\n");

  console.log("Template composition features:");
  console.log("  - Basic prompt templates with variables");
  console.log("  - Template chaining (when available)");
  console.log("  - Template search and browsing (when available)");
  console.log("  - Template storage (filesystem, S3, R2) (when available)");
  console.log("  - Prompt template storage (when available)");
  console.log();
  console.log("Note: Advanced template features are being added to JS bindings.");
  console.log("      Basic PromptTemplate is available now.\n");
}

async function main(): Promise<void> {
  console.log("=== Templates Examples ===\n");

  basicTemplates();
  templateReuse();
  templateExamples();

  console.log("=== Examples Complete ===");
  console.log("\nTips:");
  console.log("- Use PromptTemplate for parameterized prompts");
  console.log("- Reuse templates with different variable values");
  console.log("- Prefer PromptTemplate when the same skeleton is reused with new variables");
}

main().catch(console.error);
