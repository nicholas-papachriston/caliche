/**
 * Caliche Prompt Templates Examples
 *
 * Demonstrates how to use prompt templates for reusable prompts with variable substitution.
 *
 * bun run examples/prompts.ts
 */

import { PromptTemplate } from "../src/index";

function exampleBasicTemplate() {
  console.log("=== Basic Prompt Template ===\n");

  const template = new PromptTemplate("Translate the following text to {language}: {text}");

  // Format with variables
  const formatted = template.format({
    language: "Spanish",
    text: "Hello, how are you?",
  });

  console.log("Variables:", template.variables());
  console.log("Formatted:", formatted);
  console.log();
}

function exampleFactoryTemplates() {
  console.log("=== Factory Template Methods ===\n");

  // Summarize template
  const summarizeTemplate = PromptTemplate.summarize();
  console.log("Summarize template variables:", summarizeTemplate.variables());
  console.log("Example:", summarizeTemplate.format({ text: "Sample text to summarize" }));
  console.log();

  // Analyze template
  const analyzeTemplate = PromptTemplate.analyze();
  console.log("Analyze template variables:", analyzeTemplate.variables());
  console.log("Example:", analyzeTemplate.format({ text: "Sample text to analyze" }));
  console.log();

  // Translate template
  const translateTemplate = PromptTemplate.translate();
  console.log("Translate template variables:", translateTemplate.variables());
  console.log("Example:", translateTemplate.format({ text: "Hello", language: "Spanish" }));
  console.log();
}

function exampleComplexTemplate() {
  console.log("=== Complex Template with Multiple Variables ===\n");

  const template = new PromptTemplate(
    "You are a {role} helping with {task}.\n\n" +
      "Context: {context}\n\n" +
      "Instructions: {instructions}\n\n" +
      "Please provide a {format} response."
  );

  const formatted = template.format({
    role: "data analyst",
    task: "analyzing sales data",
    context: "Q4 sales increased by 15% compared to Q3",
    instructions: "Identify key trends and provide insights",
    format: "structured summary",
  });

  console.log("Formatted prompt:");
  console.log(formatted);
  console.log();
}

function main() {
  exampleBasicTemplate();
  exampleFactoryTemplates();
  exampleComplexTemplate();
}

if (require.main === module) {
  main();
}
