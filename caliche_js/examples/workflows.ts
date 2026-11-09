/**
 * Workflow Orchestration Example
 *
 * Demonstrates how to create and execute DAG-based workflows with
 * conditional branching, parallel execution, loops, and human-in-the-loop.
 *
 * bun run examples/workflow-orchestration.ts
 */

import { WorkflowBuilder, WorkflowEngine } from "../src/index";

async function main() {
  console.log("Workflow Orchestration Example\n");

  // Build a workflow
  const workflow = await WorkflowBuilder.create("article_workflow")
    // Add agent nodes
    .addAgent("research", "Research the topic: {{topic}}. Provide key findings.")
    .addAgent("write", "Write a comprehensive article based on: {{research_output}}")
    .addAgent("review", "Review and improve the article: {{write_output}}")
    // Add a condition node
    .addCondition("needs_revision", "{{review_score}} < 0.8")
    // Add a loop for revisions
    .addLoop(
      "revision_loop",
      3,
      "{{needs_revision}}",
      "Revise the article based on feedback: {{feedback}}"
    )
    // Add a human-in-the-loop checkpoint
    .addHumanInterrupt(
      "final_approval",
      "Please review the final article and approve or request changes.",
      ["approved", "feedback"]
    )
    // Connect nodes
    .connectNodes("research", "write")
    .connectNodes("write", "review")
    .connectNodes("review", "needs_revision")
    .connectNodes("needs_revision", "revision_loop", "needs_revision == true")
    .connectNodes("needs_revision", "final_approval", "needs_revision == false")
    .connectNodes("revision_loop", "review")
    // Set entry point
    .setEntryPoint("research")
    .build();

  // Create workflow engine
  const engine = WorkflowEngine.create();

  // Register workflow
  await engine.registerWorkflow(workflow);

  // Get workflow ID - workflows are registered by their internal ID
  const workflowId = workflow.getId();

  // Start execution
  console.log("Starting workflow execution...");
  const executionId = await engine.startExecution(workflowId);
  console.log(`Execution started with ID: ${executionId}`);

  // Wait a bit for execution to progress
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Check execution state
  const state = await engine.getExecutionState(executionId);
  console.log("Workflow state:", state);

  // Get execution result if completed
  if (state === "completed") {
    const result = await engine.getExecutionResult(executionId);
    console.log("Output:", JSON.stringify(result, null, 2));
  }

  // Check for pending interrupts
  const interrupts = await engine.getPendingInterrupts();
  if (interrupts.length > 0) {
    console.log("\nPending interrupts:");
    for (const interrupt of interrupts) {
      console.log(`  - ${interrupt.node_name}: ${interrupt.message}`);
      console.log(`    Interrupt ID: ${interrupt.interrupt_id}`);
      console.log(`    Execution ID: ${interrupt.execution_id}`);
    }

    // Resume with human input
    const firstInterrupt = interrupts[0];
    if (firstInterrupt) {
      console.log("\nResuming with approval...");
      await engine.resumeExecution(firstInterrupt.interrupt_id, {
        approved: true,
        feedback: "Looks good!",
      });

      // Wait for completion
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Check final state
      const finalState = await engine.getExecutionState(executionId);
      console.log("Final state:", finalState);
    }
  }
}

main().catch(console.error);
