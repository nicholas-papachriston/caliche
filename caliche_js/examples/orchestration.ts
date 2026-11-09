/**
 * Agent Coordinator Example
 *
 * Demonstrates multi-agent coordination with load balancing,
 * capability-based routing, and inter-agent messaging.
 *
 * bun run examples/agent-coordinator.ts
 */

import { AgentBuilder, AgentCoordinator } from "../src/index";

async function main() {
  console.log("Agent Coordinator Example\n");

  // Create coordinator with load balancing strategy
  const coordinator = AgentCoordinator.withStrategy("least_connections");

  // Create specialized agents
  const researchAgent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4")
    .systemPrompt("You are a research specialist.")
    .build();

  const codingAgent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4")
    .systemPrompt("You are a coding specialist.")
    .build();

  const writingAgent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4")
    .systemPrompt("You are a writing specialist.")
    .build();

  // Register agents with capabilities
  const researchAgentId = await coordinator.registerAgent(
    "Research Agent",
    "Research Specialist",
    ["research", "data_analysis"],
    researchAgent
  );
  console.log(`Registered: ${researchAgentId}`);

  const codingAgentId = await coordinator.registerAgent(
    "Coding Agent",
    "Software Developer",
    ["coding"],
    codingAgent
  );
  console.log(`Registered: ${codingAgentId}`);

  const writingAgentId = await coordinator.registerAgent(
    "Writing Agent",
    "Technical Writer",
    ["writing"],
    writingAgent
  );
  console.log(`Registered: ${writingAgentId}\n`);

  // Delegate tasks based on capability
  console.log("Delegating research task...");
  const researchResult = await coordinator.delegateTask(
    "Research the latest trends in machine learning",
    "research"
  );
  console.log(`Agent ${researchResult.agent_id} responded:`);
  console.log(`${researchResult.response.substring(0, 200)}...\n`);

  console.log("Delegating coding task...");
  const codingResult = await coordinator.delegateTask(
    "Write a Python function to calculate fibonacci numbers",
    "coding"
  );
  console.log(`Agent ${codingResult.agent_id} responded:`);
  console.log(`${codingResult.response.substring(0, 200)}...\n`);

  // Send message between agents
  console.log("Sending message between agents...");
  await coordinator.sendMessage(
    researchAgentId,
    writingAgentId,
    "Here are the research findings: [findings]"
  );
  console.log("Message sent\n");

  // Broadcast to all agents
  console.log("Broadcasting announcement...");
  await coordinator.broadcast(
    researchAgentId,
    "Important: New research guidelines have been updated."
  );
  console.log("Broadcast sent\n");

  // Get coordinator statistics
  const stats = await coordinator.getStats();
  console.log("Coordinator Statistics:");
  console.log(`  Total agents: ${stats.total_agents}`);
  console.log(`  Total load: ${stats.total_load}`);
  console.log(`  Total capacity: ${stats.total_capacity}`);
  console.log("\n  Agent Stats:");
  for (const agentStat of stats.agent_stats) {
    console.log(
      `    ${agentStat.name}: ${agentStat.current_load}/${agentStat.max_concurrent_tasks}`
    );
  }

  // Get message bus statistics
  const messageBus = coordinator.getMessageBus();
  const busStats = await messageBus.getStats();
  console.log("\nMessage Bus Statistics:");
  console.log(`  Queue size: ${busStats.queue_size}`);
  console.log(`  Subscribers: ${busStats.subscriber_count}`);
}

main().catch(console.error);
