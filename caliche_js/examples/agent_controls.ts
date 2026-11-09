/**
 * Agent Lifecycle Controls Example
 *
 * This example demonstrates advanced agent control features:
 * - Lifecycle hooks (PreToolUse/PostToolUse) for intercepting tool execution
 * - Permission modes (AcceptEdits, AcceptAll, DenyAll) for automatic permission handling
 * - Allowed tools filtering to restrict which tools can be used
 * - In-process MCP servers for direct tool registration without subprocess overhead
 * - Session forking for parallel work and subagents
 *
 * Run with:
 * ```bash
 * bun run examples/agent_controls.ts
 * ```
 */

import {
  AgentBuilder,
  AgentSession,
  type HookContext,
  InProcessMcpServerBuilder,
  type LifecycleHook,
  PermissionMode,
} from "../src/index";

async function exampleLifecycleHooks(): Promise<void> {
  console.log("=== Example 1: Lifecycle Hooks ===\n");

  // Example 1a: Synchronous hook
  const syncSecurityHook: LifecycleHook = {
    onPreToolUse(ctx: HookContext) {
      // Synchronous check - no async operations
      if (ctx.tool_name === "Bash") {
        const command = (ctx.tool_input as any).command;
        if (typeof command === "string" && command.includes("rm -rf")) {
          return {
            permission_decision: {
              decision: "deny",
              reason: "Dangerous command detected (sync check)",
            },
          };
        }
      }
      return { permission_decision: { decision: "allow" } };
    },
  };

  console.log("Synchronous hook example:");
  console.log("- Blocks dangerous Bash commands synchronously");
  console.log("- Returns HookResult directly (no Promise)");
  console.log("Hook implementation:", syncSecurityHook);
  console.log();

  // Example 1b: Async hook that performs async checks
  const asyncSecurityHook: LifecycleHook = {
    async onPreToolUse(ctx: HookContext) {
      // Simulate async check (e.g., database lookup, API call, external validation)
      await new Promise((resolve) => setTimeout(resolve, 10));

      if (ctx.tool_name === "Bash") {
        const command = (ctx.tool_input as any).command;
        if (typeof command === "string" && command.includes("rm -rf")) {
          return {
            permission_decision: {
              decision: "deny",
              reason: "Dangerous command detected (async check)",
            },
          };
        }
      }
      return { permission_decision: { decision: "allow" } };
    },

    async onPostToolUse(
      ctx: HookContext,
      result: { success: boolean; content: string; error?: string }
    ) {
      // Async logging after tool execution
      await new Promise((resolve) => setTimeout(resolve, 5));
      console.log(`Tool ${ctx.tool_name} executed: ${result.success ? "success" : "failed"}`);
      return { permission_decision: { decision: "allow" } };
    },
  };

  console.log("Async hook example:");
  console.log("- Performs async validation (e.g., database lookup, API call)");
  console.log("- Returns Promise<HookResult>");
  console.log("- Includes async post-tool-use logging");
  console.log("Hook implementation:", asyncSecurityHook);
  console.log();

  // Example usage - both sync and async hooks are supported
  // Uncomment to test:
  /*
  import { HookMatcher } from "../src/index";
  
  // Using sync hook
  const agentSync = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .enableTool("calculator")
    .hook(HookMatcher.tool("Bash"), syncSecurityHook)
    .build();
  
  // Using async hook
  const agentAsync = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .enableTool("calculator")
    .hook(HookMatcher.tool("Bash"), asyncSecurityHook)
    .build();
  */
}

async function examplePermissionModes(): Promise<void> {
  console.log("=== Example 2: Permission Modes ===\n");

  // AcceptEdits mode - auto-accept file edits
  await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .permissionMode(PermissionMode.acceptEdits())
    .enableTool("fs")
    .build();

  console.log("Agent configured with AcceptEdits permission mode");
  console.log("(File edits will be auto-accepted)\n");

  // DenyAll mode - block all tool calls
  await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .permissionMode(PermissionMode.denyAll())
    .enableTool("calculator")
    .build();

  console.log("Agent configured with DenyAll permission mode");
  console.log("(All tool calls will be blocked)\n");
}

async function exampleAllowedTools(): Promise<void> {
  console.log("=== Example 3: Allowed Tools Filtering ===\n");

  const allowed = ["calculator", "fs"];
  const agentAllowedTools = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .allowedTools(allowed)
    .enableAllDefaultTools(); // Enable all, but only calculator and echo will be allowed

  const agent = await agentAllowedTools.build();

  console.log(`Agent configured with allowed_tools: ${JSON.stringify(allowed)}`);
  console.log("Available tools (allowed tools marked with *):");
  const tools = await agent.listTools();
  for (const tool of tools) {
    const marker = allowed.includes(tool.name) ? "*" : " ";
    console.log(`  ${marker} ${tool.name}`);
  }
  console.log();
}

async function exampleInProcessMcp(): Promise<void> {
  console.log("=== Example 4: In-Process MCP Server ===\n");

  // Create an in-process MCP server with custom tools
  const server = new InProcessMcpServerBuilder("my-tools")
    .version("1.0.0")
    .addTool({
      name: "greet",
      description: "Greet a user by name",
      parameters: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "Name of the person to greet",
          },
        },
        required: ["name"],
      },
      execute: (args: Record<string, unknown>) => {
        const name = (args.name as string) || "World";
        return {
          success: true,
          content: `Hello, ${name}!`,
        };
      },
    })
    .build();

  console.log(`Created in-process MCP server '${server.name}' v${server.version}`);
  const tools = await server.listTools();
  console.log("Tools in server:");
  for (const tool of tools) {
    console.log(`  - ${tool.name}: ${tool.description}`);
  }

  // Register the MCP server with an agent
  const agentWithMcp = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .mcpServer("my-tools", server)
    .build();

  console.log("\nAgent configured with in-process MCP server");
  const agentTools = await agentWithMcp.listTools();
  console.log("Available tools (including MCP server tools):");
  for (const tool of agentTools) {
    console.log(`  - ${tool.name}`);
  }
  console.log();
}

async function exampleSessionForking(): Promise<void> {
  console.log("=== Example 5: Session Forking ===\n");

  // Create a main session
  const mainSession = new AgentSession("main-session");
  mainSession.addMessage({
    role: "user",
    content: "Hello, I'm starting a task",
  });

  console.log(`Created main session: ${mainSession.id}`);
  const history = mainSession.getHistory();
  console.log(`Main session history: ${history.length} messages`);

  // Fork the session for parallel work
  const forkedSession = mainSession.fork("parallel-worker");
  console.log(`\nForked session: ${forkedSession.id}`);
  const forkedHistory = forkedSession.getHistory();
  console.log(`Forked session history: ${forkedHistory.length} messages (copied from main)`);

  // Add messages to the forked session
  forkedSession.addMessage({
    role: "user",
    content: "Working on parallel task",
  });
  forkedSession.addMessage({
    role: "assistant",
    content: "Completed parallel work",
  });

  const updatedForkedHistory = forkedSession.getHistory();
  console.log(`After adding messages to fork: ${updatedForkedHistory.length} messages`);

  // Merge the forked session back into main
  mainSession.merge(forkedSession);
  console.log("\nMerged fork back into main session");
  const finalHistory = mainSession.getHistory();
  console.log(`Main session history after merge: ${finalHistory.length} messages`);

  console.log("\nSession forking enables:");
  console.log("  - Parallel work with isolated state");
  console.log("  - Subagents that can work independently");
  console.log("  - Merging results back into main conversation");
  console.log();
}

async function exampleCombinedFeatures(): Promise<void> {
  console.log("=== Example 6: Combined Features ===\n");

  // Create a hook that works with all tools
  const combinedHook: LifecycleHook = {
    async onPreToolUse(ctx: HookContext) {
      // Log all tool usage
      console.log(`Tool about to be used: ${ctx.tool_name}`);
      return { permission_decision: { decision: "allow" } };
    },
  };

  await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .allowedTools(["calculator", "fs"])
    .permissionMode(PermissionMode.acceptEdits())
    // .hook(HookMatcher.all(), combinedHook) // Uncomment to enable hook
    .build();

  console.log("Agent configured with:");
  console.log("  - Allowed tools: [calculator, echo]");
  console.log("  - Permission mode: AcceptEdits");
  console.log("  - Lifecycle hook for all tools (async supported)");
  console.log("  Hook:", combinedHook);
  console.log();
}

async function main(): Promise<void> {
  try {
    await exampleLifecycleHooks();
    await examplePermissionModes();
    await exampleAllowedTools();
    await exampleInProcessMcp();
    await exampleSessionForking();
    await exampleCombinedFeatures();

    console.log("=== Examples Complete ===");
    console.log("\nKey Features:");
    console.log(
      "1. Lifecycle Hooks: Intercept and control tool execution (sync and async supported)"
    );
    console.log("2. Permission Modes: Control tool call permissions automatically");
    console.log("3. Allowed Tools: Restrict which tools can be used");
    console.log("4. In-Process MCP Servers: Direct tool registration without subprocess overhead");
    console.log("5. Session Forking: Enable parallel work and subagents");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main().catch(console.error);
