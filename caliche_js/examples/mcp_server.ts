/**
 * Example: Creating a custom MCP server with TypeScript/JavaScript
 *
 * This example demonstrates how to create a custom MCP server that exposes
 * custom tools via the Model Context Protocol.
 *
 * bun run examples/mcp_server.ts
 */

import { McpServerBuilder, type McpTool } from "../src/mcp";

/**
 * A simple calculator tool for MCP.
 */
const calculatorTool: McpTool = {
  name: "calculator",
  description: "Performs basic arithmetic operations (add, subtract, multiply, divide)",
  parameters: {
    type: "object",
    properties: {
      operation: {
        type: "string",
        enum: ["add", "subtract", "multiply", "divide"],
        description: "The arithmetic operation to perform",
      },
      a: {
        type: "number",
        description: "First number",
      },
      b: {
        type: "number",
        description: "Second number",
      },
    },
    required: ["operation", "a", "b"],
  },
  execute: async (args: Record<string, unknown>) => {
    const operation = args.operation as string;
    const a = args.a as number;
    const b = args.b as number;

    try {
      let result: number;
      switch (operation) {
        case "add":
          result = a + b;
          break;
        case "subtract":
          result = a - b;
          break;
        case "multiply":
          result = a * b;
          break;
        case "divide":
          if (b === 0) {
            return { success: false, content: "", error: "Division by zero" };
          }
          result = a / b;
          break;
        default:
          return { success: false, content: "", error: `Unknown operation: ${operation}` };
      }

      return {
        success: true,
        content: `Result: ${result}`,
      };
    } catch (error) {
      return {
        success: false,
        content: "",
        error: error instanceof Error ? error.message : String(error),
      };
    }
  },
};

/**
 * A mock weather tool for MCP.
 */
const weatherTool: McpTool = {
  name: "get_weather",
  description: "Gets the current weather for a location (mock implementation)",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "The location to get weather for",
      },
    },
    required: ["location"],
  },
  execute: async (args: Record<string, unknown>) => {
    const location = args.location as string;
    // Mock weather data
    return {
      success: true,
      content: `Weather in ${location}: Sunny, 72°F`,
    };
  },
};

/**
 * Create and run an MCP server.
 */
async function main() {
  // Create server builder
  const server = await new McpServerBuilder()
    .name("example-mcp-server")
    .version("1.0.0")
    .enableTools(true)
    .enablePrompts(false)
    .enableResources(false)
    .addTool(calculatorTool)
    .addTool(weatherTool)
    .build();

  console.error("Starting MCP server on stdio...");
  console.error("");
  console.error("=".repeat(70));
  console.error("MCP Client Configuration");
  console.error("=".repeat(70));
  console.error("");
  console.error("Add this to your MCP client configuration (e.g., Claude Desktop):");
  console.error("");

  // Get the script path - use process.argv[1] which contains the script path
  const path = require("node:path");
  // For Bun, process.argv[1] contains the script path
  const scriptPath = process.argv[1] || path.resolve(__dirname || ".", "mcp_server.ts");
  const config = {
    mcpServers: {
      "example-mcp-server": {
        command: "bun",
        args: ["run", scriptPath],
      },
    },
  };

  console.error(JSON.stringify(config, null, 2));
  console.error("");
  console.error("=".repeat(70));
  console.error("Send JSON-RPC requests to stdin");

  // Run the server (this blocks until stdin is closed)
  await server.runStdio();
}

// Run if this is the main module
if (require.main === module) {
  main().catch((error) => {
    console.error("Error running MCP server:", error);
    process.exit(1);
  });
}

export { main };
