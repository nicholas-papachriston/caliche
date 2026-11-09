/**
 * Tools Example
 *
 * This example demonstrates:
 * - Built-in tools (calculator, echo, file operations, HTTP, etc.)
 * - Implementing custom tools using the Tool interface
 * - Using toolkits to bundle related tools
 * - Tool credentials management
 * - LangChain tools integration
 *
 * Best practices shown:
 * - Proper error handling in tool execution
 * - Clear tool descriptions and parameters
 * - Explicit tool selection for security
 * - Credential management patterns
 *
 * Run with:
 * ```bash
 * bun run examples/tools.ts
 * ```
 */

import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import {
  type Agent,
  AgentBuilder,
  createTool,
  LangChainAdapter,
  type Tool,
  type Toolkit,
  type ToolResult,
  waitForPendingTasks,
} from "../src/index";

// Example 1: Custom Tools
const reverseStringTool: Tool = {
  name: "reverse_string",
  description:
    "Reverses a string. Use this tool to reverse any text string. Takes a 'text' parameter and returns the reversed text.",
  parameters: {
    type: "object",
    properties: {
      text: {
        type: "string",
        description: "The text to reverse",
      },
    },
    required: ["text"],
  },
  execute: async (args: Record<string, unknown>): Promise<ToolResult> => {
    const text = args.text || "";
    if (!text) {
      return {
        success: false,
        content: "",
        error: "Missing 'text' parameter",
      };
    }

    const textStr = typeof text === "string" ? text : String(text);
    const reversed = textStr.split("").reverse().join("");
    return {
      success: true,
      content: JSON.stringify({
        original: text,
        reversed: reversed,
      }),
    };
  },
};

const wordCountTool: Tool = {
  name: "count_words",
  description:
    "Counts the number of words, characters, and lines in a text string. Use this tool to analyze text statistics.",
  parameters: {
    type: "object",
    properties: {
      text: {
        type: "string",
        description: "The text to analyze",
      },
    },
    required: ["text"],
  },
  execute: async (args: Record<string, unknown>): Promise<ToolResult> => {
    const text = args.text || "";
    if (!text) {
      return {
        success: false,
        content: "",
        error: "Missing 'text' parameter",
      };
    }

    const textStr = typeof text === "string" ? text : String(text);
    const words = textStr.split(/\s+/).filter((w: string) => w.length > 0);
    const wordCount = words.length;
    const charCount = textStr.length;
    const lineCount = textStr.split("\n").length;

    return {
      success: true,
      content: JSON.stringify({
        text: text,
        words: wordCount,
        characters: charCount,
        lines: lineCount,
      }),
    };
  },
};

// Example 2: Advanced Custom Tools
const weatherTool: Tool = {
  name: "get_weather",
  description:
    "Get current weather for a location. Use this tool to retrieve weather information for any city.",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "City name or location",
      },
      units: {
        type: "string",
        enum: ["celsius", "fahrenheit"],
        description: "Temperature units",
        default: "celsius",
      },
    },
    required: ["location"],
  },
  execute: async (args: Record<string, unknown>): Promise<ToolResult> => {
    const location = args.location as string;
    const units = (args.units as string) || "celsius";

    try {
      // Simulate API call (in real implementation, call actual weather API)
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Mock weather data
      const mockWeather = {
        location,
        temperature: units === "celsius" ? 22 : 72,
        units,
        condition: "sunny",
        humidity: 65,
      };

      return {
        success: true,
        content: JSON.stringify(mockWeather),
      };
    } catch (error) {
      return {
        success: false,
        content: "",
        error: `Failed to get weather: ${error}`,
      };
    }
  },
};

// Example 3: Custom Toolkit
class CalculatorToolkit implements Toolkit {
  readonly name = "calculator";

  get_tools(): Tool[] {
    return [
      {
        name: "add",
        description: "Adds two numbers together. Use this tool to perform addition.",
        parameters: {
          type: "object",
          properties: {
            a: { type: "number", description: "First number" },
            b: { type: "number", description: "Second number" },
          },
          required: ["a", "b"],
        },
        async execute(args: Record<string, unknown>): Promise<ToolResult> {
          const a = args.a as number;
          const b = args.b as number;
          const result = a + b;
          return {
            success: true,
            content: JSON.stringify({ result }),
          };
        },
      },
      {
        name: "multiply",
        description: "Multiplies two numbers together. Use this tool to perform multiplication.",
        parameters: {
          type: "object",
          properties: {
            a: { type: "number", description: "First number" },
            b: { type: "number", description: "Second number" },
          },
          required: ["a", "b"],
        },
        async execute(args: Record<string, unknown>): Promise<ToolResult> {
          const a = args.a as number;
          const b = args.b as number;
          const result = a * b;
          return {
            success: true,
            content: JSON.stringify({ result }),
          };
        },
      },
    ];
  }
}

async function builtInTools(): Promise<void> {
  console.log("=== Example 1: Built-in Tools ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .enableTool("calculator")
    .enableTool("fs")
    .build();

  console.log("Agent created with built-in tools enabled");

  try {
    const response = await agent.chat("What is 42 multiplied by 17?");
    console.log(`\nQuery: What is 42 multiplied by 17?`);
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function customTools(): Promise<void> {
  console.log("=== Example 2: Custom Tools ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .addTool(reverseStringTool)
    .addTool(wordCountTool)
    .enableTool("calculator")
    .build();

  try {
    // Use explicit prompts that mention the tool names
    const response1 = await agent.chat(
      "Use the reverse_string tool to reverse this text: 'Hello, World!'"
    );
    console.log("Query: Use the reverse_string tool to reverse this text: 'Hello, World!'");
    console.log(`Response: ${response1.response}\n`);

    const response2 = await agent.chat(
      "Use the count_words tool to count the words, characters, and lines in this text: 'The quick brown fox jumps over the lazy dog'"
    );
    console.log("Query: Use the count_words tool to analyze the text");
    console.log(`Response: ${response2.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function advancedTools(): Promise<void> {
  console.log("=== Example 3: Advanced Custom Tools ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .addTool(weatherTool)
    .build();

  try {
    // Use explicit prompt mentioning the tool name
    const response = await agent.chat(
      "Use the get_weather tool to get the weather in San Francisco"
    );
    console.log("Query: Use the get_weather tool to get the weather in San Francisco");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function toolkits(): Promise<void> {
  console.log("=== Example 4: Toolkits ===\n");

  const toolkit = new CalculatorToolkit();
  console.log(`Toolkit name: ${toolkit.name}`);
  console.log(
    `Tools in toolkit: ${toolkit
      .get_tools()
      .map((t) => t.name)
      .join(", ")}\n`
  );

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("bedrock")
      .model("anthropic.claude-3-sonnet-20240229-v1:0")
      .registerToolkit(toolkit)
      .build();

    console.log("Toolkit registered successfully");
    console.log(`Agent created: ${agent.provider}, ${agent.model}\n`);

    // Use explicit prompt mentioning the tool name
    const response = await agent.chat("Use the add tool to calculate 5 + 3");
    console.log("Query: Use the add tool to calculate 5 + 3");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function zodTools(): Promise<void> {
  console.log("=== Example 5: Zod Schema Tools ===\n");

  // Create tools using Zod schemas with createTool helper
  // This automatically converts Zod schemas to JSON Schema and validates inputs
  const weatherZodSchema = z.object({
    city: z.string().describe("The city name"),
    unit: z.enum(["celsius", "fahrenheit"]).default("celsius").describe("Temperature unit"),
  });

  const weatherZodTool = createTool({
    name: "get_weather_zod",
    description:
      "Get current weather for a city using Zod schema. Use this tool to retrieve weather information.",
    schema: weatherZodSchema,
    execute: async (args) => {
      // args is typed based on the Zod schema
      const { city, unit } = args;

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 100));

      const mockWeather = {
        location: city,
        temperature: unit === "celsius" ? 22 : 72,
        units: unit,
        condition: "sunny",
        humidity: 65,
      };

      return {
        success: true,
        content: JSON.stringify(mockWeather),
      };
    },
  });

  const mathZodSchema = z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
    operation: z.enum(["add", "subtract", "multiply", "divide"]).describe("Operation to perform"),
  });

  const mathZodTool = createTool({
    name: "math_zod",
    description:
      "Perform basic math operations (add, subtract, multiply, divide) on two numbers. Use this tool for mathematical calculations.",
    schema: mathZodSchema,
    execute: async (args) => {
      const { a, b, operation } = args;

      switch (operation) {
        case "add": {
          const result = a + b;
          return {
            success: true,
            content: JSON.stringify({ result, operation, a, b }),
          };
        }
        case "subtract": {
          const result = a - b;
          return {
            success: true,
            content: JSON.stringify({ result, operation, a, b }),
          };
        }
        case "multiply": {
          const result = a * b;
          return {
            success: true,
            content: JSON.stringify({ result, operation, a, b }),
          };
        }
        case "divide": {
          if (b === 0) {
            return {
              success: false,
              content: "",
              error: "Division by zero",
            };
          }
          const result = a / b;
          return {
            success: true,
            content: JSON.stringify({ result, operation, a, b }),
          };
        }
        default:
          return {
            success: false,
            content: "",
            error: `Unknown operation: ${operation}`,
          };
      }
    },
  });

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("openai")
      .model("gpt-3.5-turbo")
      .addTool(weatherZodTool)
      .addTool(mathZodTool)
      .build();

    console.log("Tools created with Zod schemas");
    console.log(
      `Weather tool parameters: ${JSON.stringify(weatherZodTool.parameters, null, 2)}\n`
    );

    // Use explicit prompts that mention the tool names
    const response1 = await agent.chat(
      "Use the get_weather_zod tool to get the weather in Paris in celsius"
    );
    console.log("Query: Use the get_weather_zod tool to get the weather in Paris");
    console.log(`Response: ${response1.response}\n`);

    const response2 = await agent.chat("Use the math_zod tool to multiply 15 by 23");
    console.log("Query: Use the math_zod tool to multiply 15 by 23");
    console.log(`Response: ${response2.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function langchainTools(): Promise<void> {
  console.log("=== Example 6: LangChain Tools ===\n");

  // Create a LangChain tool with Zod schema
  const calculatorTool = new DynamicStructuredTool({
    name: "calculator",
    description: "Performs basic arithmetic operations. Input should be a mathematical expression.",
    schema: z.object({
      expression: z
        .string()
        .describe("Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"),
    }),
    func: async ({ expression }: { expression: string }) => {
      try {
        // Simple safe evaluation (in production, use a proper math parser)
        const sanitized = expression.replace(/[^0-9+\-*/().\s]/g, "");
        const result = Function(`"use strict"; return (${sanitized})`)();
        return String(result);
      } catch (error) {
        return `Error: Invalid expression - ${error}`;
      }
    },
  });

  const reverseStringLangChainTool = new DynamicStructuredTool({
    name: "reverse_string",
    description: "Reverses a string",
    schema: z.object({
      text: z.string().describe("The text to reverse"),
    }),
    func: async ({ text }: { text: string }) => {
      return text.split("").reverse().join("");
    },
  });

  // Wrap LangChain tools with Caliche adapter
  const calculatorAdapter = new LangChainAdapter(calculatorTool);
  const reverseAdapter = new LangChainAdapter(reverseStringLangChainTool);

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("openai")
      .model("gpt-3.5-turbo")
      .addTool(calculatorAdapter)
      .addTool(reverseAdapter)
      .build();

    const response1 = await agent.chat("What is 15 * 23? Use the calculator tool.");
    console.log("Query: What is 15 * 23?");
    console.log(`Response: ${response1.response}\n`);

    const response2 = await agent.chat("Please reverse this text: 'Hello, World!'");
    console.log("Query: Please reverse this text: 'Hello, World!'");
    console.log(`Response: ${response2.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function multipleTools(): Promise<void> {
  console.log("=== Example 7: Multiple Tools ===\n");

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("bedrock")
      .model("anthropic.claude-3-sonnet-20240229-v1:0")
      .addTool(reverseStringTool)
      .addTool(wordCountTool)
      .addTool(weatherTool)
      .enableTool("calculator")
      .build();

    // Use explicit prompt mentioning all tool names
    const response = await agent.chat(
      "First use the calculator tool to calculate 5 * 7, then use the reverse_string tool to reverse the result as text, and finally use the get_weather tool to get the weather in Tokyo"
    );
    console.log("Query: Combine multiple tools (calculator, reverse_string, get_weather)");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function main(): Promise<void> {
  console.log("=== Tools Examples ===\n");

  await builtInTools();
  await customTools();
  await advancedTools();
  await toolkits();
  await zodTools();
  await langchainTools();
  await multipleTools();

  console.log("=== Examples Complete ===");
  console.log("\nTips:");
  console.log("- Use built-in tools when possible for reliability");
  console.log("- Create custom tools for domain-specific functionality");
  console.log("- Use createTool() with Zod schemas for type-safe tool creation");
  console.log("- Use toolkits to bundle related tools together");
  console.log("- LangChain tools can be integrated via LangChainAdapter");
  console.log("- Always validate tool parameters and handle errors gracefully");
}

main()
  .then(async () => {
    // Wait for any pending async tasks to complete before exiting
    try {
      await waitForPendingTasks();
    } catch (_error) {
      // waitForPendingTasks might not be available in all builds
      // Ignore the error and continue
    }
    process.exit(0);
  })
  .catch((error) => {
    console.error("Error:", error);
    process.exit(1);
  });
