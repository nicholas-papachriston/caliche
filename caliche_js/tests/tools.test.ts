import { describe, expect, test } from "bun:test";
import { AgentBuilder, type Tool, type ToolResult } from "../src/index";
import type { JsonRecord, ToolArguments } from "../src/types";

describe("Custom Tools", () => {
  test("should accept a tool with all required properties", () => {
    const tool: Tool = {
      name: "reverse_string",
      description: "Reverses a string",
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
      execute: async (args: ToolArguments): Promise<ToolResult> => {
        const text = args.text || "";
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

    const builder = new AgentBuilder();
    builder.addTool(tool);
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should execute tool with parameters", async () => {
    const tool: Tool = {
      name: "add_numbers",
      description: "Adds two numbers",
      parameters: {
        type: "object",
        properties: {
          a: { type: "number" },
          b: { type: "number" },
        },
        required: ["a", "b"],
      },
      execute: async (args: ToolArguments): Promise<ToolResult> => {
        const a = args.a || 0;
        const b = args.b || 0;
        return {
          success: true,
          content: JSON.stringify({ result: Number(a) + Number(b) }),
        };
      },
    };

    const result = await tool.execute({ a: 5, b: 3 });
    expect((result as ToolResult).success).toBe(true);
    const content = JSON.parse((result as ToolResult).content as string);
    expect(content.result).toBe(8);
  });

  test("should handle tool error", async () => {
    const tool: Tool = {
      name: "failing_tool",
      description: "A tool that fails",
      parameters: {
        type: "object",
        properties: {},
      },
      execute: async (): Promise<ToolResult> => {
        return {
          success: false,
          content: "",
          error: "Tool execution failed",
        };
      },
    };

    const result = await tool.execute({});
    expect((result as ToolResult).success).toBe(false);
    expect((result as ToolResult).error).toBe("Tool execution failed");
  });

  test("should handle tool async execution", async () => {
    const tool: Tool = {
      name: "async_tool",
      description: "An async tool",
      parameters: {
        type: "object",
        properties: {},
      },
      execute: async (): Promise<ToolResult> => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return {
          success: true,
          content: "Async result",
        };
      },
    };

    const result = await tool.execute({});
    expect((result as ToolResult).success).toBe(true);
    expect((result as ToolResult).content).toBe("Async result");
  });

  test("should register multiple tools", () => {
    const tool1: Tool = {
      name: "tool1",
      description: "Tool 1",
      parameters: { type: "object", properties: {} },
      execute: async () => ({ success: true, content: "1" }),
    };

    const tool2: Tool = {
      name: "tool2",
      description: "Tool 2",
      parameters: { type: "object", properties: {} },
      execute: async () => ({ success: true, content: "2" }),
    };

    const builder = new AgentBuilder().addTool(tool1).addTool(tool2);
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should format tool result correctly", async () => {
    const tool: Tool = {
      name: "formatter",
      description: "Formats output",
      parameters: {
        type: "object",
        properties: {
          data: { type: "string" },
        },
        required: ["data"],
      },
      execute: async (args: ToolArguments): Promise<ToolResult> => {
        return {
          success: true,
          content: JSON.stringify({ formatted: `Formatted: ${args.data}` }),
        };
      },
    };

    const result = await tool.execute({ data: "test" });
    expect((result as ToolResult).success).toBe(true);
    const parsed = JSON.parse((result as ToolResult).content);
    expect(parsed.formatted).toBe("Formatted: test");
  });

  test("should enable built-in calculator tool", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .enableTool("calculator")
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should enable built-in echo tool", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .enableTool("fs")
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should accept a tool that returns a string", () => {
    const tool: Tool = {
      name: "fs",
      description: "Echoes back the input",
      parameters: {
        type: "object",
        properties: {
          message: { type: "string" },
        },
      },
      execute: async (args: ToolArguments): Promise<string> => {
        const message = args.message;
        return typeof message === "string" ? message : String(message || "");
      },
    };

    const builder = new AgentBuilder();
    builder.addTool(tool);
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should accept a tool that returns an object", () => {
    const tool: Tool = {
      name: "get_info",
      description: "Returns information",
      parameters: {
        type: "object",
        properties: {},
      },
      execute: async (): Promise<JsonRecord> => {
        return { success: true, data: "test" };
      },
    };

    const builder = new AgentBuilder();
    builder.addTool(tool);
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should accept multiple tools", () => {
    const tool1: Tool = {
      name: "tool1",
      description: "Tool 1",
      parameters: { type: "object", properties: {} },
      execute: async () => ({ success: true, content: "1" }),
    };

    const tool2: Tool = {
      name: "tool2",
      description: "Tool 2",
      parameters: { type: "object", properties: {} },
      execute: async () => ({ success: true, content: "2" }),
    };

    const builder = new AgentBuilder().addTool(tool1).addTool(tool2);
    expect(builder).toBeInstanceOf(AgentBuilder);
  });
});
