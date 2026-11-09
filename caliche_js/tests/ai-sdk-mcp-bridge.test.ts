import { describe, expect, it } from "bun:test";
import { CalicheMcpBridge, type CalicheMcpServerBinding } from "../src/adapters/ai-sdk-mcp";

function createMockBinding(): CalicheMcpServerBinding {
  const server = {
    async listTools() {
      return [
        {
          name: "lookup",
          description: "Look up a value",
          input_schema: {
            type: "object",
            properties: {
              key: { type: "string" },
            },
            required: ["key"],
          },
        },
      ];
    },
    async callTool(name: string, args: Record<string, unknown>) {
      if (name !== "lookup") {
        return {
          is_error: true,
          content: [{ type: "text", text: `unknown:${name}` }],
        };
      }
      return {
        is_error: false,
        content: [{ type: "text", text: `value:${String(args.key ?? "")}` }],
      };
    },
  };

  return {
    name: "mock",
    server: server as any,
    resources: [
      {
        uri: "resource://mock/config",
        name: "Mock Config",
      },
    ],
    prompts: [
      {
        name: "mock_prompt",
      },
    ],
  };
}

describe("CalicheMcpBridge", () => {
  it("lists tools, resources, and prompts", async () => {
    const bridge = new CalicheMcpBridge([createMockBinding()]);

    const tools = await bridge.listTools();
    expect(tools).toHaveLength(1);
    expect(tools[0]?.name).toBe("mock__lookup");

    const resources = bridge.listResources();
    expect(resources[0]?.uri).toBe("resource://mock/config");

    const prompts = bridge.listPrompts();
    expect(prompts[0]?.name).toBe("mock_prompt");
  });

  it("invokes namespaced MCP tools and maps payloads", async () => {
    const bridge = new CalicheMcpBridge([createMockBinding()]);

    const success = await bridge.callTool("mock__lookup", { key: "alpha" });
    expect(success.success).toBe(true);
    expect(success.content).toBe("value:alpha");

    const failure = await bridge.callTool("mock__missing", {});
    expect(failure.success).toBe(false);
    expect(failure.error).toContain("unknown:missing");
  });

  it("returns agent tool wrappers for MCP tools", async () => {
    const bridge = new CalicheMcpBridge([createMockBinding()]);
    const tools = await bridge.asAgentTools();

    expect(tools).toHaveLength(1);
    expect(tools[0]?.name).toBe("mock__lookup");

    const result = await tools[0]?.execute({ key: "beta" });
    expect(result).toEqual({ success: true, content: "value:beta" });
  });
});
