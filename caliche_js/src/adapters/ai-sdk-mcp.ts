import type { InProcessMcpServer } from "../mcp";
import type { Tool } from "../streaming/events";

export interface McpResourceDescriptor {
  readonly uri: string;
  readonly name?: string;
  readonly description?: string;
  readonly mimeType?: string;
}

export interface McpPromptDescriptor {
  readonly name: string;
  readonly description?: string;
}

export interface CalicheMcpServerBinding {
  readonly name: string;
  readonly server: InProcessMcpServer;
  readonly resources?: readonly McpResourceDescriptor[];
  readonly prompts?: readonly McpPromptDescriptor[];
  readonly transport?: "stdio" | "http";
  readonly url?: string;
}

interface McpToolDescriptor {
  readonly name: string;
  readonly description: string;
  readonly input_schema: Record<string, unknown>;
}

function parseToolResultText(
  payload: {
    content?: Array<{ type: string; text?: string; resource?: unknown }>;
    is_error?: boolean;
  } | null
): { success: boolean; content: string; error?: string } {
  if (!payload) {
    return { success: false, content: "", error: "MCP tool call returned no payload" };
  }

  const textParts: string[] = [];
  for (const part of payload.content ?? []) {
    if (part.type === "text" && typeof part.text === "string") {
      textParts.push(part.text);
      continue;
    }
    if (part.type === "resource" && part.resource !== undefined) {
      textParts.push(JSON.stringify(part.resource));
    }
  }

  const content = textParts.join("\n").trim();
  if (payload.is_error) {
    return { success: false, content: "", error: content || "MCP tool returned error response" };
  }

  return { success: true, content };
}

/**
 * Bridge Caliche in-process MCP servers to namespaced agent tools.
 */
export class CalicheMcpBridge {
  private readonly servers: readonly CalicheMcpServerBinding[];

  constructor(bindings: readonly CalicheMcpServerBinding[] = []) {
    this.servers = bindings;
  }

  hasServers(): boolean {
    return this.servers.length > 0;
  }

  listResources(): readonly McpResourceDescriptor[] {
    const resources: McpResourceDescriptor[] = [];
    for (const binding of this.servers) {
      for (const resource of binding.resources ?? []) {
        resources.push(resource);
      }
    }
    return resources;
  }

  listPrompts(): readonly McpPromptDescriptor[] {
    const prompts: McpPromptDescriptor[] = [];
    for (const binding of this.servers) {
      for (const prompt of binding.prompts ?? []) {
        prompts.push(prompt);
      }
    }
    return prompts;
  }

  async listTools(): Promise<
    readonly { name: string; description: string; parameters: unknown }[]
  > {
    const tools: Array<{ name: string; description: string; parameters: unknown }> = [];

    for (const binding of this.servers) {
      const serverTools = (await binding.server.listTools()) as readonly McpToolDescriptor[];
      for (const tool of serverTools) {
        tools.push({
          name: `${binding.name}__${tool.name}`,
          description: tool.description,
          parameters: tool.input_schema,
        });
      }
    }

    return tools;
  }

  async callTool(
    qualifiedName: string,
    args: Record<string, unknown>
  ): Promise<{ success: boolean; content: string; error?: string }> {
    const separatorIdx = qualifiedName.indexOf("__");
    if (separatorIdx <= 0 || separatorIdx >= qualifiedName.length - 2) {
      return {
        success: false,
        content: "",
        error: `Invalid MCP tool name "${qualifiedName}". Expected "server__tool".`,
      };
    }

    const serverName = qualifiedName.slice(0, separatorIdx);
    const toolName = qualifiedName.slice(separatorIdx + 2);
    const binding = this.servers.find((candidate) => candidate.name === serverName);
    if (!binding) {
      return {
        success: false,
        content: "",
        error: `MCP server "${serverName}" is not registered`,
      };
    }

    const payload = await binding.server.callTool(toolName, args);
    return parseToolResultText(payload);
  }

  async asAgentTools(): Promise<readonly Tool[]> {
    const mcpTools = await this.listTools();
    return mcpTools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: (tool.parameters ?? {
        type: "object",
        properties: {},
      }) as Tool["parameters"],
      execute: async (args) => {
        return await this.callTool(tool.name, args as Record<string, unknown>);
      },
    }));
  }
}
