/**
 * MCP (Model Context Protocol) server functionality for Caliche.
 *
 * This module allows you to create custom MCP servers that expose tools, prompts,
 * and resources via the Model Context Protocol.
 */

import { wrapMcpToolExecuteForNapi } from "./mcp_execute_bridge";
import { native } from "./native/module";
import type { JsonValue } from "./types";

type McpExecuteFn = (err: Error | null, arg: JsonValue) => unknown;

interface NativeMcpToolDescriptor {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

interface NativeMcpServerBuilderHandle {
  setName?(name: string): void;
  set_name?(name: string): void;
  setVersion?(version: string): void;
  set_version?(version: string): void;
  enableTools?(enabled: boolean): void;
  enable_tools?(enabled: boolean): void;
  enablePrompts?(enabled: boolean): void;
  enable_prompts?(enabled: boolean): void;
  enableResources?(enabled: boolean): void;
  enable_resources?(enabled: boolean): void;
  addTool?(tool: NativeMcpToolDescriptor, executeFn: McpExecuteFn): void;
  add_tool?(tool: NativeMcpToolDescriptor, executeFn: McpExecuteFn): void;
  build(): Promise<NativeMcpServerHandle> | NativeMcpServerHandle;
}

interface NativeMcpServerHandle {
  runStdio?(): Promise<void>;
  run_stdio?(): Promise<void>;
}

interface NativeInProcessMcpServerBuilderHandle {
  version?(version: string): void;
  addTool?(tool: NativeMcpToolDescriptor, executeFn: McpExecuteFn): void;
  add_tool?(tool: NativeMcpToolDescriptor, executeFn: McpExecuteFn): void;
  build(): NativeInProcessMcpServerHandle;
}

interface InProcessToolListing {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

interface InProcessToolCallResult {
  content?: Array<{ type: string; text?: string; resource?: unknown }>;
  is_error?: boolean;
}

interface NativeInProcessMcpServerHandle {
  readonly name: string;
  readonly version: string;
  listTools?(): Promise<InProcessToolListing[]> | InProcessToolListing[];
  list_tools?(): Promise<InProcessToolListing[]> | InProcessToolListing[];
  callTool?(
    name: string,
    args: Record<string, unknown>
  ): Promise<InProcessToolCallResult> | InProcessToolCallResult;
  call_tool?(
    name: string,
    args: Record<string, unknown>
  ): Promise<InProcessToolCallResult> | InProcessToolCallResult;
}

/**
 * Protocol for MCP tools that can be added to a server.
 */
export interface McpTool {
  /** Tool name */
  name: string;
  /** Tool description */
  description: string;
  /** Tool parameters schema (JSON Schema) */
  parameters: Record<string, unknown>;
  /**
   * Execute the tool with the given arguments.
   * @param args - Tool arguments as an object
   * @returns Result or Promise of result: object with 'success', 'content', optional 'error', or a plain string for successful content.
   */
  execute: (
    args: Record<string, unknown>
  ) =>
    | Promise<{ success: boolean; content: string; error?: string } | string>
    | { success: boolean; content: string; error?: string }
    | string;
}

/**
 * Builder for creating MCP servers.
 */
export class McpServerBuilder {
  private _builder: NativeMcpServerBuilderHandle;

  constructor() {
    if (!native.McpServerBuilder) {
      throw new Error(
        "McpServerBuilder is not available. " +
          "Make sure the native module is built with MCP support. " +
          "Rebuild with: cd caliche_js && bun run build:rust"
      );
    }
    this._builder = new native.McpServerBuilder() as NativeMcpServerBuilderHandle;
  }

  /**
   * Set the server name.
   */
  name(name: string): this {
    if (typeof this._builder.set_name === "function") {
      this._builder.set_name(name);
    } else if (typeof this._builder.setName === "function") {
      this._builder.setName(name);
    }
    return this;
  }

  /**
   * Set the server version.
   */
  version(version: string): this {
    if (typeof this._builder.set_version === "function") {
      this._builder.set_version(version);
    } else if (typeof this._builder.setVersion === "function") {
      this._builder.setVersion(version);
    }
    return this;
  }

  /**
   * Enable or disable the tools capability.
   */
  enableTools(enabled: boolean = true): this {
    if (typeof this._builder.enable_tools === "function") {
      this._builder.enable_tools(enabled);
    } else if (typeof this._builder.enableTools === "function") {
      this._builder.enableTools(enabled);
    }
    return this;
  }

  /**
   * Enable or disable the prompts capability.
   */
  enablePrompts(enabled: boolean = true): this {
    if (typeof this._builder.enable_prompts === "function") {
      this._builder.enable_prompts(enabled);
    } else if (typeof this._builder.enablePrompts === "function") {
      this._builder.enablePrompts(enabled);
    }
    return this;
  }

  /**
   * Enable or disable the resources capability.
   */
  enableResources(enabled: boolean = true): this {
    if (typeof this._builder.enable_resources === "function") {
      this._builder.enable_resources(enabled);
    } else if (typeof this._builder.enableResources === "function") {
      this._builder.enableResources(enabled);
    }
    return this;
  }

  /**
   * Add a custom tool to the server.
   */
  addTool(tool: McpTool): this {
    const executeWrapper = wrapMcpToolExecuteForNapi(tool.execute.bind(tool));

    if (typeof this._builder.add_tool === "function") {
      this._builder.add_tool(
        {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
        executeWrapper
      );
    } else if (typeof this._builder.addTool === "function") {
      this._builder.addTool(
        {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
        executeWrapper
      );
    } else {
      throw new Error("McpServerBuilder.add_tool/addTool is not available");
    }
    return this;
  }

  /**
   * Build the MCP server.
   */
  async build(): Promise<McpServer> {
    const server = await this._builder.build();
    return new McpServer(server);
  }
}

/**
 * MCP server instance.
 */
export class McpServer {
  private _server: NativeMcpServerHandle;

  constructor(server: NativeMcpServerHandle) {
    this._server = server;
  }

  /**
   * Run the server with stdio transport.
   *
   * This will read JSON-RPC requests from stdin and write responses to stdout.
   * The function will block until stdin is closed (EOF).
   */
  async runStdio(): Promise<void> {
    if (typeof this._server.run_stdio === "function") {
      await this._server.run_stdio();
    } else if (typeof this._server.runStdio === "function") {
      await this._server.runStdio();
    }
  }
}

/**
 * Latest MCP protocol version supported by this server.
 */
export const LATEST_PROTOCOL_VERSION = "2025-06-18";

/**
 * Supported MCP protocol versions.
 */
export const SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18"] as const;

// ============================================================================
// In-Process MCP Server
// ============================================================================

/**
 * Tool definition for in-process MCP server
 */
export interface InProcessMcpTool {
  /** Tool name */
  name: string;
  /** Tool description */
  description: string;
  /** Tool parameters schema (JSON Schema) */
  parameters: Record<string, unknown>;
  /**
   * Execute the tool with the given arguments.
   * @param args - Tool arguments as an object
   * @returns Result or Promise of result (same shape as {@link McpTool.execute}).
   */
  execute: (
    args: Record<string, unknown>
  ) =>
    | Promise<{ success: boolean; content: string; error?: string } | string>
    | { success: boolean; content: string; error?: string }
    | string;
}

/**
 * Builder for creating in-process MCP servers.
 */
export class InProcessMcpServerBuilder {
  private _builder: NativeInProcessMcpServerBuilderHandle;

  constructor(name: string) {
    if (!native.InProcessMcpServerBuilder) {
      throw new Error(
        "InProcessMcpServerBuilder is not available. " +
          "Make sure the native module is built with MCP support. " +
          "Rebuild with: cd caliche_js && bun run build:rust"
      );
    }
    this._builder = new native.InProcessMcpServerBuilder(
      name
    ) as NativeInProcessMcpServerBuilderHandle;
  }

  /**
   * Set the server version.
   */
  version(version: string): this {
    // Methods with &mut self in Rust need to be called directly
    if (typeof this._builder.version === "function") {
      this._builder.version(version);
    } else {
      // Fallback: use native module wrapper function if available
      if (native.inProcessMcpServerBuilderVersion) {
        native.inProcessMcpServerBuilderVersion(this._builder, version);
      }
    }
    return this;
  }

  /**
   * Add a custom tool to the server.
   */
  addTool(tool: InProcessMcpTool): this {
    const executeWrapper = wrapMcpToolExecuteForNapi(tool.execute.bind(tool));

    const addTool =
      typeof this._builder.add_tool === "function"
        ? this._builder.add_tool.bind(this._builder)
        : typeof this._builder.addTool === "function"
          ? this._builder.addTool.bind(this._builder)
          : null;
    if (!addTool) {
      const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(this._builder));
      throw new Error(
        `InProcessMcpServerBuilder.add_tool/addTool is not available. Methods: ${methods.join(", ")}`
      );
    }
    try {
      addTool(
        {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
        executeWrapper
      );
    } catch (error) {
      if (error instanceof Error && error.message.includes("ThreadsafeFunction")) {
        throw new Error(
          "Failed to add tool: native ThreadsafeFunction conversion failed. Rebuild the NAPI module."
        );
      }
      throw error;
    }
    return this;
  }

  /**
   * Build the in-process MCP server.
   */
  build(): InProcessMcpServer {
    const server = this._builder.build();
    return new InProcessMcpServer(server);
  }
}

/**
 * In-process MCP server instance.
 */
export class InProcessMcpServer {
  private _server: NativeInProcessMcpServerHandle;

  constructor(server: NativeInProcessMcpServerHandle) {
    this._server = server;
  }

  /** @internal Native binding handle for builder registration. */
  get native(): NativeInProcessMcpServerHandle {
    return this._server;
  }

  /**
   * Get the server name.
   */
  get name(): string {
    return this._server.name;
  }

  /**
   * Get the server version.
   */
  get version(): string {
    return this._server.version;
  }

  /**
   * List all available tools as MCP tools.
   */
  async listTools(): Promise<InProcessToolListing[]> {
    if (typeof this._server.listTools === "function") {
      return await this._server.listTools();
    }
    if (typeof this._server.list_tools === "function") {
      return await this._server.list_tools();
    }
    throw new Error(
      "InProcessMcpServer.list_tools/listTools is not available. Rebuild the native module."
    );
  }

  /** Call a tool by name with arguments. */
  async callTool(name: string, args: Record<string, unknown>): Promise<InProcessToolCallResult> {
    if (typeof this._server.call_tool === "function") {
      return await this._server.call_tool(name, args);
    }
    if (typeof this._server.callTool === "function") {
      return await this._server.callTool(name, args);
    }
    throw new Error(
      "InProcessMcpServer.call_tool/callTool is not available. " +
        "The native module may need to be rebuilt."
    );
  }
}
