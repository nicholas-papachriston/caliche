/**
 * LangChain Tool Adapter for Caliche
 *
 * This adapter allows you to use LangChain tools directly with Caliche agents.
 * It wraps LangChain tools and converts them to Caliche's Tool interface.
 */

import type { z } from "zod";
import type { Tool, ToolResult } from "../index";
import type { JsonRecord, JsonValue, ToolArguments, ToolParameterSchema } from "../types";

type JsonSchemaConverter = (schema: unknown, options?: Record<string, unknown>) => unknown;

/**
 * Convert a Zod schema to a draft-07 JSON Schema. Zod 4 schemas carry `_zod`
 * and convert with Zod's own `toJSONSchema`; Zod 3 schemas convert with
 * `zod-to-json-schema`, which cannot read Zod 4. Both are optional peers,
 * loaded only when a LangChain tool carries a schema of that version.
 */
function zodSchemaToJsonSchema(schema: object): unknown {
  if ("_zod" in schema) {
    const { toJSONSchema } = require("zod") as { toJSONSchema: JsonSchemaConverter };
    return toJSONSchema(schema, { target: "draft-7" });
  }
  const { zodToJsonSchema } = require("zod-to-json-schema") as {
    zodToJsonSchema: JsonSchemaConverter;
  };
  return zodToJsonSchema(schema, { target: "jsonSchema7" });
}

/**
 * LangChain tool interface - any object that implements these methods
 */
interface LangChainToolLike {
  name: string;
  description: string;
  invoke(args: ToolArguments | any): Promise<JsonValue | string | any>;
  schema?: z.ZodTypeAny | ToolParameterSchema;
  argsSchema?: z.ZodTypeAny | ToolParameterSchema;
}

/**
 * Adapter that wraps a LangChain tool to make it compatible with Caliche
 *
 * @example
 * ```typescript
 * import { DynamicStructuredTool } from "@langchain/core/tools";
 * import { LangChainAdapter } from "@caliche/adapters-langchain";
 *
 * const langchainTool = new DynamicStructuredTool({
 *   name: "calculator",
 *   description: "Performs basic arithmetic",
 *   schema: z.object({
 *     expression: z.string().describe("Mathematical expression")
 *   }),
 *   func: async ({ expression }) => {
 *     return String(eval(expression));
 *   }
 * });
 *
 * const adapter = new LangChainAdapter(langchainTool);
 * const agent = await new AgentBuilder()
 *   .addTool(adapter)
 *   .build();
 * ```
 */
export class LangChainAdapter implements Tool {
  readonly name: string;
  readonly description: string;
  readonly parameters: ToolParameterSchema;
  private readonly tool: LangChainToolLike;

  /**
   * Create a new LangChain adapter
   *
   * @param tool - The LangChain tool to wrap (must have name, description, and invoke methods)
   */
  constructor(tool: LangChainToolLike) {
    this.tool = tool;
    this.name = tool.name;
    this.description = tool.description;

    // Convert LangChain schema to JSON Schema
    // LangChain tools can have either:
    // 1. A Zod schema (via schema property)
    // 2. A JSON Schema object (via schema property)
    // 3. An argsSchema property (Zod schema)
    const schema = tool.schema || tool.argsSchema;

    if (!schema) {
      // Fallback: empty schema if no schema is provided
      this.parameters = {
        type: "object",
        properties: {},
        required: [],
      };
    } else if (typeof schema === "object" && "parse" in schema) {
      // It's a Zod schema - convert to JSON Schema
      try {
        const jsonSchema = zodSchemaToJsonSchema(schema);
        // Ensure we have a valid object structure
        if (typeof jsonSchema === "object" && jsonSchema !== null) {
          this.parameters = jsonSchema as ToolParameterSchema;
        } else {
          throw new Error("Invalid JSON schema result");
        }
      } catch (error) {
        console.warn(
          `Failed to convert Zod schema for tool ${this.name}, using empty schema:`,
          error
        );
        this.parameters = {
          type: "object",
          properties: {},
          required: [],
        };
      }
    } else if (typeof schema === "object" && "type" in schema) {
      // It's already a JSON Schema object
      this.parameters = schema as ToolParameterSchema;
    } else {
      // Unknown schema format, use empty schema
      console.warn(`Unknown schema format for tool ${this.name}, using empty schema`);
      this.parameters = {
        type: "object",
        properties: {},
        required: [],
      };
    }
  }

  /**
   * Execute the LangChain tool
   *
   * @param args - Tool arguments as a record
   * @returns Promise resolving to ToolResult
   */
  async execute(args: ToolArguments): Promise<ToolResult> {
    try {
      // LangChain tools use invoke() method
      const result = await this.tool.invoke(args);

      // Convert result to ToolResult format
      // LangChain tools may return ToolMessage, string, or other types
      let content: string;
      if (typeof result === "string") {
        content = result;
      } else if (result && typeof result === "object" && "content" in result) {
        content = String((result as any).content);
      } else {
        content = JSON.stringify(result);
      }

      return {
        success: true,
        content,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        success: false,
        content: "",
        error: errorMessage,
      };
    }
  }
}

/**
 * LangChain Memory Adapter for Caliche
 *
 * This adapter bridges LangChain memory to Caliche memory strategies.
 */
import type { LangChainMemory, LangChainMessage } from "../types";

export class LangChainMemoryAdapter {
  private memory: LangChainMemory;

  constructor(langchainMemory: LangChainMemory) {
    this.memory = langchainMemory;
  }

  getMessages(): Array<{ role: string; content: string }> {
    try {
      // Try multiple ways to access messages from LangChain memory
      const messages =
        this.memory.chatHistory?.messages ||
        this.memory.chat_memory?.messages ||
        this.memory.messages ||
        [];

      // Convert LangChain messages to Caliche format
      return messages.map((msg: LangChainMessage) => {
        let role = "user";
        if (msg.type === "human" || msg.type === "user") {
          role = "user";
        } else if (msg.type === "ai" || msg.type === "assistant") {
          role = "assistant";
        } else if (msg.type === "system") {
          role = "system";
        }

        const content = msg.content ? String(msg.content) : "";

        return { role, content };
      });
    } catch {
      return [];
    }
  }

  addMessage(role: string, content: string): void {
    try {
      // Try chatHistory first (newer LangChain API)
      const chatHistory = this.memory?.chatHistory;
      if (chatHistory) {
        if (role === "user" && typeof chatHistory.addUserMessage === "function") {
          chatHistory.addUserMessage(content);
        } else if (role === "assistant" && typeof chatHistory.addAIChatMessage === "function") {
          chatHistory.addAIChatMessage(content);
        }
      } else {
        // Fallback to chat_memory (older LangChain API)
        const chatMemory = this.memory?.chat_memory;
        if (chatMemory) {
          if (role === "user" && typeof chatMemory.addUserMessage === "function") {
            chatMemory.addUserMessage(content);
          } else if (role === "assistant" && typeof chatMemory.addAIMessage === "function") {
            chatMemory.addAIMessage(content);
          }
        }
      }
    } catch {
      // Ignore errors
    }
  }
}

/**
 * LangChain Retriever Adapter for Caliche RAG
 *
 * This adapter uses LangChain retrievers for RAG in Caliche.
 */
import type { LangChainDocument, LangChainRetriever } from "../types";

export class LangChainRetrieverAdapter {
  private retriever: LangChainRetriever;

  constructor(retriever: LangChainRetriever) {
    this.retriever = retriever;
  }

  async getRelevantDocuments(
    query: string,
    topK: number = 5
  ): Promise<Array<{ content: string; metadata?: JsonRecord }>> {
    try {
      // Call LangChain retriever
      let docs: readonly LangChainDocument[];
      if (this.retriever.getRelevantDocuments) {
        docs = await this.retriever.getRelevantDocuments(query);
      } else if (this.retriever.agetRelevantDocuments) {
        docs = await this.retriever.agetRelevantDocuments(query);
      } else {
        return [];
      }

      // Convert to Caliche format
      return docs.slice(0, topK).map((doc: LangChainDocument) => {
        const content = doc.page_content || doc.content || (typeof doc === "string" ? doc : "");

        const metadata = doc.metadata || {};

        return { content, metadata };
      });
    } catch (error) {
      console.warn(`LangChain retriever failed: ${error}`);
      return [];
    }
  }
}

/**
 * LangChain Chain Adapter for Caliche
 *
 * This adapter wraps LangChain chains as Caliche chains.
 */
import type { LangChainChain } from "../types";

export class LangChainChainAdapter {
  private chain: LangChainChain;

  constructor(chain: LangChainChain) {
    this.chain = chain;
  }

  async run(inputs: JsonRecord): Promise<JsonRecord> {
    try {
      // Call LangChain chain
      let result: JsonValue;
      if (this.chain.ainvoke) {
        result = await this.chain.ainvoke(inputs);
      } else if (this.chain.invoke) {
        result = await this.chain.invoke(inputs);
      } else if (this.chain.arun) {
        result = await this.chain.arun(inputs);
      } else if (this.chain.run) {
        result = await this.chain.run(inputs);
      } else {
        throw new Error("Chain does not have invoke, ainvoke, run, or arun method");
      }

      // Convert result to dict format
      if (typeof result === "object" && result !== null && !Array.isArray(result)) {
        return result as JsonRecord;
      } else if (typeof result === "string") {
        return { output: result };
      } else {
        return { output: String(result) };
      }
    } catch (error) {
      throw new Error(
        `LangChain chain execution failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}
