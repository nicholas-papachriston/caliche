import { type NativeAgentBuilder, type NativeRagTool, native } from "../native/module";
import type {
  BufferMemoryConfig,
  EntityMemoryConfig,
  SummaryMemoryConfig,
  VectorMemoryConfig,
} from "../streaming/events";

export interface AgentBuilderMemoryHost {
  _builder: NativeAgentBuilder;
  _memoryPersistence: {
    persistence: {
      save_snapshot(conversationId: string, snapshot: unknown): Promise<void>;
      load_snapshot(conversationId: string): Promise<unknown | undefined>;
      clear_snapshot(conversationId: string): Promise<void>;
    };
    conversationId: string;
    autosave: boolean;
  } | null;
}

/** Minimal surface for sharing a built-in RAG tool instance. */
export interface SharedRagToolRef {
  readonly _tool: unknown;
}

type MemoryHost = AgentBuilderMemoryHost;

export const agentBuilderMemoryMethods = {
  bufferMemory(this: MemoryHost, config: BufferMemoryConfig): MemoryHost {
    if (typeof this._builder.bufferMemoryConfig === "function") {
      this._builder.bufferMemoryConfig({
        max_tokens: config.max_tokens ?? 4096,
        min_messages: config.min_messages ?? 4,
        preserve_system: config.preserve_system ?? true,
        summary_placeholder: config.summary_placeholder ?? true,
      });
      return this;
    }

    native.builderBufferMemory(
      this._builder,
      config.max_tokens ?? 4096,
      config.min_messages ?? 4,
      config.preserve_system ?? true,
      config.summary_placeholder ?? true
    );
    return this;
  },

  summaryMemory(this: MemoryHost, config: SummaryMemoryConfig): MemoryHost {
    native.builderSummaryMemory(
      this._builder,
      config.max_tokens ?? 8192,
      config.summarize_threshold_pct ?? 0.8,
      config.chunk_size ?? 12,
      config.max_summaries ?? 8
    );
    return this;
  },

  entityMemory(this: MemoryHost, config: EntityMemoryConfig): MemoryHost {
    native.builderEntityMemory(
      this._builder,
      config.max_tokens ?? 6144,
      config.max_entities ?? 128,
      config.extraction_interval ?? 3,
      config.similarity_threshold ?? 0.65
    );
    return this;
  },

  vectorMemory(this: MemoryHost, config: VectorMemoryConfig): MemoryHost {
    if (typeof this._builder.vectorMemory === "function") {
      this._builder.vectorMemory(
        config.max_tokens ?? 8192,
        config.max_memories ?? 1000,
        config.top_k ?? 5,
        config.similarity_threshold ?? 0.7,
        config.buffer_tokens ?? 4096
      );
    } else {
      native.builderVectorMemory(
        this._builder,
        config.max_tokens ?? 8192,
        config.max_memories ?? 1000,
        config.top_k ?? 5,
        config.similarity_threshold ?? 0.7,
        config.buffer_tokens ?? 4096,
        config.namespace
      );
    }
    return this;
  },

  persistMemoryWith(
    this: MemoryHost,
    persistence: MemoryHost["_memoryPersistence"] extends null
      ? never
      : NonNullable<MemoryHost["_memoryPersistence"]>["persistence"],
    conversationId: string,
    autosave: boolean = false
  ): MemoryHost {
    this._memoryPersistence = { persistence, conversationId, autosave };
    return this;
  },

  sharedRagTool(this: MemoryHost, ragTool: SharedRagToolRef): MemoryHost {
    const nativeRagTool = ragTool._tool as NativeRagTool;
    if (typeof this._builder.sharedRagTool === "function") {
      this._builder.sharedRagTool(nativeRagTool);
    } else if (native.builderSharedRagTool) {
      native.builderSharedRagTool(this._builder, nativeRagTool);
    }
    return this;
  },
};
