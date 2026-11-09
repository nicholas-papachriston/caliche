/**
 * Native binding result shapes shared by `native/module.ts` and feature modules.
 *
 * Kept in a leaf module to avoid circular imports between native bindings and
 * workflow/RAG/evaluation wrappers.
 */

import type { TokenUsage } from "../streaming/events";
import type { ChainOutput, ContextUpdate, Metadata } from "../types";

export interface CoordinatorStats {
  readonly total_agents: number;
  readonly total_load: number;
  readonly total_capacity: number;
  readonly agent_stats: readonly {
    readonly agent_id: string;
    readonly name: string;
    readonly current_load: number;
    readonly max_concurrent_tasks: number;
  }[];
}

export interface DelegationResult {
  readonly agent_id: string;
  readonly response: string;
  readonly usage: TokenUsage;
}

export interface ExecutionResult {
  readonly response: string;
  readonly usage: TokenUsage;
}

export interface MessageBusStats {
  readonly queue_size: number;
  readonly max_queue_size: number;
  readonly low_priority: number;
  readonly normal_priority: number;
  readonly high_priority: number;
  readonly critical_priority: number;
  readonly subscriber_count: number;
}

export type WorkflowState = "pending" | "running" | "paused" | "completed" | "failed" | "cancelled";

export interface PendingInterrupt {
  readonly interrupt_id: string;
  readonly execution_id: string;
  readonly node_id: string;
  readonly node_name: string;
  readonly message: string;
  readonly required_fields?: readonly string[];
}

export interface WorkflowExecutionResult {
  readonly state: WorkflowState;
  readonly output: ChainOutput;
  readonly error?: string;
}

export interface EvaluationResult {
  readonly score: number;
  readonly passed: boolean;
  readonly details: Record<string, unknown>;
  readonly error?: string;
}

export interface TestCase {
  readonly input: string;
  readonly expected_output: string;
  readonly context?: ContextUpdate;
  readonly metadata?: Metadata;
}

export interface EvaluationDataset {
  readonly name: string;
  readonly description?: string;
  readonly test_cases: readonly TestCase[];
}

export interface RagIndexResult {
  readonly chunk_count: number;
  readonly usage?: {
    readonly embedding_tokens?: number;
    readonly total_tokens?: number;
  };
}

export interface RagChunkHit {
  readonly id?: string;
  readonly text: string;
  readonly score?: number;
  readonly metadata?: Metadata;
}

export interface RagQueryResult {
  /** Convenience text snippets for the top hits. */
  readonly results: readonly string[];
  /** Structured hits when the native backend provides them. */
  readonly hits: readonly RagChunkHit[];
  readonly top_similarity?: number;
  readonly usage?: {
    readonly embedding_tokens?: number;
    readonly total_tokens?: number;
  };
}

export interface EnhancedQueryOptions {
  readonly query: string;
  readonly top_k: number;
  readonly min_similarity?: number;
  /** Dripstone retrieval mode: `dense`, `hybrid`, `full_text`, or `hybrid_full_text`. */
  readonly search_mode?: "dense" | "hybrid" | "full_text" | "hybrid_full_text";
  readonly dense_weight?: number;
  readonly sparse_weight?: number;
  readonly rerank?: boolean;
  readonly metadata_filter?: Metadata;
  readonly max_context_tokens?: number;
}

export interface EnhancedQueryResult {
  readonly results: readonly {
    readonly id: string;
    readonly score: number;
    readonly text?: string;
    readonly metadata?: Metadata;
  }[];
  readonly top_similarity?: number;
  readonly usage?: {
    readonly embedding_tokens?: number;
    readonly total_tokens?: number;
  };
  readonly warnings: readonly string[];
  readonly context_stats?: {
    readonly included_chunks: number;
    readonly omitted_chunks: number;
    readonly total_chunks: number;
    readonly truncated: boolean;
  };
}

export interface GraphSearchResult {
  readonly entry_nodes: readonly {
    readonly id: string;
    readonly type: string;
  }[];
  readonly expanded_nodes: readonly {
    readonly id: string;
    readonly type: string;
  }[];
  readonly expanded_edges: readonly {
    readonly source: string;
    readonly target: string;
    readonly relation: string;
  }[];
  readonly context: string;
}

export interface RagStats {
  readonly embedding_count: number;
  readonly text_count: number;
}
