/**
 * LangChain Embedding Provider Adapter for Caliche
 *
 * This adapter allows you to use LangChain embedding models (OpenAIEmbeddings, etc.)
 * as embedding providers for Caliche's vector memory and RAG features.
 *
 * @example
 * ```typescript
 * import { OpenAIEmbeddings } from "@langchain/openai";
 * import { LangChainEmbeddingProvider } from "@caliche/adapters-langchain";
 * import { AgentBuilder } from "@caliche/core";
 *
 * const embeddings = new OpenAIEmbeddings();
 * const provider = new LangChainEmbeddingProvider(embeddings);
 *
 * const agent = await new AgentBuilder()
 *   .provider("openai")
 *   .model("gpt-4o")
 *   .langchainEmbeddings(provider)
 *   .vectorMemory(...)
 *   .build();
 * ```
 */

import type { JsonRecord, LangChainEmbeddingModel } from "../types";

export interface EmbeddingRequest {
  texts: string[];
  model?: string;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    embedding_tokens?: number;
  };
}

/**
 * Adapter that wraps a LangChain embedding model to make it compatible with Caliche.
 */
export class LangChainEmbeddingProvider {
  private embeddings: LangChainEmbeddingModel;
  public readonly modelName: string;

  constructor(embeddings: LangChainEmbeddingModel) {
    this.embeddings = embeddings;
    this.modelName = embeddings.model || embeddings.modelName || "langchain-embeddings";
  }

  /**
   * Generate embeddings for the given texts using the LangChain embedding model.
   */
  async embed(request: EmbeddingRequest | JsonRecord): Promise<EmbeddingResponse> {
    // Handle both EmbeddingRequest and JsonRecord formats
    let texts: string[];
    if (Array.isArray((request as any).texts)) {
      texts = (request as EmbeddingRequest).texts;
    } else if ((request as JsonRecord).texts) {
      texts = Array.isArray((request as JsonRecord).texts)
        ? ((request as JsonRecord).texts as string[])
        : [];
    } else {
      texts = [];
    }
    if (texts.length === 0) {
      return {
        embeddings: [],
        model: this.modelName,
      };
    }

    // Call LangChain embeddings
    try {
      let embeddingsList: number[][];
      if (this.embeddings.embedDocuments) {
        embeddingsList = await this.embeddings.embedDocuments(texts);
      } else if (this.embeddings.aembedDocuments) {
        embeddingsList = await this.embeddings.aembedDocuments(texts);
      } else {
        throw new Error("Embeddings model does not have embedDocuments or aembedDocuments method");
      }

      // Convert to list of lists of numbers
      const embeddings = embeddingsList.map((emb: number[] | ArrayLike<number>) => {
        if (Array.isArray(emb)) {
          return emb.map((x: number) => Number(x));
        }
        // Handle numpy arrays or other formats
        return Array.from(emb).map((x: number) => Number(x));
      });

      // Estimate token usage (rough approximation: ~4 chars per token)
      const totalChars = texts.reduce((sum, text) => sum + text.length, 0);
      const estimatedTokens = Math.max(1, Math.floor(totalChars / 4));

      return {
        embeddings,
        model: String((request as any).model || this.modelName),
        usage: {
          prompt_tokens: estimatedTokens,
          completion_tokens: 0,
          total_tokens: estimatedTokens,
          embedding_tokens: estimatedTokens,
        },
      };
    } catch (error) {
      throw new Error(
        `LangChain embedding failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}
