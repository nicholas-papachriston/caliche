import {
  type HnswConfig,
  type NativeAgentBuilder,
  native,
  type VectorBackendTieredOptions,
  type VectorBackendTierSpec,
} from "../native/module";

export interface AgentBuilderVectorHost {
  _builder: NativeAgentBuilder;
}

type VectorHost = AgentBuilderVectorHost;

export const agentBuilderVectorBackendMethods = {
  vectorBackendMemory(this: VectorHost, ttlSecs?: number, maxEntries?: number): VectorHost {
    native.builderSetVectorBackendMemory(this._builder, ttlSecs, maxEntries);
    return this;
  },

  vectorBackendRedis(this: VectorHost, url: string, prefix: string): VectorHost {
    native.builderSetVectorBackendRedis(this._builder, url, prefix);
    return this;
  },

  vectorBackendPostgres(
    this: VectorHost,
    connectionString: string,
    tableName: string,
    vectorDimension?: number,
    maxConnections?: number,
    hnswConfig?: HnswConfig
  ): VectorHost {
    native.builderSetVectorBackendPostgres(
      this._builder,
      connectionString,
      tableName,
      vectorDimension,
      maxConnections,
      hnswConfig
    );
    return this;
  },

  vectorBackendQdrant(
    this: VectorHost,
    url: string,
    vectorDimension: number,
    collectionPrefix?: string,
    apiKey?: string
  ): VectorHost {
    native.builderSetVectorBackendQdrant(
      this._builder,
      url,
      vectorDimension,
      collectionPrefix,
      apiKey
    );
    return this;
  },

  vectorBackendQdrantGrpc(
    this: VectorHost,
    grpcUrl: string,
    vectorDimension: number,
    collectionPrefix?: string,
    apiKey?: string,
    batchSize?: number
  ): VectorHost {
    native.builderSetVectorBackendQdrantGrpc(
      this._builder,
      grpcUrl,
      vectorDimension,
      collectionPrefix,
      apiKey,
      batchSize
    );
    return this;
  },

  vectorBackendPinecone(
    this: VectorHost,
    baseUrl: string,
    apiKey: string,
    dimension: number,
    timeoutSecs?: number
  ): VectorHost {
    native.builderSetVectorBackendPinecone(this._builder, baseUrl, apiKey, dimension, timeoutSecs);
    return this;
  },

  vectorBackendChroma(
    this: VectorHost,
    baseUrl: string,
    collectionPrefix?: string,
    expectedDimension?: number,
    timeoutSecs?: number
  ): VectorHost {
    native.builderSetVectorBackendChroma(
      this._builder,
      baseUrl,
      collectionPrefix,
      expectedDimension,
      timeoutSecs
    );
    return this;
  },

  vectorBackendS3(
    this: VectorHost,
    bucket: string,
    keyPrefix: string,
    enableCompression?: boolean,
    batchSize?: number,
    cacheIndices?: boolean
  ): VectorHost {
    native.builderSetVectorBackendS3(
      this._builder,
      bucket,
      keyPrefix,
      enableCompression,
      batchSize,
      cacheIndices
    );
    return this;
  },

  vectorBackendColdforge(
    this: VectorHost,
    bucket: string,
    keyPrefix: string,
    region: string,
    vectorDimension: number = 1536,
    cacheSizeMb?: number,
    numCentroids?: number
  ): VectorHost {
    native.builderSetVectorBackendColdforge(
      this._builder,
      bucket,
      keyPrefix,
      region,
      vectorDimension,
      cacheSizeMb,
      numCentroids
    );
    return this;
  },

  vectorBackendDynamodb(
    this: VectorHost,
    tableName: string,
    indexName: string,
    region: string,
    vectorDimension: number,
    ensureTable?: boolean
  ): VectorHost {
    const setDynamodb = native.builderSetVectorBackendDynamodb;
    if (!setDynamodb) {
      throw new Error("DynamoDB vector backend requires a native build with vector-store-dynamodb");
    }
    setDynamodb(this._builder, tableName, indexName, region, vectorDimension, ensureTable);
    return this;
  },

  vectorBackendTiered(
    this: VectorHost,
    tiers: VectorBackendTierSpec[],
    options?: VectorBackendTieredOptions
  ): VectorHost {
    const tiersJson = JSON.stringify(tiers);
    this._builder.setVectorBackendTiered(
      tiersJson,
      options?.writeStrategy,
      options?.backgroundSync,
      options?.syncIntervalSecs,
      options?.searchStrategy
    );
    return this;
  },

  vectorBackendFromEnv(this: VectorHost): VectorHost {
    native.builderVectorBackendFromEnv(this._builder);
    return this;
  },

  withMetrics(this: VectorHost, enabled: boolean): VectorHost {
    native.builderWithMetrics(this._builder, enabled);
    return this;
  },
};
