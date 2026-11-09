import {
  type NativeAgentTemplate,
  type NativePromptTemplate,
  type NativePromptTemplateStore,
  type NativeStorageConfig,
  type NativeTemplateRegistry,
  native,
} from "../native/module";
import type { PromptVariables } from "../types";

export class PromptTemplate {
  private readonly _template: NativePromptTemplate;

  constructor(template: string) {
    this._template = native.createPromptTemplate(template);
  }

  /**
   * Get the internal native template (for internal use)
   */
  get internalTemplate(): NativePromptTemplate {
    return this._template;
  }

  /**
   * Format the template with variables
   *
   * @param variables - Variables to substitute
   * @returns Formatted prompt string
   */
  format(variables: PromptVariables): string {
    return native.promptTemplateFormat(this._template, variables);
  }

  /**
   * Get the variable names required by this template
   *
   * @returns Array of variable names
   */
  variables(): string[] {
    return native.promptTemplateVariables(this._template);
  }

  /**
   * Create a summarization template
   */
  static summarize(): PromptTemplate {
    const template = native.createSummarizeTemplate();
    return Object.create(PromptTemplate.prototype, {
      _template: { value: template, writable: false },
    }) as PromptTemplate;
  }

  /**
   * Create an analysis template
   */
  static analyze(): PromptTemplate {
    const template = native.createAnalyzeTemplate();
    return Object.create(PromptTemplate.prototype, {
      _template: { value: template, writable: false },
    }) as PromptTemplate;
  }

  /**
   * Create a translation template
   */
  static translate(): PromptTemplate {
    const template = native.createTranslateTemplate();
    return Object.create(PromptTemplate.prototype, {
      _template: { value: template, writable: false },
    }) as PromptTemplate;
  }
}

// ============================================================================
// Storage Configuration
// ============================================================================

/**
 * Storage configuration for templates and prompts
 */
export class StorageConfig {
  private readonly _config: NativeStorageConfig;

  private constructor(config: NativeStorageConfig) {
    this._config = config;
  }

  /**
   * Get the internal native config (for internal use)
   */
  get internalConfig(): NativeStorageConfig {
    return this._config;
  }

  /**
   * Create a filesystem storage configuration
   */
  static filesystem(basePath: string): StorageConfig {
    const config = native.createStorageConfigFilesystem(basePath);
    return new StorageConfig(config);
  }

  /**
   * Create an AWS S3 storage configuration
   */
  static awsS3(
    bucket: string,
    region?: string,
    accessKeyId?: string,
    secretAccessKey?: string,
    sessionToken?: string
  ): StorageConfig {
    if (!native.createStorageConfigAwsS3) {
      throw new Error("AWS S3 storage is not available. Build with the 's3' feature enabled.");
    }
    const config = native.createStorageConfigAwsS3(
      bucket,
      region,
      accessKeyId,
      secretAccessKey,
      sessionToken
    );
    return new StorageConfig(config);
  }

  /**
   * Create a Cloudflare R2 storage configuration
   */
  static cloudflareR2(
    bucket: string,
    accountId: string,
    accessKeyId: string,
    secretAccessKey: string,
    endpoint?: string
  ): StorageConfig {
    if (!native.createStorageConfigCloudflareR2) {
      throw new Error(
        "Cloudflare R2 storage is not available. Build with the 's3' feature enabled."
      );
    }
    const config = native.createStorageConfigCloudflareR2(
      bucket,
      accountId,
      accessKeyId,
      secretAccessKey,
      endpoint
    );
    return new StorageConfig(config);
  }

  /**
   * Enable or disable caching
   */
  withCache(enabled: boolean): StorageConfig {
    const config = native.storageConfigWithCache(this._config, enabled);
    return new StorageConfig(config);
  }

  /**
   * Set cache TTL in seconds
   */
  withCacheTtl(ttlSeconds: number): StorageConfig {
    const config = native.storageConfigWithCacheTtl(this._config, ttlSeconds);
    return new StorageConfig(config);
  }

  /**
   * Set cache max size in MB
   */
  withCacheMaxSize(maxSizeMb: number): StorageConfig {
    const config = native.storageConfigWithCacheMaxSize(this._config, maxSizeMb);
    return new StorageConfig(config);
  }
}

// ============================================================================
// Template Registry
// ============================================================================

/**
 * Template registry for managing agent templates
 */
export class TemplateRegistry {
  private readonly _registry: NativeTemplateRegistry;

  private constructor(registry: NativeTemplateRegistry) {
    this._registry = registry;
  }

  /**
   * Create a new template registry
   */
  static create(): TemplateRegistry {
    const registry = native.createTemplateRegistry();
    return new TemplateRegistry(registry);
  }

  /**
   * Create a template registry with storage configuration
   */
  static async withStorage(config: StorageConfig): Promise<TemplateRegistry> {
    const registry = await native.createTemplateRegistryWithStorage(config.internalConfig);
    return new TemplateRegistry(registry);
  }

  /**
   * Add a template directory to scan
   */
  addTemplateDir(dir: string): void {
    native.templateRegistryAddTemplateDir(this._registry, dir);
  }

  /**
   * Load templates from configured sources
   */
  async loadTemplates(): Promise<number> {
    return await native.templateRegistryLoadTemplates(this._registry);
  }

  /**
   * Get a template by ID
   */
  async get(id: string): Promise<AgentTemplate | null> {
    const template = await native.templateRegistryGet(this._registry, id);
    if (!template) {
      return null;
    }
    return new AgentTemplate(template);
  }

  /**
   * List all templates
   */
  async list(): Promise<AgentTemplate[]> {
    const templates = await native.templateRegistryList(this._registry);
    return templates.map((t) => new AgentTemplate(t));
  }
}

/**
 * Agent template loaded from the template registry.
 */
export class AgentTemplate {
  private readonly _template: NativeAgentTemplate;

  constructor(template: NativeAgentTemplate) {
    this._template = template;
  }

  /** Template identifier */
  id(): string {
    return this._template.id();
  }

  /** Human-readable template name */
  name(): string {
    return this._template.name();
  }

  /** Template description */
  description(): string {
    return this._template.description();
  }

  /** Template version string */
  version(): string {
    return this._template.version();
  }

  /**
   * @internal Native template handle for advanced integrations.
   */
  get internalTemplate(): NativeAgentTemplate {
    return this._template;
  }
}

// ============================================================================
// Prompt Template Store
// ============================================================================

/**
 * Prompt template store for managing saved prompts
 */
export class PromptTemplateStore {
  private readonly _store: NativePromptTemplateStore;

  private constructor(store: NativePromptTemplateStore) {
    this._store = store;
  }

  /**
   * Create a prompt template store with storage configuration
   */
  static async create(config: StorageConfig): Promise<PromptTemplateStore> {
    if (!native.createPromptTemplateStore) {
      throw new Error(
        "Prompt template store is not available. Build with the 's3' feature enabled."
      );
    }
    const store = await native.createPromptTemplateStore(config.internalConfig);
    return new PromptTemplateStore(store);
  }

  /**
   * Save a prompt template
   */
  async save(name: string, template: PromptTemplate): Promise<void> {
    if (!native.promptTemplateStoreSave) {
      throw new Error(
        "Prompt template store is not available. Build with the 's3' feature enabled."
      );
    }
    await native.promptTemplateStoreSave(this._store, name, template.internalTemplate);
  }

  /**
   * Load a prompt template
   */
  async load(name: string): Promise<PromptTemplate> {
    if (!native.promptTemplateStoreLoad) {
      throw new Error(
        "Prompt template store is not available. Build with the 's3' feature enabled."
      );
    }
    const template = await native.promptTemplateStoreLoad(this._store, name);
    return Object.create(PromptTemplate.prototype, {
      _template: { value: template, writable: false },
    }) as PromptTemplate;
  }

  /**
   * List all prompt template names
   */
  async list(): Promise<string[]> {
    if (!native.promptTemplateStoreList) {
      throw new Error(
        "Prompt template store is not available. Build with the 's3' feature enabled."
      );
    }
    return await native.promptTemplateStoreList(this._store);
  }
}
