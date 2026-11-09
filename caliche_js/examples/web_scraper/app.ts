/**
 * Core web scraping application with RAG indexing.
 * bun run examples/web_scraper/app.ts
 */

import { type Agent, AgentBuilder, type Provider, type RagService, RagTool } from "../../src/index";
import {
  extractUrlsFromResponse,
  extractUrlsFromText,
  filterUrls,
  normalizeUrl,
  urlToDocId,
} from "./url-utils";

export interface ScrapeResult {
  url: string;
  content?: string;
  chunks?: number;
  success: boolean;
  error?: string;
}

export interface Stats {
  urls_scraped: number;
  embeddings: number;
  text_chunks: number;
}

export interface ExploreResult {
  scraped: number;
  skipped: number;
  errors: number;
  total_explored?: number;
  message?: string;
}

export class WebScrapingApp {
  private scrapingAgent: Agent;
  private chatAgent: Agent;
  private ragService: RagService;
  private scrapedUrls: string[] = [];
  private discoveredUrls: Set<string> = new Set();

  private provider?: Provider;
  private model?: string;
  private apiKey?: string;

  /**
   * Initialize the web scraping app.
   *
   * @param provider - LLM provider type (defaults to environment or OpenAI)
   * @param model - Model name (defaults to environment or gpt-4o)
   * @param apiKey - API key (defaults to environment)
   */
  constructor(provider?: Provider, model?: string, apiKey?: string) {
    this.provider = provider;
    this.model = model;
    this.apiKey = apiKey;
    // This will be set in the async initialization
    this.scrapingAgent = null as unknown as Agent;
    this.chatAgent = null as unknown as Agent;
    this.ragService = null as unknown as RagService;
  }

  /**
   * Initialize the app asynchronously.
   * This must be called after construction.
   */
  async initialize(): Promise<void> {
    // Determine provider from environment or use OpenAI as default
    const finalProvider: Provider = this.provider || this.getProviderFromEnv();
    const finalModel = this.model || process.env.AGENT_MODEL || "gpt-4o";
    const finalApiKey = this.apiKey || this.getApiKeyFromEnv(finalProvider);

    // Validate credentials before building agents
    if (finalProvider !== "bedrock" && !finalApiKey) {
      const providerName = this.getProviderName(finalProvider);
      const envVar = this.getEnvVarForProvider(finalProvider);
      throw new Error(
        `No API key found for ${providerName}. ` +
          `Please set the ${envVar} environment variable or use --api-key flag.\n` +
          `Example: export ${envVar}=your_api_key_here`
      );
    }

    // Build scraping agent with web scraping tools
    console.log("Initializing web scraping agent...");
    const scrapingBuilder = new AgentBuilder();
    scrapingBuilder.provider(finalProvider);
    scrapingBuilder.model(finalModel);
    scrapingBuilder.enableTool("http");
    scrapingBuilder.systemPrompt(
      "You are a web scraping specialist. You extract structured data from websites, " +
        "handle different content formats, and organize information efficiently. " +
        "You respect robots.txt and rate limits. You're skilled at parsing HTML, " +
        "extracting text, and identifying relevant content from web pages."
    );

    if (finalApiKey && finalProvider !== "bedrock") {
      scrapingBuilder.apiKey(finalApiKey);
    } else if (finalProvider === "bedrock") {
      // Bedrock uses AWS credentials from environment
      // The SDK will automatically use AWS credentials from environment
    }

    try {
      this.scrapingAgent = await scrapingBuilder.build();
    } catch (error: any) {
      throw new Error(
        `Failed to initialize scraping agent: ${error}\n` +
          "Make sure you have valid API credentials configured."
      );
    }
    console.log(
      `Scraping agent ready: ${this.scrapingAgent.provider}, ${this.scrapingAgent.model}`
    );

    // Build chat agent with RAG tool enabled
    // Create a shared RAG tool first so we can share the backend between RagService and the agent's RAG tool
    console.log("Initializing chat agent with RAG...");

    // Create a temporary agent to get the embedding provider for RagTool
    const tempBuilder = new AgentBuilder();
    tempBuilder.provider(finalProvider);
    tempBuilder.model(finalModel);
    if (finalApiKey && finalProvider !== "bedrock") {
      tempBuilder.apiKey(finalApiKey);
    }

    const tempAgent = await tempBuilder.build();

    // Create a shared RagTool that will be used by both the agent and RagService
    console.log("   Creating shared RAG tool with shared backend...");
    const sharedRagTool = await RagTool.new(tempAgent);

    // Get the RagService from the shared RagTool to ensure they use the same backend
    this.ragService = sharedRagTool.service();

    // Build chat agent with the shared RAG tool
    const chatBuilder = new AgentBuilder();
    chatBuilder.provider(finalProvider);
    chatBuilder.model(finalModel);
    chatBuilder.sharedRagTool(sharedRagTool); // Use shared RAG tool
    chatBuilder.enableTool("semantic_search");
    chatBuilder.systemPrompt(
      "You are a helpful assistant that answers questions based on scraped web content. " +
        "Use the RAG tool to search through indexed documents when answering questions. " +
        "Cite sources when referencing specific information from the scraped content."
    );

    if (finalApiKey && finalProvider !== "bedrock") {
      chatBuilder.apiKey(finalApiKey);
    }

    try {
      this.chatAgent = await chatBuilder.build();
      console.log(`Chat agent ready: ${this.chatAgent.provider}, ${this.chatAgent.model}`);
      console.log("RAG tool enabled on chat agent (shared backend)");
      console.log("RagService ready for fast indexing (shared backend with RAG tool)");
    } catch (error: any) {
      throw new Error(
        `Failed to initialize chat agent: ${error}\n` +
          "Make sure you have valid API credentials configured."
      );
    }
    console.log();
  }

  /**
   * Scrape a URL using the agent's http tool (fetch_html operation).
   *
   * @param url - URL to scrape
   * @returns Dictionary with scraping results
   */
  async scrapeUrl(url: string): Promise<ScrapeResult> {
    // Ensure URL has a protocol
    const originalUrl = url;
    const normalizedUrl = normalizeUrl(url);
    if (normalizedUrl !== originalUrl) {
      console.log("[WARNING] Added https:// protocol to URL");
    }

    console.log(`\nScraping: ${normalizedUrl}`);
    console.log("Please wait...");

    try {
      // Use the agent to scrape the URL
      // Ask explicitly for the full tool output including discovered URLs
      const response = await this.scrapingAgent.chat(
        `Use the http tool with operation='fetch_html' to scrape this URL: ${normalizedUrl}. ` +
          "IMPORTANT: After using the tool, show me the COMPLETE tool output including: " +
          "1. The scraped text content " +
          "2. The 'Discovered URLs' section with all links found on the page. " +
          "Do not summarize - include the full '=== Discovered URLs ===' section."
      );

      // The response should contain the scraped content
      // The agent's response includes the tool result
      let scrapedContent = response.response;

      // Extract discovered URLs from the response
      // Try multiple methods to find URLs
      let discoveredUrls = extractUrlsFromResponse(response.response, normalizedUrl);

      // If we didn't find URLs in the response, make a separate call to get links
      if (discoveredUrls.size === 0) {
        try {
          const linksResponse = await this.scrapingAgent.chat(
            `Use the http tool with operation='fetch_html' on ${normalizedUrl}. ` +
              "After fetching, show me the complete tool output including the 'Discovered URLs' section. " +
              "List every URL found on that page."
          );
          discoveredUrls = extractUrlsFromResponse(linksResponse.response, normalizedUrl);
          // Also extract from the original response text
          const textUrls = extractUrlsFromText(response.response, normalizedUrl);
          for (const url of textUrls) {
            discoveredUrls.add(url);
          }
        } catch (_error) {
          // Fallback: extract URLs from the content text itself
          discoveredUrls = extractUrlsFromText(scrapedContent, normalizedUrl);
        }
      }

      // Filter out the current URL and common non-page URLs
      discoveredUrls = filterUrls(discoveredUrls, normalizedUrl);

      if (discoveredUrls.size > 0) {
        const newUrls = new Set(
          Array.from(discoveredUrls).filter((u) => !this.scrapedUrls.includes(u))
        );
        for (const url of newUrls) {
          this.discoveredUrls.add(url);
        }
        console.log(`   Discovered ${discoveredUrls.size} links (${newUrls.size} new)`);
      } else {
        console.log(`   [WARNING] No links discovered from this page`);
      }

      // If the content is too short, it might not have been scraped properly
      if (scrapedContent.trim().length < 50) {
        console.log("[WARNING] Scraped content seems too short, trying alternative approach...");
        // Try a more direct approach
        const response2 = await this.scrapingAgent.chat(
          `Fetch the content from ${normalizedUrl} using the http tool with operation='fetch_html'. ` +
            "Return only the extracted text content, not any explanations."
        );
        if (response2.response.trim().length > scrapedContent.trim().length) {
          scrapedContent = response2.response;
        }
      }

      // Index the content directly using RagService (much faster - no LLM calls)
      // This only calls the embedding API, not the LLM, so it's much faster
      console.log("Indexing content...");
      const docId = urlToDocId(normalizedUrl);

      // Use RagService directly for fast indexing (embeddings only, no LLM)
      // This is much faster than going through the agent's chat interface
      const indexResult = await this.ragService.indexDocument(scrapedContent, docId);

      this.scrapedUrls.push(normalizedUrl);

      console.log(`Scraped and indexed: ${normalizedUrl}`);
      console.log(`   Content length: ${scrapedContent.length} characters`);
      console.log(`   Chunks: ${indexResult.chunk_count}`);
      if (indexResult.usage) {
        const tokens = indexResult.usage.embedding_tokens || 0;
        console.log(`   Embedding tokens: ${tokens}`);
      }

      return {
        url: normalizedUrl,
        content:
          scrapedContent.length > 500 ? `${scrapedContent.slice(0, 500)}...` : scrapedContent,
        chunks: indexResult.chunk_count,
        success: true,
      };
    } catch (error: any) {
      console.error(`[ERROR] Error scraping ${normalizedUrl}: ${error}`);
      console.error(error.stack);
      return {
        url: normalizedUrl,
        error: String(error),
        success: false,
      };
    }
  }

  /**
   * Scrape multiple URLs.
   *
   * @param urls - List of URLs to scrape
   * @returns List of scraping results
   */
  async scrapeUrls(urls: string[]): Promise<ScrapeResult[]> {
    const results: ScrapeResult[] = [];
    for (const url of urls) {
      const result = await this.scrapeUrl(url);
      results.push(result);
    }
    return results;
  }

  /**
   * Chat with the agent about scraped content.
   *
   * @param query - User query
   * @returns Agent response
   */
  async chat(query: string): Promise<string> {
    try {
      // Enhance query to encourage RAG usage
      let enhancedQuery = query;
      const keywords = ["use rag", "search", "find", "look"];
      if (!keywords.some((keyword) => query.toLowerCase().includes(keyword))) {
        // Add context hint if not already present
        enhancedQuery = `${query} (Use the RAG tool to search through the scraped content if needed.)`;
      }

      const response = await this.chatAgent.chat(enhancedQuery);
      return response.response;
    } catch (error: any) {
      return `Error: ${error}`;
    }
  }

  /**
   * Get statistics about scraped content.
   */
  async getStats(): Promise<Stats> {
    const stats = await this.ragService.stats();
    return {
      urls_scraped: this.scrapedUrls.length,
      embeddings: stats.embedding_count,
      text_chunks: stats.text_count,
    };
  }

  /**
   * Explore links discovered from scraped pages.
   *
   * @param maxPages - Maximum number of pages to scrape
   * @param maxDepth - Maximum depth to explore (1 = direct links, 2 = links from links, etc.)
   * @param sameDomainOnly - Only explore links from the same domain
   * @param filterKeywords - Optional list of keywords - only explore links containing these
   * @returns Dictionary with exploration results
   */
  async exploreLinks(
    maxPages: number = 5,
    maxDepth: number = 2,
    sameDomainOnly: boolean = true,
    filterKeywords?: string[]
  ): Promise<ExploreResult> {
    if (this.discoveredUrls.size === 0) {
      return {
        scraped: 0,
        skipped: 0,
        errors: 0,
        message: "No discovered URLs to explore. Scrape a page first.",
      };
    }

    console.log(`\nExploring links (max ${maxPages} pages, depth ${maxDepth})...`);

    // Filter URLs
    let urlsToExplore = Array.from(this.discoveredUrls);

    if (sameDomainOnly && this.scrapedUrls.length > 0) {
      // Get base domain from first scraped URL
      const firstUrl = this.scrapedUrls[0];
      if (!firstUrl) {
        return {
          scraped: 0,
          skipped: 0,
          errors: 0,
          message: "No scraped URLs available for domain filtering.",
        };
      }
      const baseDomain = new URL(firstUrl).hostname;
      urlsToExplore = urlsToExplore.filter((u) => {
        try {
          return new URL(u).hostname === baseDomain;
        } catch {
          return false;
        }
      });
      console.log(`   Filtered to ${urlsToExplore.length} URLs from same domain (${baseDomain})`);
    }

    if (filterKeywords && filterKeywords.length > 0) {
      const filtered: string[] = [];
      for (const url of urlsToExplore) {
        if (filterKeywords.some((keyword) => url.toLowerCase().includes(keyword.toLowerCase()))) {
          filtered.push(url);
        }
      }
      urlsToExplore = filtered;
      console.log(
        `   Filtered to ${urlsToExplore.length} URLs matching keywords: ${filterKeywords}`
      );
    }

    // Limit to maxPages
    urlsToExplore = urlsToExplore.slice(0, maxPages);

    // Remove already scraped URLs
    urlsToExplore = urlsToExplore.filter((u) => !this.scrapedUrls.includes(u));

    if (urlsToExplore.length === 0) {
      return {
        scraped: 0,
        skipped: 0,
        errors: 0,
        message: "All discovered URLs have already been scraped.",
      };
    }

    console.log(`   Exploring ${urlsToExplore.length} URLs in parallel (max 8 workers)...\n`);

    let scrapedCount = 0;
    let errorCount = 0;
    const totalUrls = urlsToExplore.length;
    let completedCount = 0;

    // Use Promise.allSettled for parallel scraping with concurrency limit
    const concurrency = 8;
    const batches: string[][] = [];
    for (let i = 0; i < urlsToExplore.length; i += concurrency) {
      batches.push(urlsToExplore.slice(i, i + concurrency));
    }

    for (const batch of batches) {
      const promises = batch.map(async (url) => {
        try {
          const result = await this.scrapeUrl(url);
          completedCount++;
          const status = result.success ? "[OK]" : "[ERROR]";
          console.log(`   [${completedCount}/${totalUrls}] ${status} ${url}`);
          return result;
        } catch (error: any) {
          completedCount++;
          console.log(`   [${completedCount}/${totalUrls}] [ERROR] ${url}: ${error}`);
          return { success: false, error: String(error), url } as ScrapeResult;
        }
      });

      const results = await Promise.allSettled(promises);
      for (const result of results) {
        if (result.status === "fulfilled") {
          if (result.value.success) {
            scrapedCount++;
          } else {
            errorCount++;
          }
        } else {
          errorCount++;
        }
      }
    }

    console.log(); // Empty line after parallel scraping

    // Remove explored URLs from discovered set
    for (const url of urlsToExplore) {
      this.discoveredUrls.delete(url);
    }

    return {
      scraped: scrapedCount,
      skipped: urlsToExplore.length - scrapedCount - errorCount,
      errors: errorCount,
      total_explored: urlsToExplore.length,
    };
  }

  /**
   * Scrape a URL and then explore its links.
   *
   * @param url - URL to start from
   * @param maxPages - Maximum number of linked pages to explore
   * @param sameDomainOnly - Only explore links from the same domain
   * @returns Dictionary with exploration results
   */
  async exploreFromUrl(
    url: string,
    maxPages: number = 5,
    sameDomainOnly: boolean = true
  ): Promise<ExploreResult> {
    // First scrape the starting URL
    console.log(`Starting exploration from: ${url}\n`);
    const result = await this.scrapeUrl(url);

    if (!result.success) {
      return {
        scraped: 0,
        skipped: 0,
        errors: 1,
        message: `Failed to scrape starting URL: ${result.error || "Unknown error"}`,
      };
    }

    // Check if we discovered any URLs
    if (this.discoveredUrls.size === 0) {
      return {
        scraped: 1, // We did scrape the starting URL
        skipped: 0,
        errors: 0,
        message: `Successfully scraped ${url}, but no links were discovered on the page.`,
      };
    }

    console.log(`\nFound ${this.discoveredUrls.size} links to explore...\n`);

    // Then explore discovered links
    const exploreResult = await this.exploreLinks(maxPages, 2, sameDomainOnly);

    // Include the initial scrape in the results
    exploreResult.scraped += 1; // Count the initial URL
    return exploreResult;
  }

  /**
   * Clear all indexed content.
   */
  async clear(): Promise<void> {
    await this.ragService.clear();
    this.scrapedUrls = [];
    this.discoveredUrls.clear();
  }

  /**
   * Get discovered URLs.
   */
  getDiscoveredUrls(): string[] {
    return Array.from(this.discoveredUrls);
  }

  private getProviderFromEnv(): Provider {
    const providerStr = (process.env.AGENT_PROVIDER || "openai").toLowerCase();
    const providerMap: Record<string, Provider> = {
      openai: "openai",
      anthropic: "anthropic",
      bedrock: "bedrock",
      openrouter: "openrouter",
      perplexity: "perplexity",
      google: "google",
      cohere: "cohere",
    };
    return providerMap[providerStr] || "openai";
  }

  private getApiKeyFromEnv(provider: Provider): string | undefined {
    const envMap: Record<string, string> = {
      openai: "OPENAI_API_KEY",
      anthropic: "ANTHROPIC_API_KEY",
      openrouter: "OPENROUTER_API_KEY",
      perplexity: "PERPLEXITY_API_KEY",
      google: "GOOGLE_API_KEY",
      cohere: "COHERE_API_KEY",
    };
    const envVar = envMap[provider];
    return envVar ? process.env[envVar] : undefined;
  }

  private getProviderName(provider: Provider): string {
    const nameMap: Record<string, string> = {
      openai: "OpenAI",
      anthropic: "Anthropic",
      bedrock: "Bedrock",
      openrouter: "OpenRouter",
      perplexity: "Perplexity",
      google: "Google",
      cohere: "Cohere",
    };
    return nameMap[provider] || "Unknown";
  }

  private getEnvVarForProvider(provider: Provider): string {
    const envMap: Record<string, string> = {
      openai: "OPENAI_API_KEY",
      anthropic: "ANTHROPIC_API_KEY",
      openrouter: "OPENROUTER_API_KEY",
      perplexity: "PERPLEXITY_API_KEY",
      google: "GOOGLE_API_KEY",
      cohere: "COHERE_API_KEY",
    };
    return envMap[provider] || "API_KEY";
  }
}
