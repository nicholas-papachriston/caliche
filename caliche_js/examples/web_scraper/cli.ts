/**
 * CLI interface and interactive mode for web scraping app.
 */

import { createInterface } from "node:readline";
import type { Provider } from "../../src/index";
import { WebScrapingApp } from "./app";

/**
 * Run interactive chat mode.
 */
async function interactiveMode(app: WebScrapingApp): Promise<void> {
  console.log(`\n${"=".repeat(60)}`);
  console.log("Web Scraping Chat Interface");
  console.log("=".repeat(60));
  console.log("\nCommands:");
  console.log("  scrape <url>              - Scrape a URL");
  console.log("  scrape <url1> <url2> ...   - Scrape multiple URLs");
  console.log("  explore [max_pages]        - Explore discovered links (default: 5 pages)");
  console.log("  explore_from <url> [max]   - Scrape URL and explore its links");
  console.log("  links                      - Show discovered links");
  console.log("  stats                     - Show statistics");
  console.log("  clear                      - Clear all indexed content");
  console.log("  quit/exit                  - Exit the application");
  console.log("\nOr just ask a question about the scraped content!");
  console.log(`${"=".repeat(60)}\n`);

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> => {
    return new Promise((resolve) => {
      rl.question(prompt, resolve);
    });
  };

  while (true) {
    try {
      const userInput = (await question("You: ")).trim();

      if (!userInput) {
        continue;
      }

      // Handle commands
      if (
        userInput.toLowerCase() === "quit" ||
        userInput.toLowerCase() === "exit" ||
        userInput.toLowerCase() === "q"
      ) {
        console.log("\nGoodbye!");
        break;
      }

      if (userInput.toLowerCase() === "stats") {
        const stats = await app.getStats();
        console.log("\nStatistics:");
        console.log(`   URLs scraped: ${stats.urls_scraped}`);
        console.log(`   Embeddings: ${stats.embeddings}`);
        console.log(`   Text chunks: ${stats.text_chunks}\n`);
        continue;
      }

      if (userInput.toLowerCase() === "clear") {
        await app.clear();
        console.log("[OK] Cleared all indexed content\n");
        continue;
      }

      if (userInput.toLowerCase() === "links") {
        const discoveredUrls = app.getDiscoveredUrls();
        if (discoveredUrls.length > 0) {
          console.log(`\nDiscovered ${discoveredUrls.length} links:`);
          for (let i = 0; i < Math.min(discoveredUrls.length, 20); i++) {
            console.log(`   ${i + 1}. ${discoveredUrls[i]}`);
          }
          if (discoveredUrls.length > 20) {
            console.log(`   ... and ${discoveredUrls.length - 20} more\n`);
          }
        } else {
          console.log("\nNo discovered links yet. Scrape a page first.\n");
        }
        continue;
      }

      if (
        userInput.toLowerCase().startsWith("explore_from") ||
        userInput.toLowerCase().startsWith("explore_from_url")
      ) {
        // Handle both "explore_from" and "explore_from_url" commands
        let parts: string[];
        if (userInput.toLowerCase().startsWith("explore_from_url")) {
          parts = userInput.slice(16).trim().split(/\s+/);
        } else {
          parts = userInput.slice(13).trim().split(/\s+/);
        }
        if (parts.length === 0 || !parts[0]) {
          console.log("[ERROR] Please provide a URL: explore_from <url> [max_pages]\n");
          continue;
        }
        const startUrl = parts[0];
        const maxPages =
          parts.length > 1 && parts[1] && /^\d+$/.test(parts[1]) ? parseInt(parts[1], 10) : 5;
        const result = await app.exploreFromUrl(startUrl, maxPages);
        console.log("\nExploration complete:");
        console.log(`   Scraped: ${result.scraped} pages`);
        console.log(`   Skipped: ${result.skipped || 0} pages`);
        console.log(`   Errors: ${result.errors || 0}`);
        if (result.message) {
          console.log(`   ${result.message}`);
        }
        console.log();
        continue;
      }

      if (userInput.toLowerCase().startsWith("explore")) {
        const parts = userInput.slice(7).trim().split(/\s+/);
        const maxPages =
          parts.length > 0 && parts[0] && /^\d+$/.test(parts[0]) ? parseInt(parts[0], 10) : 5;
        const result = await app.exploreLinks(maxPages);
        console.log("\nExploration complete:");
        console.log(`   Scraped: ${result.scraped} pages`);
        console.log(`   Skipped: ${result.skipped || 0} pages`);
        console.log(`   Errors: ${result.errors || 0}`);
        if (result.message) {
          console.log(`   ${result.message}`);
        }
        console.log();
        continue;
      }

      if (userInput.toLowerCase().startsWith("scrape ")) {
        const urls = userInput.slice(7).trim().split(/\s+/);
        if (urls.length === 0) {
          console.log("[ERROR] Please provide at least one URL to scrape\n");
          continue;
        }

        for (const url of urls) {
          await app.scrapeUrl(url);
        }
        console.log(); // Empty line after scraping
        continue;
      }

      // Regular chat query
      console.log("\nThinking...");
      const response = await app.chat(userInput);
      console.log(`\nAgent: ${response}\n`);
    } catch (error: any) {
      if (error.name === "SIGINT" || error.name === "SIGTERM") {
        console.log("\n\nGoodbye!");
        break;
      }
      console.error(`\n[ERROR] Error: ${error}\n`);
    }
  }

  rl.close();
}

/**
 * Main entry point.
 */
export async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Simple argument parsing
  let provider: Provider | undefined;
  let model: string | undefined;
  let apiKey: string | undefined;
  const scrapeUrls: string[] = [];
  let query: string | undefined;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--provider" && i + 1 < args.length) {
      provider = args[++i] as Provider;
    } else if (arg === "--model" && i + 1 < args.length) {
      model = args[++i];
    } else if (arg === "--api-key" && i + 1 < args.length) {
      apiKey = args[++i];
    } else if (arg === "--scrape") {
      // Collect all URLs until next flag or end
      i++;
      while (i < args.length) {
        const nextArg = args[i];
        if (!nextArg || nextArg.startsWith("--")) {
          break;
        }
        scrapeUrls.push(nextArg);
        i++;
      }
      i--; // Back up one since we'll increment in the loop
    } else if (arg === "--query" && i + 1 < args.length) {
      query = args[++i];
    } else if (arg === "--help" || arg === "-h") {
      console.log(`
Web scraping app with Caliche

Usage:
  bun run web_scraper [options]

Options:
  --provider <provider>    LLM provider (openai, anthropic, bedrock, openrouter, perplexity, google, cohere)
  --model <model>         Model name (default: gpt-4o)
  --api-key <key>         API key (default: from environment)
  --scrape <url>...       URLs to scrape
  --query <query>         Query to ask about scraped content
  --help, -h              Show this help message

Examples:
  # Interactive mode
  bun run web_scraper

  # Scrape URLs from command line
  bun run web_scraper --scrape https://example.com

  # Use specific provider with API key
  bun run web_scraper --provider openai --model gpt-4o --api-key sk-...

Environment Variables:
  Set one of these based on your provider:
  - OPENAI_API_KEY (for OpenAI)
  - ANTHROPIC_API_KEY (for Anthropic)
  - OPENROUTER_API_KEY (for OpenRouter)
  - PERPLEXITY_API_KEY (for Perplexity)
  - GOOGLE_API_KEY (for Google)
  - COHERE_API_KEY (for Cohere)
  - AWS credentials (for Bedrock)
      `);
      process.exit(0);
    }
  }

  // Map provider string to Provider type
  let finalProvider: Provider | undefined;
  if (provider) {
    const providerMap: Record<string, Provider> = {
      openai: "openai",
      anthropic: "anthropic",
      bedrock: "bedrock",
      openrouter: "openrouter",
      perplexity: "perplexity",
      google: "google",
      cohere: "cohere",
    };
    finalProvider = providerMap[provider.toLowerCase()];
    if (!finalProvider) {
      console.error(`Unknown provider: ${provider}`);
      process.exit(1);
    }
  }

  // Initialize app
  const app = new WebScrapingApp(finalProvider, model, apiKey);
  await app.initialize();

  // Handle command-line scraping
  if (scrapeUrls.length > 0) {
    console.log("Scraping URLs from command line...\n");
    await app.scrapeUrls(scrapeUrls);
    console.log("\nScraping complete!\n");

    // Show stats
    const stats = await app.getStats();
    console.log("Statistics:");
    console.log(`   URLs scraped: ${stats.urls_scraped}`);
    console.log(`   Embeddings: ${stats.embeddings}`);
    console.log(`   Text chunks: ${stats.text_chunks}\n`);

    // Handle query if provided
    if (query) {
      console.log(`Query: ${query}\n`);
      const response = await app.chat(query);
      console.log(`Agent: ${response}\n`);
    } else {
      // Enter interactive mode
      await interactiveMode(app);
    }
  } else {
    // Interactive mode
    await interactiveMode(app);
  }
}

// Note: This file should be run via cli-main.ts for proper entry point handling
