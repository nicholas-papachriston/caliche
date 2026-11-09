/**
 * Web Scraping App with Caliche
 *
 * A web scraping application that uses Caliche to scrape websites,
 * index content using RAG, and provide an interactive chat interface.
 *
 * bun run examples/web_scraper/cli-main.ts
 */

export type { ExploreResult, ScrapeResult, Stats } from "./app";
export { WebScrapingApp } from "./app";
export { main } from "./cli";
export * as urlUtils from "./url-utils";
