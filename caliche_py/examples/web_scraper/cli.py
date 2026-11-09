"""CLI interface and interactive mode for web scraping app."""

from caliche import ProviderType

from .app import WebScrapingApp


def interactive_mode(app: WebScrapingApp):
    """Run interactive chat mode."""
    print("\n" + "=" * 60)
    print("Web Scraping Chat Interface")
    print("=" * 60)
    print("\nCommands:")
    print("  scrape <url>              - Scrape a URL")
    print("  scrape <url1> <url2> ...   - Scrape multiple URLs")
    print("  explore [max_pages]        - Explore discovered links (default: 5 pages)")
    print("  explore_from <url> [max]   - Scrape URL and explore its links")
    print("  links                      - Show discovered links")
    print("  stats                     - Show statistics")
    print("  clear                      - Clear all indexed content")
    print("  quit/exit                  - Exit the application")
    print("\nOr just ask a question about the scraped content!")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            elif user_input.lower() == "stats":
                stats = app.get_stats()
                print(f"\nStatistics:")
                print(f"   URLs scraped: {stats['urls_scraped']}")
                print(f"   Embeddings: {stats['embeddings']}")
                print(f"   Text chunks: {stats['text_chunks']}\n")
                continue

            elif user_input.lower() == "clear":
                app.rag_service.clear()
                app.scraped_urls.clear()
                app.discovered_urls.clear()
                print("[OK] Cleared all indexed content\n")
                continue

            elif user_input.lower() == "links":
                if app.discovered_urls:
                    print(f"\nDiscovered {len(app.discovered_urls)} links:")
                    for i, url in enumerate(sorted(app.discovered_urls)[:20], 1):
                        print(f"   {i}. {url}")
                    if len(app.discovered_urls) > 20:
                        print(f"   ... and {len(app.discovered_urls) - 20} more\n")
                else:
                    print("\nNo discovered links yet. Scrape a page first.\n")
                continue

            elif user_input.lower().startswith("explore_from") or user_input.lower().startswith("explore_from_url"):
                # Handle both "explore_from" and "explore_from_url" commands
                if user_input.lower().startswith("explore_from_url"):
                    parts = user_input[16:].strip().split()
                else:
                    parts = user_input[13:].strip().split()
                if not parts:
                    print("[ERROR] Please provide a URL: explore_from <url> [max_pages]\n")
                    continue
                start_url = parts[0]
                max_pages = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                result = app.explore_from_url(start_url, max_pages=max_pages)
                print(f"\nExploration complete:")
                print(f"   Scraped: {result['scraped']} pages")
                print(f"   Skipped: {result.get('skipped', 0)} pages")
                print(f"   Errors: {result.get('errors', 0)}")
                if result.get('message'):
                    print(f"   {result['message']}")
                print()
                continue

            elif user_input.lower().startswith("explore"):
                parts = user_input[7:].strip().split()
                max_pages = int(parts[0]) if parts and parts[0].isdigit() else 5
                result = app.explore_links(max_pages=max_pages)
                print(f"\nExploration complete:")
                print(f"   Scraped: {result['scraped']} pages")
                print(f"   Skipped: {result.get('skipped', 0)} pages")
                print(f"   Errors: {result.get('errors', 0)}")
                if result.get('message'):
                    print(f"   {result['message']}")
                print()
                continue

            elif user_input.lower().startswith("scrape "):
                urls = user_input[7:].strip().split()
                if not urls:
                    print("[ERROR] Please provide at least one URL to scrape\n")
                    continue

                for url in urls:
                    app.scrape_url(url)
                print()  # Empty line after scraping
                continue

            # Regular chat query
            print("\nThinking...")
            response = app.chat(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Error: {e}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Web scraping app with Caliche",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (from caliche_py directory)
  uv run python -m web_scraper.app

  # Scrape URLs from command line
  uv run python -m web_scraper.app --scrape https://example.com

  # Use specific provider with API key
  uv run python -m web_scraper.app --provider openai --model gpt-4o --api-key sk-...

Environment Variables:
  Set one of these based on your provider:
  - OPENAI_API_KEY (for OpenAI)
  - ANTHROPIC_API_KEY (for Anthropic)
  - OPENROUTER_API_KEY (for OpenRouter)
  - PERPLEXITY_API_KEY (for Perplexity)
  - GOOGLE_API_KEY (for Google)
  - COHERE_API_KEY (for Cohere)
  - AWS credentials (for Bedrock)
        """,
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "bedrock", "openrouter", "perplexity", "google", "cohere"],
        help="LLM provider (default: from AGENT_PROVIDER env or openai)",
    )
    parser.add_argument("--model", type=str, help="Model name (default: from AGENT_MODEL env or gpt-4o)")
    parser.add_argument("--api-key", type=str, help="API key (default: from environment)")
    parser.add_argument("--scrape", nargs="+", metavar="URL", help="URLs to scrape")
    parser.add_argument("--query", type=str, help="Query to ask about scraped content")

    args = parser.parse_args()

    # Map provider string to enum
    provider = None
    if args.provider:
        provider_map = {
            "openai": ProviderType.OpenAI,
            "anthropic": ProviderType.Anthropic,
            "bedrock": ProviderType.Bedrock,
            "openrouter": ProviderType.OpenRouter,
            "perplexity": ProviderType.Perplexity,
            "google": ProviderType.Google,
            "cohere": ProviderType.Cohere,
        }
        provider = provider_map[args.provider]

    # Initialize app
    app = WebScrapingApp(provider=provider, model=args.model, api_key=args.api_key)

    # Handle command-line scraping
    if args.scrape:
        print("Scraping URLs from command line...\n")
        results = app.scrape_urls(args.scrape)
        print("\nScraping complete!\n")

        # Show stats
        stats = app.get_stats()
        print(f"Statistics:")
        print(f"   URLs scraped: {stats['urls_scraped']}")
        print(f"   Embeddings: {stats['embeddings']}")
        print(f"   Text chunks: {stats['text_chunks']}\n")

        # Handle query if provided
        if args.query:
            print(f"Query: {args.query}\n")
            response = app.chat(args.query)
            print(f"Agent: {response}\n")
        else:
            # Enter interactive mode
            interactive_mode(app)
    else:
        # Interactive mode
        interactive_mode(app)


if __name__ == "__main__":
    main()

