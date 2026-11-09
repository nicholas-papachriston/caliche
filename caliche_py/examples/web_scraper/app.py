"""
Core web scraping application with RAG indexing.
uv run python -m caliche_py.examples.web_scraper
"""

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set
from urllib.parse import urlparse

from caliche import AgentBuilder, ProviderType, RagService, RagTool

from .url_utils import (
    url_to_doc_id,
    extract_urls_from_response,
    extract_urls_from_text,
    normalize_url,
    filter_urls,
)


class WebScrapingApp:
    """Web scraping application with RAG indexing and chat interface."""

    def __init__(
        self,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the web scraping app.

        Args:
            provider: LLM provider type (defaults to environment or OpenAI)
            model: Model name (defaults to environment or gpt-4o)
            api_key: API key (defaults to environment)
        """
        # Determine provider from environment or use OpenAI as default
        if provider is None:
            provider_str = os.getenv("AGENT_PROVIDER", "openai").lower()
            provider_map = {
                "openai": ProviderType.OpenAI,
                "anthropic": ProviderType.Anthropic,
                "bedrock": ProviderType.Bedrock,
                "openrouter": ProviderType.OpenRouter,
                "perplexity": ProviderType.Perplexity,
                "google": ProviderType.Google,
                "cohere": ProviderType.Cohere,
            }
            provider = provider_map.get(provider_str, ProviderType.OpenAI)

        # Determine model from environment or use default
        if model is None:
            model = os.getenv("AGENT_MODEL", "gpt-4o")

        # Get API key from environment if not provided
        if api_key is None:
            if provider == ProviderType.OpenAI:
                api_key = os.getenv("OPENAI_API_KEY", "")
            elif provider == ProviderType.Anthropic:
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
            elif provider == ProviderType.Bedrock:
                # Bedrock uses AWS credentials, not API key
                api_key = None
            elif provider == ProviderType.OpenRouter:
                api_key = os.getenv("OPENROUTER_API_KEY", "")
            elif provider == ProviderType.Perplexity:
                api_key = os.getenv("PERPLEXITY_API_KEY", "")
            elif provider == ProviderType.Google:
                api_key = os.getenv("GOOGLE_API_KEY", "")
            elif provider == ProviderType.Cohere:
                api_key = os.getenv("COHERE_API_KEY", "")

        # Validate credentials before building agents
        if provider != ProviderType.Bedrock:
            if not api_key:
                provider_name = {
                    ProviderType.OpenAI: "OpenAI",
                    ProviderType.Anthropic: "Anthropic",
                    ProviderType.OpenRouter: "OpenRouter",
                    ProviderType.Perplexity: "Perplexity",
                    ProviderType.Google: "Google",
                    ProviderType.Cohere: "Cohere",
                }.get(provider, "Unknown")
                env_var = {
                    ProviderType.OpenAI: "OPENAI_API_KEY",
                    ProviderType.Anthropic: "ANTHROPIC_API_KEY",
                    ProviderType.OpenRouter: "OPENROUTER_API_KEY",
                    ProviderType.Perplexity: "PERPLEXITY_API_KEY",
                    ProviderType.Google: "GOOGLE_API_KEY",
                    ProviderType.Cohere: "COHERE_API_KEY",
                }.get(provider, "API_KEY")
                raise ValueError(
                    f"No API key found for {provider_name}. "
                    f"Please set the {env_var} environment variable or use --api-key flag.\n"
                    f"Example: export {env_var}=your_api_key_here"
                )

        # Build scraping agent with web scraping tools
        print("Initializing web scraping agent...")
        scraping_builder = AgentBuilder()
        scraping_builder.provider(provider)
        scraping_builder.model(model)
        scraping_builder.enable_tool("http")
        scraping_builder.system_prompt(
            "You are a web scraping specialist. You extract structured data from websites, "
            "handle different content formats, and organize information efficiently. "
            "You respect robots.txt and rate limits. You're skilled at parsing HTML, "
            "extracting text, and identifying relevant content from web pages."
        )

        if api_key and provider != ProviderType.Bedrock:
            scraping_builder.api_key(api_key)
        elif provider == ProviderType.Bedrock:
            # Bedrock uses AWS credentials from environment
            scraping_builder.aws_credentials()

        try:
            self.scraping_agent = scraping_builder.build()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize scraping agent: {e}\n"
                "Make sure you have valid API credentials configured."
            ) from e
        print(f"Scraping agent ready: {self.scraping_agent.provider}, {self.scraping_agent.model}")

        # Build chat agent with RAG tool enabled
        # Create a shared RAG tool first so we can share the backend between RagService and the agent's RAG tool
        print("Initializing chat agent with RAG...")
        
        # First, create a temporary agent to get the embedding provider for RagTool
        temp_builder = AgentBuilder()
        temp_builder.provider(provider)
        temp_builder.model(model)
        if api_key and provider != ProviderType.Bedrock:
            temp_builder.api_key(api_key)
        elif provider == ProviderType.Bedrock:
            temp_builder.aws_credentials()
        
        temp_agent = temp_builder.build()
        
        # Create a shared RagTool that will be used by both the agent and RagService
        print("   Creating shared RAG tool with shared backend...")
        shared_rag_tool = RagTool(temp_agent)
        
        # Get the RagService from the shared RagTool to ensure they use the same backend
        self.rag_service = shared_rag_tool.service()
        
        # Build chat agent with the shared RAG tool
        chat_builder = AgentBuilder()
        chat_builder.provider(provider)
        chat_builder.model(model)
        # Pass the underlying PyRagTool to the Rust binding
        chat_builder.shared_rag_tool(shared_rag_tool._tool)  # Use shared RAG tool
        chat_builder.enable_tool("semantic_search")
        chat_builder.system_prompt(
            "You are a helpful assistant that answers questions based on scraped web content. "
            "Use the RAG tool to search through indexed documents when answering questions. "
            "Cite sources when referencing specific information from the scraped content."
        )

        if api_key and provider != ProviderType.Bedrock:
            chat_builder.api_key(api_key)
        elif provider == ProviderType.Bedrock:
            chat_builder.aws_credentials()

        try:
            self.chat_agent = chat_builder.build()
            print(f"Chat agent ready: {self.chat_agent.provider}, {self.chat_agent.model}")
            print("RAG tool enabled on chat agent (shared backend)")
            print("RagService ready for fast indexing (shared backend with RAG tool)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize chat agent: {e}\n"
                "Make sure you have valid API credentials configured."
            ) from e
        print()

        self.scraped_urls: List[str] = []
        self.discovered_urls: Set[str] = set()  # Track discovered but not yet scraped URLs

    def scrape_url(self, url: str) -> dict:
        """
        Scrape a URL using the agent's http tool (fetch_html operation).

        Args:
            url: URL to scrape

        Returns:
            Dictionary with scraping results
        """
        # Ensure URL has a protocol
        original_url = url
        url = normalize_url(url)
        if url != original_url:
            print(f"[WARNING] Added https:// protocol to URL")

        print(f"\nScraping: {url}")
        print("Please wait...")

        try:
            # Use the agent to scrape the URL
            # Ask explicitly for the full tool output including discovered URLs
            response, usage, _ = self.scraping_agent.chat(
                f"Use the http tool with operation='fetch_html' to scrape this URL: {url}. "
                "IMPORTANT: After using the tool, show me the COMPLETE tool output including: "
                "1. The scraped text content "
                "2. The 'Discovered URLs' section with all links found on the page. "
                "Do not summarize - include the full '=== Discovered URLs ===' section."
            )

            # The response should contain the scraped content
            # The agent's response includes the tool result
            scraped_content = response
            
            # Extract discovered URLs from the response
            # Try multiple methods to find URLs
            discovered_urls = extract_urls_from_response(response, url)
            
            # If we didn't find URLs in the response, make a separate call to get links
            if not discovered_urls:
                try:
                    links_response, _, _ = self.scraping_agent.chat(
                        f"Use the http tool with operation='fetch_html' on {url}. "
                        "After fetching, show me the complete tool output including the 'Discovered URLs' section. "
                        "List every URL found on that page."
                    )
                    discovered_urls = extract_urls_from_response(links_response, url)
                    # Also extract from the original response text
                    discovered_urls.update(extract_urls_from_text(response, url))
                except Exception as e:
                    # Fallback: extract URLs from the content text itself
                    discovered_urls = extract_urls_from_text(scraped_content, url)
            
            # Filter out the current URL and common non-page URLs
            discovered_urls = filter_urls(discovered_urls, exclude_url=url)
            
            if discovered_urls:
                new_urls = discovered_urls - set(self.scraped_urls)
                self.discovered_urls.update(new_urls)
                print(f"   Discovered {len(discovered_urls)} links ({len(new_urls)} new)")
            else:
                print(f"   [WARNING] No links discovered from this page")

            # If the content is too short, it might not have been scraped properly
            if len(scraped_content.strip()) < 50:
                print("[WARNING] Scraped content seems too short, trying alternative approach...")
                # Try a more direct approach
                response2, _, _ = self.scraping_agent.chat(
                    f"Fetch the content from {url} using the http tool with operation='fetch_html'. "
                    "Return only the extracted text content, not any explanations."
                )
                if len(response2.strip()) > len(scraped_content.strip()):
                    scraped_content = response2

            # Index the content directly using RagService (much faster - no LLM calls)
            # This only calls the embedding API, not the LLM, so it's much faster
            print("Indexing content...")
            doc_id = url_to_doc_id(url)
            
            # Use RagService directly for fast indexing (embeddings only, no LLM)
            # This is much faster than going through the agent's chat interface
            index_result = self.rag_service.index_document(scraped_content, doc_id)

            self.scraped_urls.append(url)

            print(f"Scraped and indexed: {url}")
            print(f"   Content length: {len(scraped_content)} characters")
            print(f"   Chunks: {index_result.chunk_count}")
            if index_result.usage:
                tokens = index_result.usage.get("embedding_tokens", 0)
                print(f"   Embedding tokens: {tokens}")

            return {
                "url": url,
                "content": scraped_content[:500] + "..." if len(scraped_content) > 500 else scraped_content,
                "chunks": index_result.chunk_count,
                "success": True,
            }

        except Exception as e:
            print(f"[ERROR] Error scraping {url}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "url": url,
                "error": str(e),
                "success": False,
            }

    def scrape_urls(self, urls: List[str]) -> List[dict]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of scraping results
        """
        results = []
        for url in urls:
            result = self.scrape_url(url)
            results.append(result)
        return results

    def chat(self, query: str) -> str:
        """
        Chat with the agent about scraped content.

        Args:
            query: User query

        Returns:
            Agent response
        """
        try:
            # Enhance query to encourage RAG usage
            enhanced_query = query
            if not any(keyword in query.lower() for keyword in ["use rag", "search", "find", "look"]):
                # Add context hint if not already present
                enhanced_query = (
                    f"{query} "
                    "(Use the RAG tool to search through the scraped content if needed.)"
                )

            response, usage, _ = self.chat_agent.chat(enhanced_query)
            return response
        except Exception as e:
            return f"Error: {e}"

    def get_stats(self) -> dict:
        """Get statistics about scraped content."""
        stats = self.rag_service.stats()
        return {
            "urls_scraped": len(self.scraped_urls),
            "embeddings": stats.embedding_count,
            "text_chunks": stats.text_count,
        }

    def explore_links(
        self,
        max_pages: int = 5,
        max_depth: int = 2,
        same_domain_only: bool = True,
        filter_keywords: Optional[List[str]] = None,
    ) -> dict:
        """
        Explore links discovered from scraped pages.
        
        Args:
            max_pages: Maximum number of pages to scrape
            max_depth: Maximum depth to explore (1 = direct links, 2 = links from links, etc.)
            same_domain_only: Only explore links from the same domain
            filter_keywords: Optional list of keywords - only explore links containing these
            
        Returns:
            Dictionary with exploration results
        """
        if not self.discovered_urls:
            return {
                "scraped": 0,
                "skipped": 0,
                "errors": 0,
                "message": "No discovered URLs to explore. Scrape a page first."
            }
        
        print(f"\nExploring links (max {max_pages} pages, depth {max_depth})...")
        
        # Filter URLs
        urls_to_explore = list(self.discovered_urls)
        
        if same_domain_only and self.scraped_urls:
            # Get base domain from first scraped URL
            base_domain = urlparse(self.scraped_urls[0]).netloc
            urls_to_explore = [
                u for u in urls_to_explore
                if urlparse(u).netloc == base_domain
            ]
            print(f"   Filtered to {len(urls_to_explore)} URLs from same domain ({base_domain})")
        
        if filter_keywords:
            filtered = []
            for url in urls_to_explore:
                if any(keyword.lower() in url.lower() for keyword in filter_keywords):
                    filtered.append(url)
            urls_to_explore = filtered
            print(f"   Filtered to {len(urls_to_explore)} URLs matching keywords: {filter_keywords}")
        
        # Limit to max_pages
        urls_to_explore = urls_to_explore[:max_pages]
        
        # Remove already scraped URLs
        urls_to_explore = [u for u in urls_to_explore if u not in self.scraped_urls]
        
        if not urls_to_explore:
            return {
                "scraped": 0,
                "skipped": 0,
                "errors": 0,
                "message": "All discovered URLs have already been scraped."
            }
        
        print(f"   Exploring {len(urls_to_explore)} URLs in parallel (max 8 workers)...\n")
        
        scraped_count = 0
        error_count = 0
        total_urls = len(urls_to_explore)
        completed_count = 0
        
        # Lock for thread-safe printing and counting
        print_lock = threading.Lock()
        
        def scrape_with_progress(url: str) -> dict:
            """Scrape URL and print progress."""
            nonlocal completed_count
            try:
                result = self.scrape_url(url)
                with print_lock:
                    completed_count += 1
                    status = "[OK]" if result.get("success") else "[ERROR]"
                    print(f"   [{completed_count}/{total_urls}] {status} {url}")
                return result
            except Exception as e:
                with print_lock:
                    completed_count += 1
                    print(f"   [{completed_count}/{total_urls}] [ERROR] {url}: {e}")
                return {"success": False, "error": str(e), "url": url}
        
        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all scraping tasks
            future_to_url = {
                executor.submit(scrape_with_progress, url): url
                for url in urls_to_explore
            }
            
            # Process results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result.get("success"):
                        scraped_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    with print_lock:
                        print(f"   [ERROR] Unexpected error scraping {url}: {e}")
        
        print()  # Empty line after parallel scraping
        
        # Remove explored URLs from discovered set
        self.discovered_urls -= set(urls_to_explore)
        
        return {
            "scraped": scraped_count,
            "skipped": len(urls_to_explore) - scraped_count - error_count,
            "errors": error_count,
            "total_explored": len(urls_to_explore)
        }

    def explore_from_url(
        self,
        url: str,
        max_pages: int = 5,
        same_domain_only: bool = True,
    ) -> dict:
        """
        Scrape a URL and then explore its links.
        
        Args:
            url: URL to start from
            max_pages: Maximum number of linked pages to explore
            same_domain_only: Only explore links from the same domain
            
        Returns:
            Dictionary with exploration results
        """
        # First scrape the starting URL
        print(f"Starting exploration from: {url}\n")
        result = self.scrape_url(url)
        
        if not result["success"]:
            return {
                "scraped": 0,
                "errors": 1,
                "message": f"Failed to scrape starting URL: {result.get('error', 'Unknown error')}"
            }
        
        # Check if we discovered any URLs
        if not self.discovered_urls:
            return {
                "scraped": 1,  # We did scrape the starting URL
                "skipped": 0,
                "errors": 0,
                "message": f"Successfully scraped {url}, but no links were discovered on the page."
            }
        
        print(f"\nFound {len(self.discovered_urls)} links to explore...\n")
        
        # Then explore discovered links
        explore_result = self.explore_links(
            max_pages=max_pages,
            same_domain_only=same_domain_only
        )
        
        # Include the initial scrape in the results
        explore_result["scraped"] += 1  # Count the initial URL
        return explore_result
