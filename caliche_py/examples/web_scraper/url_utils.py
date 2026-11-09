"""URL extraction and utility functions for web scraping."""

import re
from typing import Set, Optional
from urllib.parse import urlparse, urljoin


def url_to_doc_id(url: str) -> str:
    """Convert URL to a valid document ID."""
    parsed = urlparse(url)
    # Use domain + path as ID, sanitize it
    doc_id = f"{parsed.netloc}{parsed.path}".replace("/", "_").replace(".", "_")
    if not doc_id:
        doc_id = "scraped_content"
    return doc_id[:100]  # Limit length


def extract_urls_from_response(response: str, base_url: str) -> Set[str]:
    """Extract URLs from the http tool response."""
    urls = set()
    
    # Look for "=== Discovered URLs ===" section
    if "=== Discovered URLs ===" in response:
        url_section = response.split("=== Discovered URLs ===")[1]
        # Extract URLs from lines starting with "- "
        for line in url_section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                url = line[2:].strip()
                # Clean up URL (remove trailing punctuation that might have been included)
                url = url.rstrip('.,;:!?)')
                # Resolve relative URLs
                if url.startswith("http://") or url.startswith("https://"):
                    urls.add(url)
                elif url.startswith("/") or url.startswith("./"):
                    # Make absolute URL
                    absolute_url = urljoin(base_url, url)
                    urls.add(absolute_url)
    
    # Also look for "Found X links" patterns and extract URLs nearby
    # The tool might format URLs differently
    lines = response.split("\n")
    for i, line in enumerate(lines):
        if "Found" in line and "link" in line.lower():
            # Look at surrounding lines for URLs
            for j in range(max(0, i-5), min(len(lines), i+10)):
                potential_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', lines[j])
                urls.update(potential_urls)
    
    # Also try to extract URLs from the response text (fallback)
    # Look for http:// or https:// URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?)]'
    found_urls = re.findall(url_pattern, response)
    urls.update(found_urls)
    
    return urls


def extract_urls_from_text(text: str, base_url: str) -> Set[str]:
    """Extract URLs from plain text content."""
    urls = set()
    
    # Find all HTTP/HTTPS URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?)]'
    found_urls = re.findall(url_pattern, text)
    urls.update(found_urls)
    
    # Also look for relative URLs (starting with /)
    relative_pattern = r'(?:href|src|url)=["\']?([/][^\s<>"\'{}|\\^`\[\]]+)["\']?'
    relative_urls = re.findall(relative_pattern, text, re.IGNORECASE)
    for rel_url in relative_urls:
        absolute_url = urljoin(base_url, rel_url)
        urls.add(absolute_url)
    
    return urls


def normalize_url(url: str) -> str:
    """Ensure URL has a protocol."""
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def filter_urls(urls: Set[str], exclude_url: Optional[str] = None) -> Set[str]:
    """Filter out unwanted URLs."""
    filtered = set()
    
    for url in urls:
        # Skip the exclude URL if provided
        if exclude_url and url == exclude_url:
            continue
        
        # Skip non-page URLs
        if url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.pdf', '.css', '.js')):
            continue
        
        # Skip mailto and tel links
        if url.startswith('mailto:') or url.startswith('tel:'):
            continue
        
        filtered.add(url)
    
    return filtered

