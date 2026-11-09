/**
 * URL extraction and utility functions for web scraping.
 */

/**
 * Convert URL to a valid document ID.
 */
export function urlToDocId(url: string): string {
  const parsed = new URL(url);
  // Use domain + path as ID, sanitize it
  let docId = `${parsed.hostname}${parsed.pathname}`.replace(/\//g, "_").replace(/\./g, "_");
  if (!docId) {
    docId = "scraped_content";
  }
  return docId.slice(0, 100); // Limit length
}

/**
 * Extract URLs from the http tool response (fetch_html operation).
 */
export function extractUrlsFromResponse(response: string, baseUrl: string): Set<string> {
  const urls = new Set<string>();

  // Look for "=== Discovered URLs ===" section
  if (response.includes("=== Discovered URLs ===")) {
    const urlSection = response.split("=== Discovered URLs ===")[1];
    if (urlSection) {
      // Extract URLs from lines starting with "- "
      for (const line of urlSection.split("\n")) {
        const trimmed = line.trim();
        if (trimmed.startsWith("- ")) {
          let url = trimmed.slice(2).trim();
          // Clean up URL (remove trailing punctuation that might have been included)
          url = url.replace(/[.,;:!?)]+$/, "");
          // Resolve relative URLs
          if (url.startsWith("http://") || url.startsWith("https://")) {
            urls.add(url);
          } else if (url.startsWith("/") || url.startsWith("./")) {
            // Make absolute URL
            const absoluteUrl = new URL(url, baseUrl).href;
            urls.add(absoluteUrl);
          }
        }
      }
    }
  }

  // Also look for "Found X links" patterns and extract URLs nearby
  // The tool might format URLs differently
  const lines = response.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line?.includes("Found") && line.toLowerCase().includes("link")) {
      // Look at surrounding lines for URLs
      const start = Math.max(0, i - 5);
      const end = Math.min(lines.length, i + 10);
      for (let j = start; j < end; j++) {
        const lineToCheck = lines[j];
        if (lineToCheck) {
          const potentialUrls = lineToCheck.match(/https?:\/\/[^\s<>"{}|\\^`[\]]+/g);
          if (potentialUrls) {
            for (const url of potentialUrls) {
              urls.add(url);
            }
          }
        }
      }
    }
  }

  // Also try to extract URLs from the response text (fallback)
  // Look for http:// or https:// URLs
  const urlPattern = /https?:\/\/[^\s<>"{}|\\^`[\]]+[^\s<>"{}|\\^`[\].,;:!?)]/g;
  const foundUrls = response.match(urlPattern);
  if (foundUrls) {
    for (const url of foundUrls) {
      urls.add(url);
    }
  }

  return urls;
}

/**
 * Extract URLs from plain text content.
 */
export function extractUrlsFromText(text: string, baseUrl: string): Set<string> {
  const urls = new Set<string>();

  // Find all HTTP/HTTPS URLs
  const urlPattern = /https?:\/\/[^\s<>"{}|\\^`[\]]+[^\s<>"{}|\\^`[\].,;:!?)]/g;
  const foundUrls = text.match(urlPattern);
  if (foundUrls) {
    for (const url of foundUrls) {
      urls.add(url);
    }
  }

  // Also look for relative URLs (starting with /)
  const relativePattern = /(?:href|src|url)=["']?([/][^\s<>"'{}|\\^`[\]]+)["']?/gi;
  const relativeMatches = text.matchAll(relativePattern);
  for (const match of relativeMatches) {
    const relUrl = match[1];
    if (relUrl) {
      const absoluteUrl = new URL(relUrl, baseUrl).href;
      urls.add(absoluteUrl);
    }
  }

  return urls;
}

/**
 * Ensure URL has a protocol.
 */
export function normalizeUrl(url: string): string {
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    return `https://${url}`;
  }
  return url;
}

/**
 * Filter out unwanted URLs.
 */
export function filterUrls(urls: Set<string>, excludeUrl?: string): Set<string> {
  const filtered = new Set<string>();

  for (const url of urls) {
    // Skip the exclude URL if provided
    if (excludeUrl && url === excludeUrl) {
      continue;
    }

    // Skip non-page URLs
    if (
      url.endsWith(".jpg") ||
      url.endsWith(".jpeg") ||
      url.endsWith(".png") ||
      url.endsWith(".gif") ||
      url.endsWith(".svg") ||
      url.endsWith(".pdf") ||
      url.endsWith(".css") ||
      url.endsWith(".js")
    ) {
      continue;
    }

    // Skip mailto and tel links
    if (url.startsWith("mailto:") || url.startsWith("tel:")) {
      continue;
    }

    filtered.add(url);
  }

  return filtered;
}
