"""
Web Scraping App with Caliche

A web scraping application that uses Caliche to scrape websites,
index content using RAG, and provide an interactive chat interface.

uv run python -m examples.web_scraper
"""

from .app import WebScrapingApp
from . import url_utils, cli

__all__ = ["WebScrapingApp", "url_utils", "cli"]

