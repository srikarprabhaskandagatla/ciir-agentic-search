# Stage 2 - Searcher  (uses Tavily - free tier, 1000 searches/month)

from __future__ import annotations
import logging
import os

from tavily import AsyncTavilyClient
from ..models import SearchResult

logger = logging.getLogger(__name__)


async def fetch_web_results(query: str, max_results: int = 8) -> list[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",   # "advanced" uses 2x quota - not needed
        include_answer=False,
        include_raw_content=True,  # full page text used as scrape fallback on cloud hosts
    )

    results = []
    for item in response.get("results", []):
        url = item.get("url", "").strip()
        if url:
            # Prefer raw_content (full page text) over snippet for richer fallback
            raw = (item.get("raw_content") or "").strip()
            snippet = raw if raw else (item.get("content") or "").strip()
            results.append(SearchResult(
                url=url,
                title=item.get("title", ""),
                snippet=snippet,
            ))

    logger.info("Tavily returned %d results for: %s", len(results), query)
    return results[:max_results]