"""
Web Search Module using Tavily AI Search API

This module provides an interface to the Tavily AI Search API for conducting
web research. It handles search execution, result parsing, error handling,
and rate limiting.

Learning Note - What is Tavily?
--------------------------------
Tavily is a search API specifically designed for AI applications and research.

Key features that make it perfect for research agents:
1. Returns clean, structured content (no HTML parsing needed)
2. Ranks results by relevance to your research query
3. Provides rich metadata (publication date, source quality, content)
4. Handles rate limiting gracefully
5. Supports different search depths (basic vs advanced)
6. Includes content from multiple sources (websites, news, blogs)

Why use Tavily instead of web scraping?
- Legal: Respects robots.txt and terms of service
- Reliable: No broken pages, captchas, or anti-scraping measures
- Fast: Optimized infrastructure for quick results
- Cost-effective: ~$0.01 per search (very affordable)
- AI-optimized: Returns content in formats perfect for LLMs

API Documentation: https://docs.tavily.com
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None
    print("WARNING: tavily-python not installed. Run: pip install tavily-python")

from config import settings


class TavilySearcher:
    """
    Handles web searches via Tavily AI Search API.

    This class encapsulates all interactions with the Tavily API,
    including error handling, rate limiting, and result parsing.

    Learning Note - Why Use a Class?
    --------------------------------
    We use a class here because:
    1. Maintains state (API key, client instance)
    2. Groups related functions together (search, parse, handle errors)
    3. Makes the code reusable (create one instance, use many times)
    4. Easier to test and maintain
    """

    def __init__(self, api_key: str = None, search_depth: str = "basic"):
        """
        Initialize the Tavily search client.

        Args:
            api_key: Tavily API key (defaults to settings.TAVILY_API_KEY)
            search_depth: "basic" (faster) or "advanced" (more thorough)

        Raises:
            ValueError: If Tavily is not installed or API key is missing

        Learning Note - Initialization:
        -------------------------------
        The __init__ method is called when you create a new instance:

            searcher = TavilySearcher()  # Calls __init__

        It sets up everything the object needs to work properly.
        """
        if TavilyClient is None:
            raise ValueError(
                "Tavily is not installed. Please install it with: "
                "pip install tavily-python"
            )

        self.api_key = api_key or settings.TAVILY_API_KEY
        if not self.api_key:
            raise ValueError(
                "Tavily API key not found. "
                "Please set TAVILY_API_KEY in your .env file. "
                "Get your key from: https://tavily.com"
            )

        self.search_depth = search_depth

        # Initialize the Tavily client
        # This creates a connection to the Tavily API
        self.client = TavilyClient(api_key=self.api_key)

        print(f"✓ Tavily Search initialized (depth: {search_depth})")

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Execute a single search query and return results.

        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 5)
            search_type: Type of search ("general", "news", "academic")

        Returns:
            List of search result dictionaries with keys:
                - title: Page title
                - url: Page URL
                - content: Main content snippet
                - score: Relevance score (0-1)
                - published_date: Publication date (if available)

        Learning Note - Search Types:
        ----------------------------
        - general: Broad web search (blogs, articles, websites)
                   Best for: General topics, how-tos, explanations

        - news: Recent news articles only
                Best for: Current events, breaking news

        - academic: Scholarly articles and papers
                    Best for: Research papers, academic topics

        For research agents, "general" is usually best as it gives
        diverse perspectives from multiple source types.
        """
        if not query or not query.strip():
            print("WARNING: Empty query provided")
            return []

        try:
            print(f"  Searching: '{query[:50]}...'")

            # Execute the search via Tavily API
            # This is where the actual HTTP request happens
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=self.search_depth,
                include_raw_content=False  # We don't need full HTML
            )

            # Parse the response into our standardized format
            results = self._parse_results(response, query)

            print(f"    → Found {len(results)} results")
            return results

        except Exception as e:
            # Catch any errors and handle them gracefully
            print(f"    → Search error: {str(e)}")

            # Check if it's a rate limit error
            if "rate limit" in str(e).lower():
                return self._handle_rate_limit(query, max_results, search_type)

            # For other errors, return empty list
            # This prevents the entire research from failing due to one bad search
            return []

    def batch_search(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple search queries and return combined results.

        Args:
            queries: List of search query strings
            max_results_per_query: Max results per individual query

        Returns:
            Combined list of all search results from all queries

        Learning Note - Batch Searching:
        -------------------------------
        When researching a topic, we want comprehensive coverage.
        A single query might miss important perspectives.

        By searching multiple related queries, we:
        - Get diverse viewpoints
        - Cover different aspects of the topic
        - Reduce bias from any single search
        - Improve overall result quality

        Example: Topic "AI in healthcare"
        Query 1: "AI healthcare applications"
        Query 2: "machine learning medical diagnosis"
        Query 3: "AI healthcare workflow automation"
        Query 4: "artificial intelligence patient care"

        Each query finds different sources and perspectives!
        """
        all_results = []

        print(f"\n🔍 Executing {len(queries)} searches...")

        for i, query in enumerate(queries, 1):
            print(f"\nSearch {i}/{len(queries)}:")

            # Execute individual search
            results = self.search(query, max_results=max_results_per_query)

            # Add results to our combined list
            all_results.extend(results)

            # Be polite to the API - add small delay between requests
            # This prevents overwhelming the server
            if i < len(queries):
                time.sleep(0.5)  # 500ms delay

        print(f"\n✓ Total results collected: {len(all_results)}")
        return all_results

    def _parse_results(
        self,
        response: Dict[str, Any],
        original_query: str
    ) -> List[Dict[str, Any]]:
        """
        Parse Tavily API response into standardized format.

        Args:
            response: Raw response from Tavily API
            original_query: The original search query (for metadata)

        Returns:
            List of standardized result dictionaries

        Learning Note - Why Parse Results?
        ----------------------------------
        APIs often return data in complex formats with lots of fields.
        We parse the response to:
        1. Extract only what we need
        2. Standardize the format for our code
        3. Add metadata for tracking
        4. Handle missing fields gracefully

        This makes the rest of our code simpler and more reliable.
        """
        results = []

        # Get the results array from the response
        # Use .get() to safely handle missing keys
        raw_results = response.get('results', [])

        for raw_result in raw_results:
            # Extract fields we care about
            # Using .get() with defaults prevents errors if fields are missing
            parsed_result = {
                'title': raw_result.get('title', 'Untitled'),
                'url': raw_result.get('url', ''),
                'content': raw_result.get('content', ''),
                'score': raw_result.get('score', 0.0),
                'published_date': raw_result.get('published_date', None),
                'search_query': original_query,
                'retrieved_date': datetime.now().isoformat(),
                'source_type': 'web_search'
            }

            # Only include results that have actual content
            if parsed_result['content'] and len(parsed_result['content']) > 50:
                results.append(parsed_result)

        return results

    def _handle_rate_limit(
        self,
        query: str,
        max_results: int,
        search_type: str,
        retry_count: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Handle rate limit errors with exponential backoff.

        Args:
            query: The search query to retry
            max_results: Maximum results to return
            search_type: Type of search
            retry_count: Current retry attempt number

        Returns:
            Search results if retry succeeds, empty list otherwise

        Learning Note - Rate Limiting:
        -----------------------------
        APIs have limits on how many requests you can make per minute/hour.
        This prevents abuse and ensures fair access for all users.

        When we hit a rate limit, we use "exponential backoff":
        - Wait 1 second, retry
        - If fails again, wait 2 seconds, retry
        - If fails again, wait 4 seconds, retry
        - If fails again, give up

        This gives the rate limit time to reset while being respectful
        to the API service.
        """
        max_retries = 3

        if retry_count >= max_retries:
            print(f"    → Max retries reached. Skipping this query.")
            return []

        # Calculate wait time with exponential backoff
        # 2^retry_count gives us: 1, 2, 4, 8, 16, ...
        wait_time = 2 ** retry_count

        print(f"    → Rate limited. Waiting {wait_time}s before retry...")
        time.sleep(wait_time)

        # Try the search again
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=self.search_depth
            )
            return self._parse_results(response, query)

        except Exception as e:
            if "rate limit" in str(e).lower():
                # Still rate limited, try again with longer wait
                return self._handle_rate_limit(
                    query, max_results, search_type, retry_count + 1
                )
            else:
                # Different error, give up
                print(f"    → Error after retry: {str(e)}")
                return []

    def get_search_quality_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall quality score for search results.

        Args:
            results: List of search results

        Returns:
            Average quality score (0-1)

        Learning Note - Quality Metrics:
        -------------------------------
        Not all search results are equally useful. We can measure quality by:
        - Relevance scores from Tavily
        - Content length (longer usually means more substance)
        - Number of results found
        - Diversity of sources

        This helps us determine if a search was successful and whether
        we should try additional queries.
        """
        if not results:
            return 0.0

        # Calculate average relevance score
        avg_score = sum(r.get('score', 0) for r in results) / len(results)

        # Calculate average content length (normalized to 0-1)
        avg_length = sum(len(r.get('content', '')) for r in results) / len(results)
        length_score = min(avg_length / 1000, 1.0)  # Cap at 1.0

        # Combined quality score (weighted average)
        quality_score = (avg_score * 0.7) + (length_score * 0.3)

        return round(quality_score, 2)


# Example usage (for testing)
if __name__ == "__main__":
    """
    This code only runs if you execute this file directly:
        python modules/web_searcher.py

    It's useful for testing the module independently.
    """
    try:
        # Initialize searcher
        searcher = TavilySearcher(search_depth="basic")

        # Test single search
        print("\n" + "="*60)
        print("Testing single search...")
        print("="*60)
        results = searcher.search("Python programming best practices", max_results=3)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Score: {result['score']}")
            print(f"   Content preview: {result['content'][:100]}...")

        # Test batch search
        print("\n" + "="*60)
        print("Testing batch search...")
        print("="*60)
        queries = [
            "machine learning basics",
            "deep learning applications"
        ]
        all_results = searcher.batch_search(queries, max_results_per_query=2)

        quality = searcher.get_search_quality_score(all_results)
        print(f"\nOverall search quality: {quality}")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Make sure TAVILY_API_KEY is set in your .env file")
