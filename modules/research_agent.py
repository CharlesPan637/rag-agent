"""
Research Agent Module

This is the main orchestrator for the web research workflow. It coordinates
all components to conduct comprehensive research on any topic.

Learning Note - What is an Orchestrator?
----------------------------------------
An orchestrator is a component that coordinates multiple other components
to accomplish a complex task. Think of it like a conductor leading an orchestra:

- The conductor doesn't play instruments
- But coordinates all musicians to create music
- Similarly, the Research Agent doesn't do searches or processing itself
- But coordinates searcher, processor, storage, and synthesis components

This is a common pattern in software architecture called "coordination" or
"orchestration." It keeps code organized and maintainable.

The Research Agent Workflow:
1. Take user's research topic
2. Generate diverse search queries (query expansion)
3. Execute searches via TavilySearcher
4. Process results via WebContentChunker
5. Deduplicate to remove redundant content
6. Store chunks in ChromaDB
7. Retrieve most relevant chunks for topic
8. Synthesize comprehensive report via RAG

Total time: ~20-30 seconds per research session
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from openai import OpenAI

from config import settings
from modules.web_searcher import TavilySearcher
from modules.web_content_processor import WebContentChunker
from modules.vector_store import ChromaDBStore
from modules.rag_query import RAGEngine


class ResearchAgent:
    """
    Orchestrates the complete web research workflow.

    This agent takes a research topic and produces a comprehensive report
    by searching the web, processing results, and synthesizing findings.

    Learning Note - Agent Architecture:
    ----------------------------------
    This class follows the "composition" pattern:
    - Instead of inheriting from multiple classes
    - It "has-a" relationship with other components
    - searcher, processor, vector_store, rag_engine are all "composed" together

    Benefits:
    - Each component can be tested independently
    - Easy to swap implementations (e.g., different search APIs)
    - Clear separation of concerns
    - More flexible than inheritance
    """

    def __init__(
        self,
        vector_store: ChromaDBStore,
        search_depth: str = "basic"
    ):
        """
        Initialize the Research Agent.

        Args:
            vector_store: ChromaDB store for web research (separate from documents)
            search_depth: "basic" or "advanced" search depth

        Learning Note - Dependency Injection:
        ------------------------------------
        We pass vector_store as a parameter instead of creating it here.
        This is called "dependency injection" and is a best practice:

        Benefits:
        - Easier testing (can pass mock/test vector store)
        - More flexible (can use different stores)
        - Clear dependencies (obvious what the agent needs)
        - Better for reusability
        """
        self.vector_store = vector_store
        self.search_depth = search_depth

        # Initialize components
        print("\n🤖 Initializing Research Agent...")

        # Web searcher for Tavily API
        self.searcher = TavilySearcher(search_depth=search_depth)

        # Content processor for cleaning and chunking
        self.processor = WebContentChunker()

        # RAG engine for synthesis
        self.rag_engine = RAGEngine(vector_store)

        # OpenAI client for query generation
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        print("✓ Research Agent ready!")

    def conduct_research(
        self,
        topic: str,
        num_queries: int = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Conduct complete research on a topic and generate report.

        Args:
            topic: Research topic (e.g., "AI applications in healthcare")
            num_queries: Number of search queries to generate (default: from settings)
            progress_callback: Optional callback function for progress updates
                              callback(message: str, progress_pct: int)

        Returns:
            Dictionary with:
                - report: Formatted research report (markdown)
                - sources: List of source dictionaries
                - num_chunks: Number of content chunks processed
                - search_quality: Quality score (0-1)

        Learning Note - Progress Callbacks:
        -----------------------------------
        Research takes time (20-30 seconds). Users need feedback!

        A callback is a function you pass to another function:

            def show_progress(msg, pct):
                print(f"{pct}%: {msg}")

            agent.conduct_research(topic, progress_callback=show_progress)

        The Research Agent will call show_progress() at various stages
        to update the user. This is used by the UI to show progress bars.
        """
        num_queries = num_queries or settings.RESEARCH_QUERIES_COUNT

        try:
            # Step 1: Generate diverse search queries
            self._update_progress(progress_callback, "Generating search queries...", 20)
            queries = self._generate_search_queries(topic, num_queries)
            print(f"\n📋 Generated {len(queries)} search queries:")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")

            # Step 2: Execute searches
            self._update_progress(progress_callback, "Searching web sources...", 30)
            search_results = self.searcher.batch_search(
                queries,
                max_results_per_query=settings.RESULTS_PER_QUERY
            )

            if not search_results:
                return {
                    'report': self._generate_no_results_message(topic),
                    'sources': [],
                    'num_chunks': 0,
                    'search_quality': 0.0
                }

            # Step 3: Deduplicate results
            self._update_progress(progress_callback, "Deduplicating results...", 45)
            unique_results = self._deduplicate_results(search_results)
            print(f"\n🔍 Found {len(unique_results)} unique sources")

            # Calculate search quality
            search_quality = self.searcher.get_search_quality_score(unique_results)
            print(f"📊 Search quality score: {search_quality}")

            # Step 4: Process and chunk content
            self._update_progress(progress_callback, "Processing content...", 55)
            chunks = self.processor.process_search_results(unique_results, topic)

            if not chunks:
                return {
                    'report': self._generate_no_results_message(topic),
                    'sources': [],
                    'num_chunks': 0,
                    'search_quality': search_quality
                }

            # Step 5: Store in vector database
            self._update_progress(progress_callback, "Storing in vector database...", 70)
            self._store_chunks(chunks, topic)

            # Step 6: Retrieve relevant content for synthesis
            self._update_progress(progress_callback, "Retrieving relevant content...", 80)
            relevant_chunks = self.vector_store.query(
                query_text=topic,
                n_results=15  # Get more chunks for comprehensive report
            )

            # Step 7: Synthesize report
            self._update_progress(progress_callback, "Synthesizing research report...", 85)
            report = self._synthesize_report(topic, relevant_chunks)

            # Step 8: Extract sources for citation
            sources = self._extract_sources(relevant_chunks)

            self._update_progress(progress_callback, "Complete!", 100)

            return {
                'report': report,
                'sources': sources,
                'num_chunks': len(chunks),
                'search_quality': search_quality
            }

        except Exception as e:
            import traceback
            error_msg = f"Research failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            print("\nFull traceback:")
            print(traceback.format_exc())
            return {
                'report': f"# Research Failed\n\n{error_msg}\n\nPlease try again or refine your topic.",
                'sources': [],
                'num_chunks': 0,
                'search_quality': 0.0
            }

    def _generate_search_queries(self, topic: str, num_queries: int) -> List[str]:
        """
        Generate diverse search queries from a single topic.

        Args:
            topic: Research topic
            num_queries: Number of queries to generate

        Returns:
            List of search query strings

        Learning Note - Query Expansion:
        -------------------------------
        A single search query gives limited perspective. Query expansion
        generates multiple related queries to get comprehensive coverage.

        Example: Topic = "AI in healthcare"

        Generated queries might be:
        1. "AI in healthcare applications 2024"           (direct + recent)
        2. "machine learning medical diagnosis"            (technical angle)
        3. "artificial intelligence patient care systems"  (practical angle)
        4. "AI healthcare workflow automation benefits"    (benefits angle)

        Why this works:
        - Different queries surface different sources
        - Captures multiple perspectives (technical, practical, benefits)
        - Improves overall coverage of the topic
        - Reduces bias from any single search

        This is like asking multiple experts about the same topic -
        you get a more complete picture!
        """
        try:
            # Build prompt for GPT to generate diverse queries
            prompt = f"""Generate {num_queries} diverse search queries for researching this topic: "{topic}"

Requirements:
1. Each query should explore a different angle or aspect
2. Use varied terminology (synonyms, related terms)
3. Include both broad and specific queries
4. Keep queries concise (5-8 words)
5. Make them suitable for web search

Output format: Return ONLY the queries, one per line, numbered.

Example for "quantum computing":
1. quantum computing fundamentals explained
2. quantum computers vs classical computers
3. quantum computing real world applications 2024
4. quantum algorithm development challenges"""

            # Call OpenAI GPT to generate queries
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research assistant that generates diverse, effective search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for more diversity
                max_tokens=200
            )

            # Parse the response
            queries_text = response.choices[0].message.content.strip()

            # Extract queries from numbered list
            queries = []
            for line in queries_text.split('\n'):
                # Remove numbering (e.g., "1. " or "1) ")
                line = line.strip()
                if line:
                    # Remove leading numbers and punctuation
                    import re
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if cleaned:
                        queries.append(cleaned)

            # Ensure we have the requested number of queries
            if len(queries) < num_queries:
                # If GPT didn't generate enough, add the original topic
                queries.append(topic)

            return queries[:num_queries]

        except Exception as e:
            # If query generation fails, fall back to simple variations
            print(f"Warning: Query generation failed ({e}), using fallback")
            return self._fallback_queries(topic, num_queries)

    def _fallback_queries(self, topic: str, num_queries: int) -> List[str]:
        """
        Generate fallback queries if GPT generation fails.

        This ensures research can proceed even if OpenAI API has issues.

        Learning Note - Graceful Degradation:
        ------------------------------------
        Good systems have fallbacks for when things go wrong:
        - Primary: Use GPT to generate smart queries
        - Fallback: Use simple template-based queries

        This is called "graceful degradation" - the system still works,
        just not quite as well. Better than failing completely!
        """
        templates = [
            topic,  # Original topic
            f"{topic} applications",
            f"{topic} explained",
            f"{topic} 2024",
            f"{topic} benefits challenges",
            f"{topic} best practices"
        ]
        return templates[:num_queries]

    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on URL and content similarity.

        Args:
            results: List of search results

        Returns:
            Deduplicated list

        Learning Note - Deduplication Strategy:
        --------------------------------------
        When running multiple queries, we often get the same sources:

        Query 1: "AI healthcare" → finds Article X
        Query 2: "machine learning medicine" → also finds Article X

        We deduplicate in two ways:

        1. URL-based: Same URL = same source (exact duplicates)
        2. Content-based: Very similar content = likely duplicates

        This ensures:
        - No redundant processing
        - Better diversity in results
        - More efficient storage
        - Higher quality synthesis
        """
        if not results:
            return []

        # Stage 1: Remove exact URL duplicates
        seen_urls = {}
        url_deduped = []

        for result in results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls[url] = result
                url_deduped.append(result)

        print(f"  After URL deduplication: {len(url_deduped)} results")

        # Stage 2: Remove content duplicates (optional, for better quality)
        # We could use embedding similarity here, but for speed we'll
        # use a simpler approach: just remove exact content matches
        seen_content = {}
        final_results = []

        for result in url_deduped:
            # Use first 200 chars as fingerprint
            content = result.get('content', '')[:200]
            if content and content not in seen_content:
                seen_content[content] = True
                final_results.append(result)

        if len(final_results) < len(url_deduped):
            print(f"  After content deduplication: {len(final_results)} results")

        # Limit to MAX_RESEARCH_RESULTS for cost control
        if len(final_results) > settings.MAX_RESEARCH_RESULTS:
            print(f"  Limiting to top {settings.MAX_RESEARCH_RESULTS} results")
            # Sort by relevance score and take top N
            final_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            final_results = final_results[:settings.MAX_RESEARCH_RESULTS]

        return final_results

    def _store_chunks(self, chunks: List[Dict[str, Any]], topic: str):
        """
        Store processed chunks in vector database.

        Args:
            chunks: List of chunk dictionaries from processor
            topic: Research topic (for collection management)

        Learning Note - Vector Storage:
        ------------------------------
        Each chunk is converted to a vector (embedding) and stored.

        Why store in vector database?
        - Enables semantic search (find by meaning, not just keywords)
        - Fast retrieval of relevant content
        - Persistent storage (survives between sessions)
        - Efficient similarity comparisons

        The vector store maps:
        Text → Embedding (vector) → Storage

        Later we can search:
        Query → Embedding → Find similar vectors → Return matching text
        """
        print(f"\n💾 Storing {len(chunks)} chunks in vector database...")

        # Store in vector database
        # add_documents expects chunks in format: [{'text': '...', 'metadata': {...}}, ...]
        self.vector_store.add_documents(chunks, show_progress=False)

        print(f"✓ Storage complete")

    def _synthesize_report(
        self,
        topic: str,
        relevant_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize research report from relevant chunks.

        Args:
            topic: Research topic
            relevant_chunks: Retrieved chunks from vector store

        Returns:
            Formatted markdown report

        Learning Note - Report Synthesis:
        --------------------------------
        This is where RAG (Retrieval Augmented Generation) happens:

        1. We retrieved relevant chunks (R = Retrieval)
        2. Now we augment our prompt with this context (A = Augmented)
        3. GPT generates a comprehensive report (G = Generation)

        The LLM doesn't "know" about the web content we found.
        We provide it as context so it can synthesize findings.

        This is more reliable than asking GPT directly because:
        - Grounded in actual sources (less hallucination)
        - Up-to-date information (from recent web search)
        - Citeable sources (transparency)
        - Factual basis (not just model knowledge)
        """
        if not relevant_chunks:
            return self._generate_no_results_message(topic)

        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk.get('metadata', {})
            text = chunk.get('document', '')
            title = metadata.get('title', 'Unknown')
            url = metadata.get('url', '')

            context_parts.append(f"[Source {i}: {title}]\n{text}\n")

        context = "\n\n".join(context_parts)

        # Build synthesis prompt
        prompt = f"""You are a research assistant synthesizing findings from multiple web sources.

Your task: Create a comprehensive research report on the topic "{topic}"

Report Requirements:
1. Start with an **Executive Summary** (2-3 sentences)
2. Present **Key Findings** organized by themes (3-5 main points)
3. Provide **Detailed Analysis** with insights from multiple sources
4. Cite sources using [Source N] notation throughout
5. Note areas of consensus and disagreement if present
6. Be objective and balanced

Context from Web Research:
{context}

Generate a well-structured research report in markdown format."""

        try:
            # Generate report using OpenAI GPT
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst who synthesizes information from multiple sources into comprehensive, well-cited reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Balanced creativity and consistency
                max_tokens=1500   # Allow for detailed reports
            )

            report = response.choices[0].message.content.strip()

            # Post-process report to add hyperlinks to source citations
            report = self._add_source_hyperlinks(report, relevant_chunks)

            return report

        except Exception as e:
            print(f"Warning: Report synthesis failed ({e})")
            return self._generate_fallback_report(topic, relevant_chunks)

    def _add_source_hyperlinks(
        self,
        report: str,
        relevant_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Convert [Source N] citations in report to clickable hyperlinks.

        Args:
            report: Generated report with [Source N] citations
            relevant_chunks: List of chunks with metadata including URLs

        Returns:
            Report with [Source N] replaced by [Source N: Title](URL)

        Learning Note - Making Citations Clickable:
        ------------------------------------------
        The GPT-generated report includes citations like [Source 1], [Source 2].
        We replace these with markdown hyperlinks so users can:
        - Click directly to visit the original source
        - Verify the information
        - Read more context

        Example transformation:
        Before: "According to [Source 1], quantum computing..."
        After:  "According to [Source 1: IBM Quantum Report](https://ibm.com/quantum), quantum computing..."
        """
        import re

        # Create a mapping of source numbers to URLs and titles
        source_map = {}
        for i, chunk in enumerate(relevant_chunks, 1):
            metadata = chunk.get('metadata', {})
            url = metadata.get('url', '')
            title = metadata.get('title', 'Source')

            # Truncate long titles for inline citations
            if len(title) > 50:
                title = title[:47] + "..."

            source_map[i] = {'url': url, 'title': title}

        # Replace [Source N] with clickable hyperlinks
        # Pattern matches: [Source 1], [Source 2], etc.
        def replace_citation(match):
            source_num = int(match.group(1))

            if source_num in source_map:
                source_info = source_map[source_num]
                url = source_info['url']
                title = source_info['title']

                # Create markdown hyperlink
                return f"[Source {source_num}: {title}]({url})"
            else:
                # Keep original if source number not found
                return match.group(0)

        # Use regex to find and replace all [Source N] patterns
        report = re.sub(r'\[Source (\d+)\]', replace_citation, report)

        return report

    def _generate_fallback_report(
        self,
        topic: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a simple report if synthesis fails.

        Fallback strategy for graceful degradation.
        """
        report = f"# Research Report: {topic}\n\n"
        report += "## Summary\n\n"
        report += "Research findings compiled from multiple web sources:\n\n"

        for i, chunk in enumerate(chunks[:5], 1):  # Top 5 chunks
            metadata = chunk.get('metadata', {})
            text = chunk.get('document', '')
            title = metadata.get('title', 'Unknown Source')

            report += f"### Finding {i}: {title}\n\n"
            report += f"{text[:300]}...\n\n"

        return report

    def _generate_no_results_message(self, topic: str) -> str:
        """Generate message when no results are found."""
        return f"""# Research Report: {topic}

## No Results Found

Unfortunately, no relevant sources were found for this research topic.

### Suggestions:
- Try rephrasing your topic with different terms
- Make your topic more specific or more general
- Check for typos in your topic
- Ensure your Tavily API key is valid

### Examples of Good Topics:
- "Latest developments in quantum computing"
- "AI applications in healthcare 2024"
- "Climate change mitigation strategies"
- "Microservices architecture best practices"
"""

    def _extract_sources(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract unique sources from chunks for citation list.

        Args:
            chunks: Retrieved chunks with metadata

        Returns:
            List of unique source dictionaries

        Learning Note - Source Extraction:
        ---------------------------------
        Multiple chunks can come from the same source (URL).
        We want a unique list of sources for the citation section.

        This function:
        1. Collects all unique URLs
        2. Gets metadata for each source
        3. Sorts by relevance
        4. Returns deduplicated source list
        """
        seen_urls = {}

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            url = metadata.get('url', '')

            if url and url not in seen_urls:
                seen_urls[url] = {
                    'title': metadata.get('title', 'Untitled'),
                    'url': url,
                    'relevance_score': metadata.get('relevance_score', 0.0),
                    'retrieved_date': metadata.get('retrieved_date', ''),
                    'published_date': metadata.get('published_date', None)
                }

        # Convert to list and sort by relevance
        sources = list(seen_urls.values())
        sources.sort(key=lambda x: x['relevance_score'], reverse=True)

        return sources

    def _update_progress(
        self,
        callback: Optional[Callable],
        message: str,
        percent: int
    ):
        """
        Update progress if callback is provided.

        Args:
            callback: Progress callback function
            message: Progress message
            percent: Progress percentage (0-100)
        """
        if callback:
            callback(message, percent)


# Example usage (for testing)
if __name__ == "__main__":
    """
    Test the Research Agent independently.
    """
    print("Testing Research Agent...")
    print("=" * 60)

    try:
        # Initialize vector store
        from modules.vector_store import ChromaDBStore
        vector_store = ChromaDBStore(collection_name="test_research")

        # Initialize agent
        agent = ResearchAgent(vector_store, search_depth="basic")

        # Define progress callback
        def show_progress(msg, pct):
            print(f"[{pct}%] {msg}")

        # Conduct research
        print("\nStarting research on 'Python programming best practices'...")
        result = agent.conduct_research(
            topic="Python programming best practices",
            num_queries=2,  # Small number for testing
            progress_callback=show_progress
        )

        # Display results
        print("\n" + "=" * 60)
        print("RESEARCH REPORT:")
        print("=" * 60)
        print(result['report'])

        print("\n" + "=" * 60)
        print(f"SOURCES ({len(result['sources'])}):")
        print("=" * 60)
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['title']}")
            print(f"   {source['url']}")
            print(f"   Relevance: {source['relevance_score']}")
            print()

        print(f"\nChunks processed: {result['num_chunks']}")
        print(f"Search quality: {result['search_quality']}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure all API keys are configured in .env")
