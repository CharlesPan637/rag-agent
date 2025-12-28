"""
Web Content Processor Module

This module processes web search results for storage in the vector database.
It handles content cleaning, intelligent chunking, and metadata enrichment.

Learning Note - Why Process Web Content Differently?
----------------------------------------------------
Web content is different from uploaded documents:

Documents (PDF, DOCX):
- Well-structured (headings, paragraphs, formatting)
- Complete and self-contained
- Typically formal and edited
- Clear beginning, middle, end

Web Content:
- May have HTML artifacts, navigation elements
- Often incomplete snippets from search results
- Varied quality and formatting
- Mixed content types (articles, blogs, forums)

Therefore, web content needs different processing:
- More aggressive cleaning
- Context-aware chunking
- Rich metadata (URLs, dates, relevance scores)
- Source tracking for citations
"""

import re
from typing import List, Dict, Any
from datetime import datetime

from config import settings


class WebContentChunker:
    """
    Processes web search results into chunks suitable for vector storage.

    This class handles the complete pipeline:
    1. Clean content (remove artifacts, normalize whitespace)
    2. Chunk content (split into manageable pieces)
    3. Enrich with metadata (URLs, titles, dates, scores)
    4. Format for storage (prepare for ChromaDB)

    Learning Note - The Chunking Pipeline:
    -------------------------------------
    Raw search result → Clean → Chunk → Add Metadata → Ready for storage

    Each step is crucial:
    - Cleaning: Removes noise that would confuse the AI
    - Chunking: Splits into sizes that work well with embeddings
    - Metadata: Enables source tracking and citations
    - Formatting: Ensures compatibility with vector database
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the web content chunker.

        Args:
            chunk_size: Size of chunks in characters (default: from settings)
            chunk_overlap: Overlap between chunks (default: from settings)

        Learning Note - Why Have Defaults?
        ----------------------------------
        Default values make the code easier to use:

            chunker = WebContentChunker()  # Uses sensible defaults

        But you can still customize when needed:

            chunker = WebContentChunker(chunk_size=500, chunk_overlap=100)

        This is called "providing good defaults" - make the common case easy!
        """
        self.chunk_size = chunk_size or settings.WEB_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.WEB_CHUNK_OVERLAP

        print(f"✓ Web Content Chunker initialized "
              f"(size: {self.chunk_size}, overlap: {self.chunk_overlap})")

    def process_search_results(
        self,
        search_results: List[Dict[str, Any]],
        research_topic: str
    ) -> List[Dict[str, Any]]:
        """
        Process a list of search results into chunks ready for storage.

        Args:
            search_results: List of search results from TavilySearcher
            research_topic: The original research topic (for metadata)

        Returns:
            List of chunk dictionaries with text and metadata

        Learning Note - The Main Pipeline:
        ----------------------------------
        This is the "orchestrator" method that coordinates all the processing:

        For each search result:
          1. Clean the content
          2. Chunk into smaller pieces
          3. Add rich metadata
          4. Append to output list

        This is a common pattern in data processing:
        - Read/receive data
        - Transform data (clean, chunk, enrich)
        - Output processed data
        """
        all_chunks = []

        print(f"\n📝 Processing {len(search_results)} search results...")

        for idx, result in enumerate(search_results, 1):
            # Extract basic info
            content = result.get('content', '')
            url = result.get('url', '')
            title = result.get('title', 'Untitled')

            # Skip if content is too short to be useful
            if len(content) < 100:
                print(f"  {idx}. Skipping (content too short): {title[:40]}...")
                continue

            # Clean the content
            cleaned_content = self._clean_content(content)

            # Chunk the content
            chunks = self._chunk_content(cleaned_content)

            print(f"  {idx}. Processing: {title[:40]}... ({len(chunks)} chunks)")

            # Create chunk objects with metadata
            for chunk_idx, chunk_text in enumerate(chunks):
                # Generate a filename-like identifier from URL for compatibility with vector store
                # Extract domain from URL for a clean identifier
                import re
                domain_match = re.search(r'https?://([^/]+)', url)
                domain = domain_match.group(1) if domain_match else 'web_source'
                filename = f"{domain}_{idx}"

                # Prepare metadata (ChromaDB doesn't accept None values)
                metadata = {
                    'source_type': 'web_search',
                    'filename': filename,  # For vector store compatibility
                    'url': url,
                    'title': title,
                    'search_query': result.get('search_query', ''),
                    'relevance_score': result.get('score', 0.0),
                    'retrieved_date': result.get('retrieved_date', datetime.now().isoformat()),
                    'upload_date': datetime.now().isoformat(),  # For vector store compatibility
                    'chunk_id': chunk_idx,
                    'total_chunks': len(chunks),
                    'research_topic': research_topic,
                    'research_date': datetime.now().isoformat()
                }

                # Add published_date only if it's not None
                published_date = result.get('published_date')
                if published_date:
                    metadata['published_date'] = published_date

                chunk_obj = {
                    'text': chunk_text,
                    'metadata': metadata
                }
                all_chunks.append(chunk_obj)

        print(f"\n✓ Created {len(all_chunks)} total chunks from {len(search_results)} sources")
        return all_chunks

    def _clean_content(self, content: str) -> str:
        """
        Clean web content by removing artifacts and normalizing text.

        Args:
            content: Raw content from search result

        Returns:
            Cleaned content

        Learning Note - Content Cleaning Strategies:
        -------------------------------------------
        Even though Tavily provides clean content, we still need to:

        1. Normalize whitespace (multiple spaces → single space)
        2. Remove excessive line breaks
        3. Strip leading/trailing whitespace
        4. Remove special characters that don't add meaning
        5. Fix encoding issues (smart quotes, em dashes)

        Why? Because LLMs work better with clean, normalized text.
        Noise in the input = noise in the output!
        """
        if not content:
            return ""

        # Replace multiple whitespace with single space
        # \s matches any whitespace (space, tab, newline)
        # \s+ matches one or more whitespace characters
        # We replace them all with a single space
        content = re.sub(r'\s+', ' ', content)

        # Remove common HTML entities that sometimes slip through
        # These are special characters encoded for HTML
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&mdash;': '—',
            '&ndash;': '–'
        }
        for entity, replacement in html_entities.items():
            content = content.replace(entity, replacement)

        # Remove URLs from content (they're already in metadata)
        # This regex matches http:// or https:// URLs
        content = re.sub(r'http[s]?://\S+', '', content)

        # Remove email addresses (privacy and not useful for research)
        content = re.sub(r'\S+@\S+', '', content)

        # Remove excessive punctuation (more than 2 in a row)
        content = re.sub(r'([!?.;,]){3,}', r'\1\1', content)

        # Strip leading and trailing whitespace
        content = content.strip()

        return content

    def _chunk_content(self, content: str) -> List[str]:
        """
        Split content into overlapping chunks.

        Args:
            content: Cleaned content to chunk

        Returns:
            List of content chunks

        Learning Note - Why Chunking Matters:
        ------------------------------------
        Embedding models work best with text of a certain length:
        - Too short (< 100 chars): Not enough context for meaningful embedding
        - Too long (> 1000 chars): Embedding loses specificity

        Overlapping chunks ensure important information isn't split:

        Chunk 1: "...the process of photosynthesis involves..."
        Chunk 2: "...involves converting light energy into..."
                 ^ Overlap ensures continuity

        Without overlap, if a user's query relates to content at a
        chunk boundary, we might miss the relevant information!
        """
        if not content or len(content) <= self.chunk_size:
            return [content] if content else []

        chunks = []
        start = 0
        prev_end = 0

        while start < len(content):
            # Calculate end position for this chunk
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a sentence
            if end < len(content):
                # Look for sentence endings near the end position
                # We search backwards from 'end' for up to chunk_size/4 characters
                search_start = max(end - self.chunk_size // 4, start)
                sentence_end = self._find_sentence_boundary(
                    content, search_start, end
                )

                # If we found a good sentence boundary, use it
                if sentence_end > start:
                    end = sentence_end

            # Extract the chunk
            chunk = content[start:end].strip()

            # Only add non-empty chunks
            if chunk:
                chunks.append(chunk)

            # Move start position forward
            # Subtract overlap to create overlapping chunks
            new_start = end - self.chunk_overlap

            # Prevent infinite loop if we're not making progress
            if new_start <= prev_end:
                start = end
            else:
                start = new_start

            prev_end = end

        return chunks

    def _find_sentence_boundary(
        self,
        content: str,
        search_start: int,
        search_end: int
    ) -> int:
        """
        Find the best sentence boundary within a range.

        Args:
            content: The full content string
            search_start: Start of search range
            search_end: End of search range

        Returns:
            Position of sentence boundary, or search_end if none found

        Learning Note - Sentence Boundaries:
        -----------------------------------
        We want chunks to end at natural boundaries (sentence endings)
        rather than in the middle of sentences. This makes chunks more
        coherent and easier for the LLM to understand.

        Sentence endings to look for:
        - Period followed by space and capital letter: ". A"
        - Exclamation or question mark: "! " or "? "
        - Period at end of string

        We search backwards from the desired endpoint to find the
        nearest sentence ending.
        """
        # Search backwards from end for sentence terminators
        search_region = content[search_start:search_end]

        # Find all sentence terminators
        # These regex patterns match: ". " or "! " or "? "
        terminators = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        best_pos = -1

        for terminator in terminators:
            # Find last occurrence of this terminator
            pos = search_region.rfind(terminator)
            if pos > best_pos:
                best_pos = pos

        if best_pos != -1:
            # Return absolute position (not relative to search_start)
            # +len(terminator) to include the punctuation
            return search_start + best_pos + len(terminator)

        # No sentence boundary found, return original end
        return search_end

    def create_citation(
        self,
        metadata: Dict[str, Any],
        index: int = None
    ) -> str:
        """
        Create a formatted citation from metadata.

        Args:
            metadata: Chunk metadata dictionary
            index: Optional citation number

        Returns:
            Formatted citation string

        Learning Note - Why Citations Matter:
        ------------------------------------
        When presenting research findings, we MUST cite sources:
        1. Builds trust (users can verify information)
        2. Gives credit (respects content creators)
        3. Enables deeper exploration (users can read more)
        4. Shows transparency (clear where info comes from)

        Good citation format:
        [1: Article Title](https://example.com)
        Retrieved: 2024-01-15

        This includes:
        - Citation number (for reference in text)
        - Title (what the source is)
        - URL (where to find it)
        - Date (when it was retrieved)
        """
        title = metadata.get('title', 'Untitled')
        url = metadata.get('url', '')
        retrieved = metadata.get('retrieved_date', '')

        # Truncate long titles
        if len(title) > 60:
            title = title[:57] + "..."

        # Format citation
        if index is not None:
            citation = f"[{index}. {title}]({url})"
        else:
            citation = f"[{title}]({url})"

        # Add retrieval date if available
        if retrieved:
            # Extract just the date part (YYYY-MM-DD)
            date_str = retrieved[:10] if len(retrieved) >= 10 else retrieved
            citation += f"\nRetrieved: {date_str}"

        # Add relevance score if available
        score = metadata.get('relevance_score')
        if score:
            citation += f" | Relevance: {score:.2f}"

        return citation

    def deduplicate_by_url(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks from the same URL.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Deduplicated list of chunks

        Learning Note - Why Deduplication Matters:
        -----------------------------------------
        When searching with multiple queries, we often get the same
        source multiple times:

        Query 1: "AI healthcare" → finds article X
        Query 2: "machine learning medicine" → also finds article X

        Storing duplicate content:
        - Wastes storage space
        - Skews retrieval (same source appears multiple times)
        - Costs more (unnecessary embeddings)
        - Reduces diversity in results

        We deduplicate by URL because if two chunks have the same URL,
        they're from the same source even if the content differs slightly.
        """
        seen_urls = set()
        unique_chunks = []

        for chunk in chunks:
            url = chunk.get('metadata', {}).get('url', '')

            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_chunks.append(chunk)

        removed = len(chunks) - len(unique_chunks)
        if removed > 0:
            print(f"  Removed {removed} duplicate sources")

        return unique_chunks


# Example usage (for testing)
if __name__ == "__main__":
    """
    Test the web content processor independently.
    """
    print("Testing Web Content Processor...")
    print("=" * 60)

    # Create sample search results
    sample_results = [
        {
            'title': 'Understanding AI in Healthcare',
            'url': 'https://example.com/ai-healthcare',
            'content': '''
                Artificial intelligence is revolutionizing healthcare in multiple ways.
                Machine learning algorithms can now detect diseases from medical images
                with accuracy comparable to human experts. Natural language processing
                helps extract insights from medical records. Predictive models identify
                patients at risk of complications. These advances are improving patient
                outcomes while reducing costs.
            ''',
            'score': 0.95,
            'search_query': 'AI healthcare applications',
            'retrieved_date': '2024-01-15T10:00:00'
        },
        {
            'title': 'Short Content Example',
            'url': 'https://example.com/short',
            'content': 'Too short.',
            'score': 0.5,
            'search_query': 'test',
            'retrieved_date': '2024-01-15T10:00:00'
        }
    ]

    # Initialize processor
    processor = WebContentChunker(chunk_size=200, chunk_overlap=50)

    # Process results
    chunks = processor.process_search_results(
        sample_results,
        research_topic="AI in Healthcare"
    )

    print(f"\nProcessed Results:")
    print(f"Total chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Text: {chunk['text'][:100]}...")
        print(f"  URL: {chunk['metadata']['url']}")
        print(f"  Title: {chunk['metadata']['title']}")
        print(f"  Score: {chunk['metadata']['relevance_score']}")
        print(f"  Chunk ID: {chunk['metadata']['chunk_id']}/{chunk['metadata']['total_chunks']}")

        # Test citation creation
        citation = processor.create_citation(chunk['metadata'], index=i)
        print(f"  Citation: {citation}")
        print()
