"""
RAG Query Engine for the RAG Agent.

This module implements the core RAG (Retrieval Augmented Generation) pipeline:
1. Take user's question
2. Retrieve relevant document chunks from vector store
3. Format chunks as context
4. Build prompt with context and question
5. Send to Claude for answer generation
6. Return answer with sources

Learning Note - What is RAG?
----------------------------
RAG = Retrieval Augmented Generation

Problem: LLMs don't know about YOUR specific documents
Solution: Give the LLM relevant context from your documents

Steps:
1. User asks: "What is the refund policy?"
2. System finds relevant chunks from documents
3. System sends to LLM: "Given this context: [chunks], answer: [question]"
4. LLM generates answer based on provided context

Benefits:
- Answers are grounded in your documents (less hallucination)
- Can cite sources (transparency)
- No need to fine-tune the model (expensive and complex)
- Can update documents without retraining
"""

import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError

from config import settings
from modules.vector_store import ChromaDBStore


class RAGEngine:
    """
    Orchestrates the RAG pipeline: retrieval + generation.

    This is the "brain" of the system that ties everything together.
    """

    def __init__(self, vector_store: ChromaDBStore):
        """
        Initialize the RAG engine.

        Args:
            vector_store: ChromaDB store for retrieval

        Learning Note:
        This class follows the "dependency injection" pattern.
        Instead of creating its own vector_store, it receives one.
        Benefits:
        - Easier to test (can pass mock objects)
        - More flexible (can use different stores)
        - Clearer dependencies
        """
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL

    def query(
        self,
        question: str,
        n_chunks: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve context and generate answer.

        Args:
            question: User's question
            n_chunks: Number of chunks to retrieve (default: from settings)
            return_sources: Whether to include source chunks in response

        Returns:
            Dict with:
            - 'answer': Generated answer from Claude
            - 'sources': Retrieved chunks (if return_sources=True)
            - 'question': Original question

        Learning Note - The RAG Pipeline:
        This is where everything comes together!

        Example:
            >>> engine = RAGEngine(vector_store)
            >>> result = engine.query("What is the refund policy?")
            >>> print(result['answer'])
            >>> print(f"Sources: {len(result['sources'])} chunks")
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if n_chunks is None:
            n_chunks = settings.DEFAULT_RETRIEVAL_COUNT

        # Step 1: Retrieve relevant chunks
        print(f"Retrieving top {n_chunks} relevant chunks...")
        retrieved_chunks = self.vector_store.query(
            query_text=question,
            n_results=n_chunks
        )

        if not retrieved_chunks:
            # No documents in database
            return {
                'answer': "I don't have any documents to search. Please upload some documents first.",
                'sources': [],
                'question': question
            }

        # Step 2: Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Step 3: Build prompt with context and question
        prompt = self._build_prompt(question, context)

        # Step 4: Generate answer using OpenAI GPT
        print("Generating answer with OpenAI GPT...")
        answer = self._generate_answer(prompt)

        # Step 5: Return results
        result = {
            'answer': answer,
            'question': question
        }

        if return_sources:
            result['sources'] = retrieved_chunks

        return result

    def synthesize_research_report(
        self,
        topic: str,
        context_chunks: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> str:
        """
        Generate a comprehensive research report from retrieved context chunks.

        This method is specifically designed for research synthesis, as opposed
        to the Q&A functionality in the query() method.

        Args:
            topic: Research topic
            context_chunks: Retrieved chunks from vector store with metadata
            max_retries: Number of retries for API calls

        Returns:
            str: Formatted research report in markdown

        Learning Note - Research vs Q&A:
        --------------------------------
        Q&A Mode:
        - User asks specific question
        - System provides direct answer
        - Focus on precision and brevity
        - Citations are helpful but optional

        Research Mode:
        - User provides broad topic
        - System synthesizes comprehensive report
        - Focus on breadth and depth
        - Multiple sources integrated
        - Citations are mandatory
        - Organized structure required

        Different modes need different prompting strategies!

        Example:
            Q&A: "What is photosynthesis?"
            → Brief explanation with source

            Research: "photosynthesis"
            → Executive summary, key findings, detailed analysis,
              multiple perspectives, applications, challenges, sources
        """
        if not context_chunks:
            return self._generate_no_results_report(topic)

        # Format context with source attribution
        context = self._format_research_context(context_chunks)

        # Build research-specific prompt
        prompt = self._build_research_prompt(topic, context)

        # Generate report
        print("Synthesizing comprehensive research report...")
        report = self._generate_answer(prompt, max_retries=max_retries)

        return report

    def _format_research_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks specifically for research synthesis.

        Args:
            chunks: Retrieved chunks with metadata

        Returns:
            Formatted context string with source attribution

        Learning Note - Research Context Formatting:
        ------------------------------------------
        For research, we need richer context than Q&A:
        - Source titles and URLs for citation
        - Publication/retrieval dates for recency
        - Relevance scores for prioritization
        - Clear source boundaries for attribution

        Format:
        [Source 1: Title - URL]
        Content...

        [Source 2: Title - URL]
        Content...

        This makes it easy for the LLM to:
        1. Identify different sources
        2. Cite them properly
        3. Compare/contrast viewpoints
        4. Detect consensus or disagreement
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            text = chunk.get('document', '')

            # Extract source information
            title = metadata.get('title', 'Unknown Source')
            url = metadata.get('url', '')
            score = metadata.get('relevance_score', 0.0)
            date = metadata.get('published_date') or metadata.get('retrieved_date', '')

            # Format source header
            source_header = f"[Source {i}: {title}]"
            if url:
                source_header += f"\nURL: {url}"
            if score:
                source_header += f"\nRelevance: {score:.2f}"
            if date:
                date_str = date[:10] if len(date) >= 10 else date
                source_header += f"\nDate: {date_str}"

            # Combine header and content
            context_parts.append(f"{source_header}\n{text}\n")

        return "\n---\n\n".join(context_parts)

    def _build_research_prompt(self, topic: str, context: str) -> str:
        """
        Build prompt specifically for research synthesis.

        Args:
            topic: Research topic
            context: Formatted context from multiple sources

        Returns:
            Complete prompt for research synthesis

        Learning Note - Research Prompt Engineering:
        ------------------------------------------
        Effective research prompts should:

        1. Set clear role ("research assistant")
        2. Define task structure (executive summary, findings, analysis)
        3. Require source citations throughout
        4. Request objective, balanced analysis
        5. Specify output format (markdown)
        6. Emphasize synthesis over summarization
        7. Encourage identifying patterns and themes

        Synthesis vs Summarization:
        - Summarization: Condense each source
        - Synthesis: Combine insights across sources,
                     identify themes, note agreements/disagreements

        Research synthesis is higher-order thinking!
        """
        system_instructions = """You are an expert research analyst who synthesizes information from multiple sources into comprehensive, well-structured reports.

Your responsibilities:
1. Analyze and integrate information from all provided sources
2. Identify key themes, patterns, and insights
3. Note areas of consensus and disagreement among sources
4. Cite sources explicitly using [Source N] notation
5. Provide balanced, objective analysis
6. Organize information logically by theme or concept
7. Distinguish between facts and interpretations

Writing style:
- Professional and academic but accessible
- Well-structured with clear headings
- Evidence-based with explicit citations
- Comprehensive yet concise"""

        user_prompt = f"""Research Topic: {topic}

Create a comprehensive research report following this structure:

## Executive Summary
Provide a concise overview (2-3 sentences) of the key findings.

## Key Findings
Present the 3-5 most important insights, organized by theme.
Each finding should cite relevant sources [Source N].

## Detailed Analysis
Provide in-depth discussion of the topic:
- Important concepts and definitions
- Major perspectives or approaches
- Current state and recent developments
- Challenges or controversies
- Practical applications or implications
- Future directions or trends

Throughout this section, cite sources explicitly.
When sources agree, note the consensus.
When sources disagree, present multiple perspectives fairly.

## Conclusion
Synthesize the overall findings (1-2 paragraphs).

Context from Web Research:
{context}

Generate the research report in markdown format following the structure above."""

        return f"{system_instructions}\n\n{user_prompt}"

    def _generate_no_results_report(self, topic: str) -> str:
        """
        Generate a report when no relevant sources are found.

        Args:
            topic: Research topic

        Returns:
            Formatted markdown report explaining no results

        Learning Note - Graceful Failure Messages:
        -----------------------------------------
        When things go wrong, provide helpful feedback:
        - Explain what happened (no sources found)
        - Suggest why it might have happened (too specific, typo)
        - Offer actionable next steps (rephrase, try different terms)
        - Show examples of what works

        Good error messages turn frustration into forward progress!
        """
        return f"""# Research Report: {topic}

## No Relevant Sources Found

Unfortunately, the search did not return relevant information for this topic.

### Possible Reasons:
- Topic may be too specific or too narrow
- Search terms might need adjustment
- Topic may be very new with limited online coverage
- There may be a typo in the topic

### Suggestions:
1. **Broaden your topic**: Try more general terms
   - Instead of "quantum error correction in topological qubits"
   - Try "quantum computing error correction"

2. **Use different terminology**: Try synonyms or related terms
   - Instead of "ML"
   - Try "machine learning"

3. **Check spelling**: Ensure all terms are spelled correctly

4. **Try concrete applications**: Instead of abstract concepts
   - Instead of "distributed systems theory"
   - Try "distributed systems real world applications"

### Examples of Effective Research Topics:
- "artificial intelligence healthcare applications 2024"
- "climate change mitigation strategies"
- "microservices architecture best practices"
- "quantum computing current state"
- "blockchain technology use cases"

Please try again with a refined topic!
"""

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string for the prompt.

        Args:
            chunks: Retrieved chunks with text and metadata

        Returns:
            str: Formatted context

        Learning Note - Context Formatting:
        Good context formatting helps the LLM:
        - Understand where each piece comes from
        - Cite sources accurately
        - Distinguish between different documents

        We include:
        - Source filename
        - Chunk number
        - The actual text

        Example output:
        ---
        [Source: document.pdf, Chunk 3]
        The refund policy states that...

        [Source: guide.docx, Chunk 1]
        To request a refund, please...
        ---
        """
        context_parts = []

        for idx, chunk in enumerate(chunks, start=1):
            text = chunk['text']
            metadata = chunk['metadata']
            filename = metadata.get('filename', 'Unknown')
            chunk_id = metadata.get('chunk_id', '?')

            # Format each chunk with source attribution
            context_parts.append(
                f"[Source: {filename}, Chunk {chunk_id}]\n{text}"
            )

        # Join all chunks with clear separation
        context = "\n\n---\n\n".join(context_parts)

        return context

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the complete prompt for Claude.

        Args:
            question: User's question
            context: Formatted context from retrieved chunks

        Returns:
            str: Complete prompt

        Learning Note - Prompt Engineering:
        --------------------------------
        Prompt engineering is the art of writing prompts that get good results.

        Key principles for RAG prompts:
        1. Clear role definition ("You are a helpful assistant...")
        2. Explicit grounding ("Answer ONLY based on provided context")
        3. Handling missing info ("If not in context, say so")
        4. Format expectations ("Be concise but comprehensive")
        5. Source attribution ("Cite sources when possible")

        Good prompting:
        - Reduces hallucination (making things up)
        - Increases accuracy
        - Provides transparency (user knows where info comes from)
        """
        # System-level instructions
        system_instructions = """You are a helpful AI assistant that answers questions based on provided document context.

Your guidelines:
1. Answer ONLY based on the provided context from the documents
2. If the answer is not in the context, clearly state: "I don't have enough information in the provided documents to answer this question."
3. When possible, mention which document your answer comes from
4. Be concise but comprehensive
5. If you're uncertain, express appropriate uncertainty
6. Do not make up or infer information not present in the context"""

        # Build the complete prompt
        prompt = f"""{system_instructions}

Context from documents:
{context}

Question: {question}

Answer:"""

        return prompt

    def _generate_answer(
        self,
        prompt: str,
        max_retries: int = 3
    ) -> str:
        """
        Generate answer using OpenAI API with retry logic.

        Args:
            prompt: The complete prompt
            max_retries: Number of times to retry on rate limit errors

        Returns:
            str: Generated answer

        Learning Note - API Error Handling:
        ----------------------------------
        APIs can fail for many reasons:
        - Rate limits (too many requests)
        - Network errors
        - Service outages
        - Invalid requests

        Good error handling:
        1. Catch specific exceptions
        2. Retry with exponential backoff (wait longer each time)
        3. Provide helpful error messages
        4. Log details for debugging

        Example retry pattern:
        - First failure: wait 1 second, retry
        - Second failure: wait 2 seconds, retry
        - Third failure: give up, show error
        """
        retries = 0

        while retries < max_retries:
            try:
                # Call OpenAI API
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1024,  # Maximum length of response
                    temperature=0.7   # Control randomness (0-2, lower = more focused)
                )

                # Extract answer from response
                answer = response.choices[0].message.content

                return answer

            except RateLimitError as e:
                # Hit rate limit, wait and retry
                retries += 1
                if retries >= max_retries:
                    return "Sorry, I'm currently experiencing high demand. Please try again in a moment."

                # Exponential backoff: wait longer each time
                wait_time = 2 ** retries  # 2, 4, 8 seconds
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry {retries}/{max_retries}...")
                time.sleep(wait_time)

            except APIError as e:
                # Other API errors (network, service issues, etc.)
                error_msg = f"API Error: {str(e)}"
                print(error_msg)
                return f"Sorry, I encountered an error while generating the answer. Please try again. Error: {str(e)}"

            except Exception as e:
                # Unexpected errors
                error_msg = f"Unexpected error: {str(e)}"
                print(error_msg)
                return f"Sorry, an unexpected error occurred. Please try again."

        return "Sorry, I couldn't generate an answer after multiple attempts."

    def get_conversation_context(
        self,
        question: str,
        chat_history: List[Dict[str, str]],
        n_chunks: int = None
    ) -> Dict[str, Any]:
        """
        Enhanced query that considers conversation history.

        Args:
            question: Current question
            chat_history: Previous Q&A pairs
                         Format: [{'question': '...', 'answer': '...'}, ...]
            n_chunks: Number of chunks to retrieve

        Returns:
            Dict with answer and sources

        Learning Note - Conversation Context:
        -----------------------------------
        Sometimes questions reference previous conversation:
        User: "What's the refund policy?"
        Bot: "Refunds are available within 30 days..."
        User: "What documents do I need for that?"
                ^^^^ "that" refers to refunds

        Advanced RAG systems can:
        - Rewrite questions to be standalone
        - Track conversation state
        - Reference previous answers

        For beginners, we keep it simple: each question is independent.
        This method is here for future enhancement.
        """
        # For now, just process the question normally
        # Future enhancement: incorporate chat_history
        return self.query(question, n_chunks=n_chunks)

    def test_retrieval(
        self,
        question: str,
        n_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Test retrieval without calling Claude (useful for debugging).

        Args:
            question: Test question
            n_chunks: Number of chunks to retrieve

        Returns:
            List of retrieved chunks with scores

        This is useful for:
        - Testing if relevant chunks are being retrieved
        - Understanding retrieval quality
        - Debugging retrieval issues
        - Saving API costs during development

        Example:
            >>> engine = RAGEngine(vector_store)
            >>> chunks = engine.test_retrieval("What is Python?")
            >>> for chunk in chunks:
            ...     print(f"Distance: {chunk['distance']:.3f}")
            ...     print(f"Text: {chunk['text'][:100]}...")
            ...     print()
        """
        chunks = self.vector_store.query(
            query_text=question,
            n_results=n_chunks
        )

        # Add more readable similarity scores
        for chunk in chunks:
            # Convert distance to similarity (0-1, higher is better)
            # This is approximate
            chunk['similarity'] = 1 / (1 + chunk['distance'])

        return chunks


# ============================================================================
# Convenience functions
# ============================================================================

def quick_query(question: str, vector_store: ChromaDBStore = None) -> str:
    """
    Quick one-line query for simple use cases.

    Args:
        question: Question to ask
        vector_store: Vector store (optional, creates one if not provided)

    Returns:
        str: Answer text only

    This is a simplified interface for quick queries.

    Example:
        >>> from modules.vector_store import get_vector_store
        >>> store = get_vector_store()
        >>> answer = quick_query("What is the refund policy?", store)
        >>> print(answer)
    """
    if vector_store is None:
        from modules.vector_store import get_vector_store
        vector_store = get_vector_store()

    engine = RAGEngine(vector_store)
    result = engine.query(question, return_sources=False)

    return result['answer']
