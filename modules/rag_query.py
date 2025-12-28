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
