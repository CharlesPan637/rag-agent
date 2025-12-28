"""
Embeddings Module for RAG Agent.

This module handles converting text into vector embeddings using
sentence-transformers. Embeddings are numerical representations of text
that capture semantic meaning.

Learning Note - What are embeddings?
------------------------------------
Imagine you need to organize thousands of books in a library. Instead of
arranging them alphabetically (which doesn't group similar topics), you
could place similar books near each other based on their content.

Embeddings do this for text:
- Convert text into a list of numbers (a vector)
- Similar texts have similar vectors
- Can measure "distance" between texts mathematically
- Enables semantic search (meaning-based, not just keyword matching)

Example:
"The cat sat on the mat" -> [0.2, -0.5, 0.8, ...]
"A feline rested on the rug" -> [0.19, -0.48, 0.82, ...]
(these vectors would be close together because meanings are similar)
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

from config import settings


class EmbeddingGenerator:
    """
    Generates vector embeddings for text using sentence-transformers.

    This class:
    1. Loads a pre-trained embedding model
    2. Converts text into numerical vectors
    3. Supports batch processing for efficiency
    4. Uses lazy loading (only loads model when first needed)

    Learning Note - Why sentence-transformers?
    -----------------------------------------
    - Free and open source
    - Runs locally (no API calls needed)
    - Fast inference
    - High quality embeddings
    - No usage limits or costs
    - Great for learning and prototyping
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use
                       If None, uses the model from settings

        Learning Note - Lazy Loading:
        We don't actually load the model here. Instead, we wait until
        the first time generate_embedding() is called. This is because:
        - Model loading takes time (5-10 seconds)
        - Model uses memory (~100MB)
        - Might not be needed if user only wants to query existing data
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None  # Will be loaded on first use

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load the embedding model.

        This property ensures the model is only loaded once, when first needed.
        Subsequent accesses return the cached model.

        Returns:
            SentenceTransformer: The loaded model

        Learning Note - The @property decorator:
        This makes model() look like an attribute but actually runs code.
        It's a Python pattern for lazy loading and computed properties.
        """
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            print("This may take a few seconds on first run...")

            # Download and load the model
            # On first run, this downloads ~80MB
            # On subsequent runs, it loads from cache
            self._model = SentenceTransformer(self.model_name)

            print("Model loaded successfully!")

        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single piece of text.

        Args:
            text: Text to convert to embedding

        Returns:
            List of floats representing the embedding vector

        Learning Note - Embedding dimensions:
        The all-MiniLM-L6-v2 model creates 384-dimensional vectors.
        This means each text is represented by 384 numbers.

        Why 384?
        - More dimensions = more nuanced meaning
        - Fewer dimensions = faster, less memory
        - 384 is a good balance for general use

        Example:
            >>> embedder = EmbeddingGenerator()
            >>> vector = embedder.generate_embedding("Hello world")
            >>> print(len(vector))
            384
            >>> print(vector[:5])  # First 5 dimensions
            [0.123, -0.456, 0.789, ...]
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        # Generate embedding
        # The model converts text to a vector
        embedding = self.model.encode(text, show_progress_bar=False)

        # Convert numpy array to Python list
        return embedding.tolist()

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (one per input text)

        Learning Note - Why batch processing?
        -------------------------------------
        Processing texts one-by-one:
            for text in texts:
                embed(text)  # Slow! Many separate operations

        Batch processing:
            embed_all(texts)  # Fast! GPU can process many at once

        Benefits:
        - Much faster (can be 10-50x faster)
        - More efficient use of CPU/GPU
        - Especially important for large documents with many chunks

        Example:
            >>> embedder = EmbeddingGenerator()
            >>> texts = ["First doc", "Second doc", "Third doc"]
            >>> vectors = embedder.generate_embeddings_batch(texts)
            >>> print(len(vectors))
            3
            >>> print(len(vectors[0]))
            384
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            raise ValueError("No valid texts to embed")

        # Generate embeddings in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        # Convert numpy array to list of lists
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            int: Number of dimensions in each embedding vector

        Learning Note:
        Different models produce different dimensional vectors:
        - all-MiniLM-L6-v2: 384 dimensions
        - all-mpnet-base-v2: 768 dimensions
        - OpenAI ada-002: 1536 dimensions

        This method is useful for:
        - Validating embeddings
        - Setting up vector databases
        - Debugging dimension mismatches
        """
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0 to 1, higher = more similar)

        Learning Note - Cosine Similarity:
        -----------------------------------
        This measures the angle between two vectors.
        - 1.0 = identical meaning
        - 0.0 = completely unrelated
        - Works regardless of text length

        Example:
            >>> embedder = EmbeddingGenerator()
            >>> sim = embedder.compute_similarity(
            ...     "The cat sat on the mat",
            ...     "A feline rested on the rug"
            ... )
            >>> print(f"Similarity: {sim:.2f}")
            Similarity: 0.78

        This is useful for:
        - Testing retrieval quality
        - Understanding what counts as "similar"
        - Debugging why certain chunks are retrieved
        """
        # Generate embeddings
        emb1 = np.array(self.generate_embedding(text1))
        emb2 = np.array(self.generate_embedding(text2))

        # Compute cosine similarity
        # Formula: (A · B) / (||A|| × ||B||)
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        similarity = dot_product / (norm1 * norm2)

        return float(similarity)


# ============================================================================
# Convenience functions for quick usage
# ============================================================================

# Global instance (created on first use)
_global_embedder = None


def get_embedder() -> EmbeddingGenerator:
    """
    Get or create the global embedding generator instance.

    This ensures we only load the model once across the entire application.

    Returns:
        EmbeddingGenerator: Shared instance

    Learning Note - Singleton Pattern:
    This is a design pattern that ensures only one instance of a class exists.
    Useful for expensive resources like ML models that should be shared.

    Usage:
        >>> embedder = get_embedder()
        >>> vector = embedder.generate_embedding("Hello")
    """
    global _global_embedder

    if _global_embedder is None:
        _global_embedder = EmbeddingGenerator()

    return _global_embedder
