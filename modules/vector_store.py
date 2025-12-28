"""
Vector Store Module for RAG Agent using ChromaDB.

This module handles:
1. Storing document chunks and their embeddings
2. Semantic search (finding similar chunks)
3. Persistent storage (data survives app restarts)
4. Collection management

Learning Note - What is a Vector Database?
------------------------------------------
Traditional databases store data in tables with rows and columns.
Vector databases store embeddings and can quickly find similar vectors.

Analogy:
- Traditional DB: Phone book (exact lookups: "Find John Smith")
- Vector DB: "Find people similar to John Smith"

This enables semantic search:
- Query: "How do I reset my password?"
- Finds chunks about password reset, account recovery, login help
- Even if exact words don't match!

ChromaDB is perfect for learning because:
- Easy to use (no complex setup)
- Runs locally (no external services)
- Persistent (data saved to disk)
- Free and open source
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from modules.embeddings import get_embedder


class ChromaDBStore:
    """
    Interface for ChromaDB vector database.

    This class:
    1. Manages connections to ChromaDB
    2. Stores document chunks with embeddings
    3. Performs semantic search
    4. Handles persistent storage
    """

    def __init__(self, persist_directory: str = None):
        """
        Initialize ChromaDB client with persistent storage.

        Args:
            persist_directory: Directory to save data (persists across restarts)
                              If None, uses directory from settings

        Learning Note - Persistent vs. In-Memory:
        -----------------------------------------
        In-memory: Fast but data lost when app closes
        Persistent: Data saved to disk, survives restarts

        For a RAG app, you want persistent storage!
        Users shouldn't have to re-upload documents every time.
        """
        if persist_directory is None:
            persist_directory = settings.CHROMA_PERSIST_DIRECTORY

        self.persist_directory = persist_directory

        # Create persistent client
        # This saves all data to disk in the specified directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize embedding generator
        self.embedder = get_embedder()

        # Current active collection (can be switched)
        self.collection_name = None
        self.collection = None

        print(f"ChromaDB initialized at: {persist_directory}")
        print(f"Multi-collection support enabled")

    def _get_or_create_collection(self, collection_name: str):
        """
        Get existing collection or create new one if it doesn't exist.

        Args:
            collection_name: Name of the collection to get/create

        Returns:
            chromadb.Collection: The collection object

        Learning Note - Collections:
        Collections in ChromaDB are like tables in SQL databases.
        Each collection can store documents for a different purpose:
        - Bioinformatics: Biology and genomics documents
        - AI Agent: AI and machine learning documents
        - Miscellaneous: Everything else
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": f"Documents for {collection_name}"}
            )
            print(f"Created new collection: {collection_name}")

        return collection

    def set_collection(self, collection_name: str):
        """
        Switch to a different collection.

        Args:
            collection_name: Name of the collection to switch to

        This allows you to work with different document collections
        without creating multiple ChromaDBStore instances.
        """
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection(collection_name)
        print(f"Switched to collection: {collection_name}")

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            show_progress: Whether to show progress

        Returns:
            int: Number of chunks added

        Learning Note - The Storage Process:
        1. Take text chunk
        2. Generate embedding (convert to vector)
        3. Store both text and vector in database
        4. Store metadata for tracking

        Later, when searching:
        1. Convert search query to vector
        2. Find similar vectors in database
        3. Return the associated text chunks

        Example:
            >>> chunks = [
            ...     {
            ...         'text': 'The cat sat on the mat',
            ...         'metadata': {'filename': 'story.txt', 'chunk_id': 0}
            ...     }
            ... ]
            >>> count = store.add_documents(chunks)
            >>> print(f"Added {count} chunks")
        """
        if not chunks:
            return 0

        # Debug: Check what we received
        if chunks and not isinstance(chunks[0], dict):
            print(f"ERROR: Expected chunks to be list of dicts, got: {type(chunks[0])}")
            print(f"First chunk: {chunks[0][:100] if isinstance(chunks[0], str) else chunks[0]}")
            raise TypeError(f"Chunks must be list of dictionaries, got list of {type(chunks[0])}")

        # Extract texts for embedding generation
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings for all chunks (batch processing is faster)
        if show_progress:
            print(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = self.embedder.generate_embeddings_batch(
            texts,
            show_progress=show_progress
        )

        # Prepare data for ChromaDB
        # ChromaDB needs: IDs, embeddings, documents (texts), metadatas

        # Generate unique IDs for each chunk
        # Format: filename_chunkID_timestamp
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # Create unique ID
            metadata = chunk['metadata']
            chunk_id = f"{metadata['filename']}_{metadata['chunk_id']}_{metadata['upload_date']}"
            ids.append(chunk_id)

            # Store text
            documents.append(chunk['text'])

            # Store metadata (ChromaDB requires dict)
            metadatas.append(metadata)

        # Add to ChromaDB
        if show_progress:
            print(f"Storing {len(chunks)} chunks in vector database...")

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        if show_progress:
            print(f"Successfully stored {len(chunks)} chunks!")

        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query text.

        Args:
            query_text: The search query
            n_results: Number of results to return (default: from settings)

        Returns:
            List of chunk dictionaries with text, metadata, and similarity distance

        Learning Note - How Semantic Search Works:
        1. Convert query to embedding vector
        2. Find vectors in database closest to query vector
        3. Return the chunks associated with those vectors

        Distance metrics:
        - Smaller distance = more similar
        - ChromaDB uses squared L2 distance by default
        - 0 = identical, larger = more different

        Example:
            >>> results = store.query("How do I reset my password?", n_results=5)
            >>> for result in results:
            ...     print(f"File: {result['metadata']['filename']}")
            ...     print(f"Text: {result['text'][:100]}...")
            ...     print(f"Distance: {result['distance']:.3f}")
            ...     print()
        """
        if n_results is None:
            n_results = settings.DEFAULT_RETRIEVAL_COUNT

        # Generate embedding for query
        query_embedding = self.embedder.generate_embedding(query_text)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Format results into nice dictionaries
        formatted_results = []

        # ChromaDB returns results in a nested structure
        # Extract and flatten for easier use
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            formatted_results.append({
                'text': text,
                'metadata': metadata,
                'distance': distance  # Lower is better (more similar)
            })

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current active collection.

        Returns:
            Dict with collection statistics

        This is useful for:
        - Showing user how many documents are stored
        - Debugging
        - Monitoring storage usage

        Example:
            >>> stats = store.get_collection_stats()
            >>> print(f"Total chunks: {stats['count']}")
            >>> print(f"Collection: {stats['name']}")
        """
        if not self.collection:
            return {
                'name': 'None',
                'count': 0,
                'persist_directory': self.persist_directory
            }

        count = self.collection.count()

        return {
            'name': self.collection_name,
            'count': count,
            'persist_directory': self.persist_directory
        }

    def get_all_collections_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all collections in the database.

        Returns:
            List of dicts with stats for each collection including document count

        Example:
            >>> all_stats = store.get_all_collections_stats()
            >>> for stats in all_stats:
            ...     print(f"{stats['name']}: {stats['document_count']} documents")
        """
        all_collections = self.client.list_collections()
        stats_list = []

        for collection in all_collections:
            chunk_count = collection.count()

            # Count unique documents by getting unique filenames
            document_count = 0
            if chunk_count > 0:
                try:
                    # Get all metadata
                    data = collection.get()
                    metadatas = data.get('metadatas', [])

                    # Extract unique filenames
                    filenames = set()
                    for metadata in metadatas:
                        if 'filename' in metadata:
                            filenames.add(metadata['filename'])

                    document_count = len(filenames)
                except Exception as e:
                    print(f"Error counting documents: {e}")
                    document_count = 0

            stats_list.append({
                'name': collection.name,
                'chunk_count': chunk_count,
                'document_count': document_count,
            })

        return stats_list

    def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks from a specific file.

        Args:
            filename: Name of file to delete

        Returns:
            int: Number of chunks deleted

        Learning Note:
        This uses metadata filtering to find and delete all chunks
        from a specific document. Useful for:
        - Removing outdated documents
        - Managing storage space
        - Correcting upload mistakes

        Example:
            >>> count = store.delete_by_filename("old_report.pdf")
            >>> print(f"Deleted {count} chunks")
        """
        # Get all chunks with this filename
        results = self.collection.get(
            where={"filename": filename}
        )

        ids = results.get('ids', [])

        if ids:
            # Delete the chunks
            self.collection.delete(ids=ids)
            print(f"Deleted {len(ids)} chunks from {filename}")
            return len(ids)
        else:
            print(f"No chunks found for {filename}")
            return 0

    def clear_collection(self) -> bool:
        """
        Delete all data from the collection.

        Returns:
            bool: True if successful

        WARNING: This deletes ALL documents! Use with caution.

        This is useful for:
        - Starting fresh
        - Testing
        - Clearing out old data

        Example:
            >>> store.clear_collection()
            >>> print("All documents deleted")
        """
        # Delete the collection
        self.client.delete_collection(name=self.collection_name)

        # Recreate empty collection
        self.collection = self._get_or_create_collection()

        print(f"Collection '{self.collection_name}' cleared")
        return True

    def list_files(self) -> List[str]:
        """
        List all unique filenames stored in the collection.

        Returns:
            List of filenames

        This shows the user what documents are currently loaded.

        Example:
            >>> files = store.list_files()
            >>> print("Stored documents:")
            >>> for file in files:
            ...     print(f"  - {file}")
        """
        # Get all documents
        all_data = self.collection.get()

        # Extract unique filenames from metadata
        metadatas = all_data.get('metadatas', [])
        filenames = set()

        for metadata in metadatas:
            if 'filename' in metadata:
                filenames.add(metadata['filename'])

        return sorted(list(filenames))

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all unique documents in the current collection.

        Returns:
            List of document info dictionaries with filename and upload date

        Example:
            >>> docs = store.get_all_documents()
            >>> for doc in docs:
            ...     print(f"{doc['filename']} - {doc['upload_date']}")
        """
        if not self.collection:
            return []

        # Get all documents in the collection
        results = self.collection.get(
            include=['metadatas']
        )

        metadatas = results.get('metadatas', [])

        if not metadatas:
            return []

        # Extract unique filenames
        seen_files = set()
        documents = []

        for metadata in metadatas:
            filename = metadata.get('filename', 'Unknown')

            if filename not in seen_files:
                seen_files.add(filename)
                documents.append({
                    'filename': filename,
                    'upload_date': metadata.get('upload_date', 'Unknown')
                })

        # Sort by filename
        documents.sort(key=lambda x: x['filename'])
        return documents

    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific file in the collection.

        Args:
            filename: Name of file to get info for

        Returns:
            Dict with file information, or None if not found

        Example:
            >>> info = store.get_file_info("report.pdf")
            >>> print(f"Chunks: {info['chunk_count']}")
            >>> print(f"Uploaded: {info['upload_date']}")
        """
        # Get all chunks for this file
        results = self.collection.get(
            where={"filename": filename}
        )

        metadatas = results.get('metadatas', [])

        if not metadatas:
            return None

        # Extract info from first chunk (upload date, etc.)
        first_metadata = metadatas[0]

        return {
            'filename': filename,
            'chunk_count': len(metadatas),
            'file_type': first_metadata.get('file_type', 'unknown'),
            'upload_date': first_metadata.get('upload_date', 'unknown'),
            'total_chunks': first_metadata.get('total_chunks', len(metadatas))
        }


# ============================================================================
# Convenience functions
# ============================================================================

# Global instance (created on first use)
_global_store = None


def get_vector_store() -> ChromaDBStore:
    """
    Get or create the global vector store instance.

    This ensures we use the same ChromaDB client across the application.

    Returns:
        ChromaDBStore: Shared instance

    Usage:
        >>> store = get_vector_store()
        >>> results = store.query("search query")
    """
    global _global_store

    if _global_store is None:
        _global_store = ChromaDBStore()

    return _global_store
