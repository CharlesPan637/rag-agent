"""
Helper utility functions for the RAG Agent.

This module provides reusable utility functions for:
- File validation
- Text formatting
- Source citation formatting
- Token estimation

Learning Note:
- Utility functions help avoid code duplication
- They make the codebase more maintainable
- Small, focused functions are easier to test and understand
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from config import settings


def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """
    Check if a file has an allowed extension.

    Args:
        filename: Name of the file to check
        allowed_types: List of allowed extensions (e.g., ['.pdf', '.txt'])
                      If None, uses SUPPORTED_FILE_TYPES from settings

    Returns:
        bool: True if file type is allowed, False otherwise

    Example:
        >>> validate_file_type("document.pdf")
        True
        >>> validate_file_type("image.jpg")
        False

    Learning Note:
        File extension checking is a simple but important security measure.
        Always validate user uploads before processing them.
    """
    if allowed_types is None:
        allowed_types = settings.SUPPORTED_FILE_TYPES

    # Get file extension (including the dot)
    file_extension = Path(filename).suffix.lower()

    return file_extension in allowed_types


def get_file_size_mb(file) -> float:
    """
    Calculate file size in megabytes.

    Args:
        file: File object (typically from Streamlit file_uploader)

    Returns:
        float: File size in megabytes

    Example:
        >>> size = get_file_size_mb(uploaded_file)
        >>> print(f"File size: {size:.2f} MB")

    Learning Note:
        Checking file sizes helps prevent:
        - Out of memory errors
        - Extremely long processing times
        - Excessive API costs
    """
    # Seek to end of file to get size
    file.seek(0, os.SEEK_END)
    size_bytes = file.tell()

    # Reset file pointer to beginning for later reading
    file.seek(0)

    # Convert bytes to megabytes
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def format_sources(chunks_with_metadata: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks as citation sources for display.

    Args:
        chunks_with_metadata: List of chunks with their metadata
                             Each chunk should have:
                             - 'text': The chunk content
                             - 'metadata': Dict with 'filename', 'chunk_id', etc.

    Returns:
        str: Formatted string showing all sources

    Example:
        >>> sources = format_sources(retrieved_chunks)
        >>> print(sources)
        Sources:
        - document.pdf (Chunk 3)
        - report.docx (Chunk 1, 7)

    Learning Note:
        Proper source attribution is crucial for:
        - Transparency (user knows where info came from)
        - Credibility (answers are grounded in documents)
        - Debugging (helps identify if wrong documents are retrieved)
    """
    if not chunks_with_metadata:
        return "No sources found."

    # Group chunks by filename
    sources_by_file = {}

    for chunk in chunks_with_metadata:
        metadata = chunk.get('metadata', {})
        filename = metadata.get('filename', 'Unknown')
        chunk_id = metadata.get('chunk_id', '?')

        if filename not in sources_by_file:
            sources_by_file[filename] = []

        sources_by_file[filename].append(chunk_id)

    # Format as readable list
    source_lines = ["Sources:"]
    for filename, chunk_ids in sources_by_file.items():
        # Sort chunk IDs and format
        chunk_ids_str = ", ".join(map(str, sorted(chunk_ids)))
        source_lines.append(f"- {filename} (Chunk {chunk_ids_str})")

    return "\n".join(source_lines)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding suffix if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        suffix: String to append if truncated (default: "...")

    Returns:
        str: Truncated text with suffix, or original if under max_length

    Example:
        >>> long_text = "This is a very long document..." * 100
        >>> short = truncate_text(long_text, 50)
        >>> print(short)
        This is a very long document...This is a very...

    Learning Note:
        Truncation is useful for:
        - Displaying previews in UI
        - Limiting log message length
        - Creating summaries
    """
    if len(text) <= max_length:
        return text

    # Truncate and add suffix
    return text[:max_length - len(suffix)] + suffix


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for text.

    Args:
        text: Text to estimate tokens for

    Returns:
        int: Estimated number of tokens

    Learning Note:
        This is a rough approximation. Actual token count varies by tokenizer.
        Rule of thumb: ~4 characters per token for English text.

        Why this matters:
        - API costs are based on token count
        - Context windows have token limits
        - Helps predict API costs before making calls

    Example:
        >>> text = "Hello world"
        >>> tokens = estimate_tokens(text)
        >>> print(f"Estimated tokens: {tokens}")
        Estimated tokens: 2
    """
    # Rough approximation: ~4 characters per token
    # This is a simplified estimate; real tokenizers are more complex
    return len(text) // 4


def format_file_size(size_bytes: int) -> str:
    """
    Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size (e.g., "1.5 MB", "345 KB")

    Example:
        >>> print(format_file_size(1536000))
        1.46 MB
        >>> print(format_file_size(2048))
        2.00 KB
    """
    # Define size units
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} TB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove potentially dangerous characters.

    Args:
        filename: Original filename

    Returns:
        str: Sanitized filename safe for file system

    Learning Note:
        File name sanitization prevents:
        - Path traversal attacks (e.g., "../../etc/passwd")
        - Invalid characters that cause errors
        - Command injection through filenames

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("my document!@#.pdf")
        'my_document___.pdf'
    """
    # Get just the filename (no path components)
    filename = os.path.basename(filename)

    # Replace potentially dangerous or invalid characters
    # Keep only alphanumeric, dots, dashes, and underscores
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '.-_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')

    sanitized = ''.join(safe_chars)

    # Ensure filename isn't empty
    if not sanitized or sanitized == '.':
        sanitized = 'unnamed_file'

    return sanitized


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into smaller chunks.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of lists, each containing up to chunk_size items

    Example:
        >>> numbers = [1, 2, 3, 4, 5, 6, 7]
        >>> chunks = chunk_list(numbers, 3)
        >>> print(chunks)
        [[1, 2, 3], [4, 5, 6], [7]]

    Learning Note:
        Batch processing is important for:
        - Efficiency (process multiple items at once)
        - API rate limits (some APIs limit batch sizes)
        - Memory management (avoid loading everything at once)
    """
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks
