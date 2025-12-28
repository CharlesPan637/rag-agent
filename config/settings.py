"""
Configuration settings for the RAG Agent.

This module centralizes all configuration management using environment variables.
It follows the 12-factor app methodology for configuration.

Learning Note:
- Environment variables keep sensitive data (like API keys) out of code
- They make the app configurable without changing code
- This is a security best practice and makes deployment easier
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads the .env file and makes variables available via os.getenv()
load_dotenv()

# Project root directory
# __file__ is this file's path, parents[1] goes up two levels to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================================
# API Configuration
# ============================================================================

# OpenAI API key for GPT models
# IMPORTANT: Never hard-code API keys! Always use environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file based on .env.example and add your API key."
    )

# ============================================================================
# Model Configuration
# ============================================================================

# OpenAI GPT model to use for answer generation
# Options: gpt-4-turbo-preview (best quality), gpt-4 (balanced),
#          gpt-3.5-turbo (fastest/cheapest)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

# Embedding model for converting text to vectors
# all-MiniLM-L6-v2 is lightweight (80MB) and works well for general text
# Alternative: all-mpnet-base-v2 (higher quality but larger)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================================
# ChromaDB Configuration
# ============================================================================

# Directory where ChromaDB stores vector embeddings
# This ensures data persists between app restarts
CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    str(PROJECT_ROOT / "data" / "chroma_db")
)

# Collection name for storing all documents
# Using a single collection makes things simpler for beginners
CHROMA_COLLECTION_NAME = "documents"

# ============================================================================
# Document Processing Configuration
# ============================================================================

# Text chunking parameters
#
# Why chunking matters:
# - LLMs have limited context windows (can't process entire books at once)
# - Smaller chunks = more precise retrieval
# - Larger chunks = more context but less precision
# - 1000 characters is a good balance for most use cases
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

# Overlap between chunks in characters
#
# Why overlap matters:
# - Prevents important information from being split across chunk boundaries
# - 200 characters (20% overlap) maintains continuity
# - Helps retrieve complete context even if the answer spans chunk boundaries
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ============================================================================
# Upload Configuration
# ============================================================================

# Maximum file size for uploads (in megabytes)
# This prevents users from uploading extremely large files that could:
# - Take too long to process
# - Consume too much memory
# - Cost too much in API calls
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

# Supported file extensions
# These are the document types our parser can handle
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]

# ============================================================================
# RAG Configuration
# ============================================================================

# Number of chunks to retrieve for each query
#
# Why 5 chunks?
# - Too few chunks: might miss relevant information
# - Too many chunks:
#   * Costs more (more tokens sent to OpenAI GPT)
#   * Takes longer to process
#   * Can dilute relevance with less-related content
# - 5 is a good starting point, can be adjusted based on your use case
DEFAULT_RETRIEVAL_COUNT = 5

# Minimum similarity score for retrieved chunks (0-1 scale)
# Chunks below this threshold won't be included in context
# Lower = more lenient, Higher = stricter
# None = no filtering
MIN_SIMILARITY_SCORE = None  # Can be set to 0.3 for stricter filtering

# ============================================================================
# Tavily API Configuration
# ============================================================================

# Tavily API key for web search (graceful degradation if missing)
# Get your free API key from: https://tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not found. Web research features will be disabled.")
    print("Get your key at: https://tavily.com")

# ============================================================================
# Research Configuration
# ============================================================================

# Number of search queries to generate per research topic
#
# Learning Note - Why Multiple Queries?
# -------------------------------------
# A single query gives you limited perspective on a topic.
# Multiple diverse queries ensure comprehensive coverage:
# - Direct query: Uses the topic as-is
# - Specific applications: Focuses on use cases
# - Technical focus: Emphasizes implementation details
# - Recent developments: Captures latest information
#
# More queries = better coverage but higher API costs
# 4 queries is a good balance for most topics
RESEARCH_QUERIES_COUNT = int(os.getenv("RESEARCH_QUERIES_COUNT", "4"))

# Number of results per search query
# 5 results × 4 queries = 20 total results to process
RESULTS_PER_QUERY = int(os.getenv("RESULTS_PER_QUERY", "5"))

# Search depth: "basic" or "advanced"
#
# Learning Note - Search Depth Options:
# -------------------------------------
# basic: Faster (1-2 seconds per query), cheaper, good for most topics
# advanced: More thorough (3-4 seconds per query), better for complex/technical topics
SEARCH_DEPTH = os.getenv("SEARCH_DEPTH", "basic")

# Maximum total results to process per research session
# This acts as a safety limit to prevent excessive API costs
MAX_RESEARCH_RESULTS = int(os.getenv("MAX_RESEARCH_RESULTS", "20"))

# Collection name for research results (separate from documents)
#
# Learning Note - Why Separate Collections?
# ------------------------------------------
# We use different ChromaDB collections for documents vs web research because:
# - Different metadata schemas (filenames vs URLs)
# - Different content characteristics (structured docs vs web content)
# - Easier to clear web research without affecting uploaded documents
# - Allows different chunk sizes and processing strategies
RESEARCH_COLLECTION_NAME = "web_research"

# Web content chunking configuration
#
# Learning Note - Why Different Chunk Sizes for Web Content?
# -----------------------------------------------------------
# Web content is often less structured than documents:
# - May have navigation elements, ads, etc. (even after cleaning)
# - Sentences may be shorter and more punchy
# - Content density varies more
# - 800 characters with 150 overlap works well for web content
WEB_CHUNK_SIZE = int(os.getenv("WEB_CHUNK_SIZE", "800"))
WEB_CHUNK_OVERLAP = int(os.getenv("WEB_CHUNK_OVERLAP", "150"))

# ============================================================================
# Directory Setup
# ============================================================================

# Ensure required directories exist
def ensure_directories():
    """
    Create necessary directories if they don't exist.

    This is called at startup to make sure the app has places to:
    - Store uploaded files temporarily
    - Persist ChromaDB data
    """
    directories = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "uploads",
        CHROMA_PERSIST_DIRECTORY,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Configuration Summary (for debugging)
# ============================================================================

def print_config():
    """
    Print configuration summary (useful for debugging).
    WARNING: Never print API keys!
    """
    print("=" * 60)
    print("RAG Agent Configuration")
    print("=" * 60)
    print(f"OpenAI Model: {OPENAI_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Chunk Size: {CHUNK_SIZE} characters")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"ChromaDB Directory: {CHROMA_PERSIST_DIRECTORY}")
    print(f"Max Upload Size: {MAX_UPLOAD_SIZE_MB} MB")
    print(f"OpenAI API Key: {'✓ Configured' if OPENAI_API_KEY else '✗ Missing'}")
    print(f"Tavily API Key: {'✓ Configured' if TAVILY_API_KEY else '✗ Missing'}")
    print(f"Research Queries: {RESEARCH_QUERIES_COUNT}")
    print(f"Web Chunk Size: {WEB_CHUNK_SIZE} characters")
    print("=" * 60)


# Auto-create directories when this module is imported
ensure_directories()
