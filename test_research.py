"""
Test script to debug research functionality
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from modules.vector_store import ChromaDBStore
from modules.research_agent import ResearchAgent
from config import settings

def test_research():
    """Run a minimal research test"""
    print("🧪 Testing Research Functionality\n")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing vector store...")
    vector_store = ChromaDBStore()
    vector_store.set_collection(settings.RESEARCH_COLLECTION_NAME)
    print("✓ Vector store ready")

    # Initialize research agent
    print("\n2. Initializing research agent...")
    research_agent = ResearchAgent(vector_store, search_depth="basic")
    print("✓ Research agent ready")

    # Run research
    print("\n3. Conducting research on 'quantum computing'...")
    result = research_agent.conduct_research(
        topic="quantum computing breakthroughs",
        num_queries=2  # Use fewer queries for faster testing
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print(f"- Report length: {len(result['report'])} chars")
    print(f"- Sources found: {len(result['sources'])}")
    print(f"- Chunks processed: {result['num_chunks']}")
    print(f"- Search quality: {result['search_quality']}")
    print("=" * 60)

    if result['sources']:
        print("\n✅ Research successful!")
        print(f"\nFirst 500 chars of report:\n{result['report'][:500]}...")
    else:
        print("\n❌ Research failed - no sources found")
        print(f"\nReport:\n{result['report']}")

if __name__ == "__main__":
    test_research()
