"""
RAG Agent - Streamlit Web Application

This is the main entry point for the RAG Agent web application.
Run this file to start the application:
    streamlit run app.py

Learning Note - What is Streamlit?
----------------------------------
Streamlit is a Python framework for building web apps quickly.
Key features:
- Pure Python (no HTML/CSS/JavaScript needed)
- Automatic UI updates when data changes
- Built-in widgets (file uploaders, chat interfaces, etc.)
- Perfect for data science and ML applications

How Streamlit works:
1. Your script runs from top to bottom
2. User interacts with a widget
3. Entire script reruns with new widget values
4. UI updates automatically

This is why we use st.session_state to persist data between reruns!
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

# Import our modules
from config import settings
from modules.document_processor import TextChunker
from modules.vector_store import ChromaDBStore
from modules.rag_query import RAGEngine
from utils.helpers import (
    validate_file_type,
    get_file_size_mb,
    format_sources,
    sanitize_filename
)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Agent - Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS for better appearance
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 3px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 3px solid #17a2b8;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """
    Initialize all session state variables.

    Learning Note - Session State in Streamlit:
    ------------------------------------------
    Streamlit reruns your script on every interaction.
    Without session state, you'd lose all data on each rerun!

    Session state persists:
    - Between reruns (when user clicks a button)
    - For the duration of the user's session
    - Not between different users (each has their own state)

    Use it for:
    - Database connections
    - Loaded models
    - Chat history
    - Any data that should survive reruns
    """
    if 'initialized' not in st.session_state:
        # Mark as initialized
        st.session_state.initialized = True

        # Initialize vector store (expensive to create, so cache it)
        st.session_state.vector_store = ChromaDBStore()

        # Available collections
        st.session_state.collections = ['Bioinformatics', 'AI_Agent', 'Miscellaneous']

        # Set default collection
        st.session_state.active_collection = 'Bioinformatics'
        st.session_state.vector_store.set_collection('Bioinformatics')

        # Initialize RAG engine
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)

        # Chat history per collection: {collection_name: [chats]}
        st.session_state.chat_history = {col: [] for col in st.session_state.collections}

        # Uploaded files per collection: {collection_name: [files]}
        st.session_state.uploaded_files = {col: [] for col in st.session_state.collections}

        # Processing status
        st.session_state.processing = False


# ============================================================================
# Helper Functions
# ============================================================================

def process_uploaded_file(uploaded_file) -> dict:
    """
    Process a single uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        dict: Processing result with success status and message

    Learning Note - File Processing Pipeline:
    1. Validate file type and size
    2. Parse document to extract text
    3. Chunk text intelligently
    4. Generate embeddings
    5. Store in vector database

    This is the document ingestion pipeline!
    """
    try:
        # Get file info
        filename = sanitize_filename(uploaded_file.name)
        file_type = Path(filename).suffix.lower()
        file_size = get_file_size_mb(uploaded_file)

        # Validate file type
        if not validate_file_type(filename):
            return {
                'success': False,
                'message': f"Unsupported file type: {file_type}. Supported: {', '.join(settings.SUPPORTED_FILE_TYPES)}"
            }

        # Validate file size
        if file_size > settings.MAX_UPLOAD_SIZE_MB:
            return {
                'success': False,
                'message': f"File too large: {file_size:.1f} MB. Maximum: {settings.MAX_UPLOAD_SIZE_MB} MB"
            }

        # Process document: parse, chunk, add metadata
        chunks = TextChunker.process_document(
            uploaded_file,
            filename,
            file_type
        )

        # Store in vector database
        count = st.session_state.vector_store.add_documents(
            chunks,
            show_progress=False
        )

        return {
            'success': True,
            'message': f"Successfully processed {filename}: {count} chunks created",
            'filename': filename,
            'chunk_count': count
        }

    except Exception as e:
        return {
            'success': False,
            'message': f"Error processing {uploaded_file.name}: {str(e)}"
        }


def display_sources(sources: list, use_expanders: bool = True):
    """
    Display retrieved sources in an attractive format.

    Args:
        sources: List of retrieved chunks with metadata
        use_expanders: Whether to use expanders for each source (set False if already in expander)
    """
    if not sources:
        return

    st.markdown("### 📄 Retrieved Sources")

    for idx, source in enumerate(sources, start=1):
        metadata = source['metadata']
        filename = metadata.get('filename', 'Unknown')
        chunk_id = metadata.get('chunk_id', '?')
        distance = source.get('distance', 0)

        # Calculate similarity score (0-1, higher is better)
        similarity = 1 / (1 + distance)

        if use_expanders:
            with st.expander(f"Source {idx}: {filename} (Chunk {chunk_id}) - Similarity: {similarity:.2%}"):
                st.markdown(f"**File:** {filename}")
                st.markdown(f"**Chunk ID:** {chunk_id}")
                st.markdown(f"**Relevance Score:** {similarity:.2%}")
                st.markdown("**Content:**")
                st.text(source['text'])
        else:
            # Display without expander (for when already inside an expander)
            st.markdown(f"**Source {idx}: {filename} (Chunk {chunk_id})**")
            st.markdown(f"- Relevance Score: {similarity:.2%}")
            st.markdown(f"- Content Preview: {source['text'][:200]}...")
            st.markdown("---")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application function."""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📚 RAG Agent - Document Q&A System</h1>', unsafe_allow_html=True)

    st.markdown("""
    Welcome to the RAG Agent! Upload your documents (PDF, Word, or text files) and ask questions about them.
    The system will find relevant information and provide answers with source citations.
    """)

    # ========================================================================
    # Sidebar - Document Upload and Management
    # ========================================================================

    with st.sidebar:
        st.header("📁 Document Management")

        # Collection selector
        st.subheader("🗂️ Select Collection")
        selected_collection = st.selectbox(
            "Choose document category:",
            options=st.session_state.collections,
            index=st.session_state.collections.index(st.session_state.active_collection),
            help="Organize your documents by category"
        )

        # Switch collection if changed
        if selected_collection != st.session_state.active_collection:
            st.session_state.active_collection = selected_collection
            st.session_state.vector_store.set_collection(selected_collection)
            st.success(f"Switched to {selected_collection} collection!")
            st.rerun()

        st.markdown("---")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, Word documents, or text files"
        )

        # Process button
        if uploaded_files and st.button("🚀 Process Documents", type="primary"):
            st.session_state.processing = True

            with st.spinner("Processing documents..."):
                results = []

                # Process each file
                for uploaded_file in uploaded_files:
                    result = process_uploaded_file(uploaded_file)
                    results.append(result)

                # Display results
                success_count = sum(1 for r in results if r['success'])
                fail_count = len(results) - success_count

                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count} document(s)!")

                    # Add to uploaded files list for current collection
                    active_col = st.session_state.active_collection
                    for result in results:
                        if result['success']:
                            st.session_state.uploaded_files[active_col].append({
                                'filename': result['filename'],
                                'chunk_count': result['chunk_count'],
                                'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M")
                            })

                if fail_count > 0:
                    st.error(f"❌ Failed to process {fail_count} document(s)")

                # Show individual results
                for result in results:
                    if result['success']:
                        st.info(f"✓ {result['message']}")
                    else:
                        st.error(f"✗ {result['message']}")

            st.session_state.processing = False

        # Display collection statistics
        st.markdown("---")
        st.subheader("📊 All Collections")

        # Show stats for all collections
        all_stats = st.session_state.vector_store.get_all_collections_stats()
        for col_stats in all_stats:
            col_name = col_stats['name']
            doc_count = col_stats['document_count']
            chunk_count = col_stats['chunk_count']

            # Highlight active collection
            if col_name == st.session_state.active_collection:
                st.metric(
                    f"✓ {col_name}",
                    f"{doc_count} docs",
                    delta=f"{chunk_count} chunks",
                    help="Currently active collection"
                )
            else:
                st.metric(
                    col_name,
                    f"{doc_count} docs",
                    delta=f"{chunk_count} chunks"
                )

        # List uploaded files for current collection
        st.markdown("---")
        active_col = st.session_state.active_collection
        uploaded_in_collection = st.session_state.uploaded_files.get(active_col, [])

        if uploaded_in_collection:
            st.subheader(f"📄 Documents in {active_col}")
            for file_info in uploaded_in_collection:
                with st.expander(f"📄 {file_info['filename']}"):
                    st.write(f"**Chunks:** {file_info['chunk_count']}")
                    st.write(f"**Uploaded:** {file_info['upload_date']}")

        # Clear current collection button
        stats = st.session_state.vector_store.get_collection_stats()
        if stats['count'] > 0:
            st.markdown("---")
            if st.button(f"🗑️ Clear {active_col} Collection", help=f"Delete all documents from {active_col}"):
                if st.session_state.vector_store.clear_collection():
                    st.session_state.uploaded_files[active_col] = []
                    st.session_state.chat_history[active_col] = []
                    st.success(f"{active_col} collection cleared!")
                    st.rerun()

    # ========================================================================
    # Main Area - Chat Interface
    # ========================================================================

    # Check if documents are loaded
    stats = st.session_state.vector_store.get_collection_stats()

    if stats['count'] == 0:
        # No documents uploaded yet
        st.info("""
        👈 **Get Started:**
        1. Upload one or more documents using the sidebar
        2. Click "Process Documents"
        3. Ask questions about your documents!

        **Supported formats:** PDF, Microsoft Word (.docx), Plain Text (.txt)
        """)
    else:
        # Documents are loaded, show chat interface
        st.markdown("### 💬 Ask Questions")

        # Display chat history for current collection
        active_col = st.session_state.active_collection
        collection_chat_history = st.session_state.chat_history.get(active_col, [])

        if collection_chat_history:
            for idx, chat in enumerate(collection_chat_history):
                # Question
                with st.chat_message("user"):
                    st.write(chat['question'])

                # Answer
                with st.chat_message("assistant"):
                    st.markdown(chat['answer'])

                    # Show sources
                    if chat.get('sources'):
                        with st.expander("📚 View Sources"):
                            display_sources(chat['sources'], use_expanders=False)

        # Chat input
        question = st.chat_input("Ask a question about your documents...")

        if question:
            # Display user question
            with st.chat_message("user"):
                st.write(question)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Query the RAG engine
                        result = st.session_state.rag_engine.query(
                            question,
                            n_chunks=settings.DEFAULT_RETRIEVAL_COUNT,
                            return_sources=True
                        )

                        # Display answer
                        st.markdown(result['answer'])

                        # Display sources
                        if result.get('sources'):
                            with st.expander("📚 View Sources"):
                                display_sources(result['sources'], use_expanders=False)

                        # Add to chat history for current collection
                        active_col = st.session_state.active_collection
                        st.session_state.chat_history[active_col].append({
                            'question': question,
                            'answer': result['answer'],
                            'sources': result.get('sources', [])
                        })

                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        st.error("Please make sure your OpenAI API key is configured correctly in the .env file.")

    # ========================================================================
    # Footer
    # ========================================================================

    st.markdown("---")
    with st.expander("ℹ️ About this application"):
        st.markdown("""
        ### How it works

        This is a **RAG (Retrieval Augmented Generation)** system that:

        1. **Parses your documents** into smaller chunks
        2. **Converts text to embeddings** (numerical vectors that capture meaning)
        3. **Stores embeddings** in a vector database (ChromaDB)
        4. **Retrieves relevant chunks** when you ask a question
        5. **Generates answers** using Claude AI based on retrieved context

        ### Key Features

        - 📄 Support for PDF, Word, and text files
        - 🔍 Semantic search (finds relevant content, not just keywords)
        - 💾 Persistent storage (documents saved between sessions)
        - 📚 Source citations (see where answers come from)
        - 🤖 Powered by OpenAI's GPT models

        ### Technologies Used

        - **Streamlit**: Web interface
        - **ChromaDB**: Vector database
        - **sentence-transformers**: Text embeddings
        - **OpenAI GPT**: Answer generation
        - **pypdf & python-docx**: Document parsing

        ### Learning Resources

        - [Streamlit Documentation](https://docs.streamlit.io)
        - [ChromaDB Documentation](https://docs.trychroma.com)
        - [OpenAI Documentation](https://platform.openai.com/docs)
        - [RAG Explained](https://platform.openai.com/docs/guides/retrieval-augmented-generation)
        """)

    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Built with Streamlit + ChromaDB + OpenAI GPT | '
        'For educational purposes</p>',
        unsafe_allow_html=True
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # This block runs when you execute: streamlit run app.py
    main()
