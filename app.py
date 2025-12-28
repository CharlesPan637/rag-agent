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
import re
from io import BytesIO
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

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

# Import research modules
from modules.web_searcher import TavilySearcher
from modules.web_content_processor import WebContentChunker
from modules.research_agent import ResearchAgent


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Agent - Q&A & Research",
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

        # Available collections (fixed list - only these three)
        st.session_state.collections = ['Bioinformatics', 'AI-Agent', 'Miscellaneous']

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

        # App mode selection (Document Q&A or Web Research)
        st.session_state.app_mode = "📄 Document Q&A"

        # Research mode variables
        st.session_state.research_history = []
        st.session_state.current_research = None


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
        distance = source.get('distance', 0)

        # Calculate similarity score (0-1, higher is better)
        similarity = 1 / (1 + distance)

        if use_expanders:
            with st.expander(f"Source {idx}: {filename} - Similarity: {similarity:.2%}"):
                st.markdown(f"**File:** {filename}")
                st.markdown(f"**Relevance Score:** {similarity:.2%}")
                st.markdown("**Content:**")
                st.text(source['text'])
        else:
            # Display without expander (for when already inside an expander)
            st.markdown(f"**Source {idx}: {filename}**")
            st.markdown(f"- Relevance Score: {similarity:.2%}")
            st.markdown(f"- Content Preview: {source['text'][:200]}...")
            st.markdown("---")


def create_research_docx(research: dict) -> BytesIO:
    """
    Create a DOCX file from research report with clickable hyperlinks.

    Args:
        research: Research record dictionary with report, sources, metadata

    Returns:
        BytesIO object containing the DOCX file

    Learning Note - Why DOCX for Reports?
    ------------------------------------
    DOCX files are better than plain text because they:
    - Support clickable hyperlinks (users can verify sources)
    - Preserve formatting (headings, bold, lists)
    - Are widely compatible (Word, Google Docs, etc.)
    - Look professional
    - Can be easily edited and shared
    """
    doc = Document()

    # Add title
    title = doc.add_heading(f"Research Report: {research['topic']}", 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Add metadata section
    doc.add_paragraph()
    metadata_para = doc.add_paragraph()
    metadata_para.add_run("Generated: ").bold = True
    metadata_para.add_run(research['timestamp'][:19])
    metadata_para.add_run(" | ")
    metadata_para.add_run("Sources: ").bold = True
    metadata_para.add_run(str(research['num_sources']))
    metadata_para.add_run(" | ")
    metadata_para.add_run("Search Quality: ").bold = True
    metadata_para.add_run(f"{research.get('search_quality', 0.0):.2f}")

    doc.add_paragraph("_" * 80)
    doc.add_paragraph()

    # Parse and add report content with hyperlinks
    report_text = research['report']

    # Split report into lines and process
    lines = report_text.split('\n')

    for line in lines:
        line = line.strip()

        if not line:
            doc.add_paragraph()
            continue

        # Check for markdown headings
        if line.startswith('###'):
            doc.add_heading(line.replace('###', '').strip(), level=3)
        elif line.startswith('##'):
            doc.add_heading(line.replace('##', '').strip(), level=2)
        elif line.startswith('#'):
            doc.add_heading(line.replace('#', '').strip(), level=1)
        else:
            # Regular paragraph - parse for hyperlinks
            para = doc.add_paragraph()
            add_hyperlinks_to_paragraph(para, line)

    # Add sources section
    doc.add_paragraph()
    doc.add_paragraph("_" * 80)
    doc.add_heading("Sources", level=1)

    for i, source in enumerate(research['sources'], 1):
        para = doc.add_paragraph(style='List Number')
        para.add_run(f"{source['title']}").bold = True
        para.add_run("\n   ")

        # Add clickable URL
        add_hyperlink(para, source['url'], source['url'])

        para.add_run(f"\n   Relevance: {source.get('relevance_score', 'N/A')}")
        para.add_run(f"\n   Retrieved: {source.get('retrieved_date', 'N/A')[:10]}")

    # Add footer
    doc.add_paragraph()
    doc.add_paragraph("_" * 80)
    footer = doc.add_paragraph("Generated by RAG Research Agent")
    footer.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    # Save to BytesIO
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio


def add_hyperlinks_to_paragraph(paragraph, text):
    """
    Parse text for markdown hyperlinks and add them to paragraph.

    Handles patterns like: [Source 1: Title](URL)
    """
    # Pattern to match markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'

    last_end = 0
    for match in re.finditer(link_pattern, text):
        # Add text before the link
        if match.start() > last_end:
            paragraph.add_run(text[last_end:match.start()])

        # Add the hyperlink
        link_text = match.group(1)
        url = match.group(2)
        add_hyperlink(paragraph, url, link_text)

        last_end = match.end()

    # Add remaining text
    if last_end < len(text):
        paragraph.add_run(text[last_end:])


def add_hyperlink(paragraph, url, text):
    """
    Add a clickable hyperlink to a paragraph.

    This is a helper function because python-docx doesn't have a simple
    add_hyperlink method - we need to work with the underlying XML.
    """
    from docx.oxml.shared import OxmlElement
    from docx.oxml.ns import qn

    # Get access to the paragraph's part (for relationship management)
    part = paragraph.part
    r_id = part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)

    # Create the hyperlink element
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    # Create a new run element for the hyperlink text
    new_run = OxmlElement('w:r')

    # Create run properties for styling (blue, underlined)
    rPr = OxmlElement('w:rPr')

    # Add blue color
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0000FF')
    rPr.append(color)

    # Add underline
    u = OxmlElement('w:u')
    u.set(qn('w:val'), 'single')
    rPr.append(u)

    new_run.append(rPr)

    # Add the text
    text_elem = OxmlElement('w:t')
    text_elem.text = text
    new_run.append(text_elem)

    # Add the run to the hyperlink
    hyperlink.append(new_run)

    # Add the hyperlink to the paragraph
    paragraph._p.append(hyperlink)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application function."""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📚 RAG Agent - Q&A & Research</h1>', unsafe_allow_html=True)

    # Mode selector
    st.markdown("### Select Mode")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("📄 Document Q&A", use_container_width=True, type="primary" if st.session_state.app_mode == "📄 Document Q&A" else "secondary"):
            st.session_state.app_mode = "📄 Document Q&A"
            st.rerun()
    with mode_col2:
        if st.button("🌐 Web Research", use_container_width=True, type="primary" if st.session_state.app_mode == "🌐 Web Research" else "secondary"):
            st.session_state.app_mode = "🌐 Web Research"
            st.rerun()

    st.markdown("---")

    # Conditional rendering based on mode
    if st.session_state.app_mode == "📄 Document Q&A":
        render_document_qa_mode()
    else:
        render_research_mode()


def render_document_qa_mode():
    """Render the Document Q&A interface."""

    st.markdown("""
    Upload your documents (PDF, Word, or text files) and ask questions about them.
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

        # Show stats for only the allowed collections
        all_stats = st.session_state.vector_store.get_all_collections_stats()
        for col_stats in all_stats:
            col_name = col_stats['name']

            # Only show stats for the three allowed collections
            if col_name not in st.session_state.collections:
                continue
            doc_count = col_stats['document_count']

            # Get actual documents from ChromaDB for this collection
            # Temporarily switch to the collection to get its documents
            current_active = st.session_state.vector_store.collection_name
            st.session_state.vector_store.set_collection(col_name)
            docs_in_collection = st.session_state.vector_store.get_all_documents()
            # Switch back to the active collection
            st.session_state.vector_store.set_collection(current_active)

            # Create expander with collection name and document count
            if col_name == st.session_state.active_collection:
                expander_label = f"✓ {col_name} ({doc_count} docs) - Active"
            else:
                expander_label = f"{col_name} ({doc_count} docs)"

            with st.expander(expander_label, expanded=False):
                if docs_in_collection:
                    st.markdown("**Documents:**")
                    for i, doc in enumerate(docs_in_collection, 1):
                        st.markdown(f"{i}. 📄 {doc['filename']}")
                        st.caption(f"   Uploaded: {doc['upload_date']}")
                else:
                    st.info("No documents in this collection yet.")

        # List uploaded files for current collection
        st.markdown("---")
        active_col = st.session_state.active_collection
        uploaded_in_collection = st.session_state.uploaded_files.get(active_col, [])

        if uploaded_in_collection:
            st.subheader(f"📄 Documents in {active_col}")
            for file_info in uploaded_in_collection:
                with st.expander(f"📄 {file_info['filename']}"):
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


def render_research_mode():
    """
    Render the Web Research interface.

    Learning Note - Research Interface Design:
    -----------------------------------------
    Research mode is different from Q&A:
    - Takes longer (20-30 seconds vs 2-3 seconds)
    - Needs progress indicators
    - Generates comprehensive reports vs quick answers
    - Requires different UI patterns
    """
    st.markdown("""
    Enter any topic and the AI will research it for you! The system will search multiple web sources,
    analyze findings, and generate a comprehensive research report with citations.
    """)

    # Check if Tavily API key is configured
    if not settings.TAVILY_API_KEY:
        st.error("❌ **Tavily API key not configured**")
        st.info("""
        To use Web Research mode:
        1. Get your free API key from [tavily.com](https://tavily.com)
        2. Add it to your `.env` file: `TAVILY_API_KEY=your_key_here`
        3. Restart the application
        """)
        return

    # Sidebar - Research History
    with st.sidebar:
        st.header("📊 Research History")

        if st.session_state.research_history:
            st.markdown(f"**{len(st.session_state.research_history)} past research(es)**")

            for i, research in enumerate(st.session_state.research_history[:5]):
                with st.expander(f"📌 {research['topic'][:30]}..."):
                    if 'collection_title' in research:
                        st.caption(f"📁 {research['collection_title']}")
                    st.caption(f"Date: {research['timestamp'][:10]}")
                    st.caption(f"Sources: {research['num_sources']}")
                    if st.button(f"View", key=f"view_research_{i}"):
                        st.session_state.current_research = research
                        st.rerun()

            if len(st.session_state.research_history) > 5:
                st.caption(f"...and {len(st.session_state.research_history) - 5} more")

            st.markdown("---")
            if st.button("🗑️ Clear History"):
                st.session_state.research_history = []
                st.session_state.current_research = None
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("No research history yet. Start your first research!")

    # Main area - Research interface
    st.markdown("### 🔍 Research a Topic")

    # Research topic input
    topic = st.text_area(
        "Research Topic:",
        placeholder="e.g., 'Latest developments in quantum computing' or 'AI applications in healthcare 2024'",
        height=100,
        help="Enter any topic you'd like to research. The agent will search multiple sources and synthesize findings."
    )

    # Collection name input
    collection_name = st.text_input(
        "Collection Name (Optional):",
        placeholder="e.g., 'Quantum Computing Research' or 'Healthcare AI Study'",
        help="Give this research collection a name to organize it separately. Leave empty to auto-generate from topic."
    )

    # Research configuration
    col1, col2 = st.columns(2)
    with col1:
        search_depth = st.selectbox(
            "Search Depth:",
            options=["basic", "advanced"],
            help="Basic: Faster and cheaper (1-2s per query). Advanced: More thorough (3-4s per query)."
        )
    with col2:
        num_queries = st.slider(
            "Number of Search Queries:",
            min_value=3,
            max_value=5,
            value=4,
            help="More queries = better coverage but higher cost. Recommended: 4"
        )

    # Start research button
    if st.button("🔍 Start Research", type="primary", disabled=not topic.strip()):
        with st.spinner("Conducting research..."):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(msg, pct):
                    """Progress callback for research agent."""
                    status_text.text(msg)
                    progress_bar.progress(pct / 100)

                # Initialize research components
                status_text.text("Initializing research agent...")
                progress_bar.progress(0.1)

                # Generate collection name if not provided
                if not collection_name or not collection_name.strip():
                    # Auto-generate from topic (sanitize for use as collection name)
                    import re
                    clean_topic = re.sub(r'[^a-zA-Z0-9\s]', '', topic)[:50]
                    clean_topic = re.sub(r'\s+', '_', clean_topic.strip())
                    final_collection_name = f"research_{clean_topic.lower()}"
                else:
                    # Use provided name (sanitize it)
                    import re
                    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', collection_name)[:50]
                    final_collection_name = re.sub(r'\s+', '_', clean_name.strip().lower())

                # Create vector store and set it to the named collection
                research_vector_store = ChromaDBStore()
                research_vector_store.set_collection(final_collection_name)
                research_agent = ResearchAgent(research_vector_store, search_depth=search_depth)

                # Conduct research
                result = research_agent.conduct_research(
                    topic=topic,
                    num_queries=num_queries,
                    progress_callback=update_progress
                )

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Store in session state
                research_record = {
                    'topic': topic,
                    'collection_name': final_collection_name,
                    'collection_title': collection_name if collection_name.strip() else topic[:50],
                    'report': result['report'],
                    'sources': result['sources'],
                    'timestamp': datetime.now().isoformat(),
                    'num_sources': len(result['sources']),
                    'num_chunks': result['num_chunks'],
                    'search_quality': result['search_quality']
                }

                st.session_state.current_research = research_record
                st.session_state.research_history.insert(0, research_record)

                st.success(f"✅ Research complete! Synthesized findings from {len(result['sources'])} sources.")

                # Trigger rerun to display the report
                st.rerun()

            except Exception as e:
                st.error(f"❌ Research failed: {str(e)}")
                st.error("Please check your API keys and try again.")

    # Display current or most recent research
    if st.session_state.current_research:
        research = st.session_state.current_research

        st.markdown("---")
        st.markdown(f"## Research Report: {research['topic']}")

        # Collection info
        if 'collection_title' in research:
            st.info(f"📁 Collection: **{research['collection_title']}** (`{research.get('collection_name', 'N/A')}`)")

        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sources", research['num_sources'])
        with col2:
            st.metric("Chunks Processed", research.get('num_chunks', 'N/A'))
        with col3:
            quality = research.get('search_quality', 0.0)
            st.metric("Search Quality", f"{quality:.2f}")

        st.caption(f"Generated: {research['timestamp'][:10]}")

        # Display report content
        st.markdown(research['report'])

        # Display sources in expandable section
        st.markdown("---")
        with st.expander(f"📚 View All Sources ({len(research['sources'])})", expanded=False):
            for i, source in enumerate(research['sources'], 1):
                st.markdown(f"""
                **{i}. [{source['title']}]({source['url']})**
                - Relevance: {source.get('relevance_score', 'N/A')}
                - Retrieved: {source.get('retrieved_date', '')[:10]}
                """)
                if i < len(research['sources']):
                    st.markdown("---")

        # Export options
        st.markdown("---")

        # Prepare report content for download
        download_content = f"""{'=' * 80}
RESEARCH REPORT
{'=' * 80}

Topic: {research['topic']}
Generated: {research['timestamp']}
Sources: {research['num_sources']}
Chunks Processed: {research.get('num_chunks', 'N/A')}
Search Quality: {research.get('search_quality', 0.0):.2f}

{'=' * 80}

{research['report']}

{'=' * 80}
SOURCES
{'=' * 80}

"""
        # Add all sources to download content
        for i, source in enumerate(research['sources'], 1):
            download_content += f"""
{i}. {source['title']}
   URL: {source['url']}
   Relevance Score: {source.get('relevance_score', 'N/A')}
   Retrieved: {source.get('retrieved_date', 'N/A')[:10]}

"""

        download_content += f"\n{'=' * 80}\n"
        download_content += "Generated by RAG Research Agent\n"
        download_content += f"{'=' * 80}\n"

        # Create filename based on topic and timestamp
        filename_base = f"research_{research['topic'][:30].replace(' ', '_')}_{research['timestamp'][:10]}"
        txt_filename = f"{filename_base}.txt"
        docx_filename = f"{filename_base}.docx"

        # Create two columns for download buttons
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download as Text File",
                data=download_content,
                file_name=txt_filename,
                mime="text/plain",
                help="Plain text format - no hyperlinks"
            )

        with col2:
            # Generate DOCX file
            docx_file = create_research_docx(research)

            st.download_button(
                label="📥 Download as DOCX (Recommended)",
                data=docx_file,
                file_name=docx_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                help="Word document with clickable hyperlinks"
            )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # This block runs when you execute: streamlit run app.py
    main()
