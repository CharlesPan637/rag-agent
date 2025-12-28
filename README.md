# 📚 RAG Agent - Document Q&A System

A beginner-friendly Retrieval Augmented Generation (RAG) system that allows you to upload documents and ask questions about them using AI. Built with Streamlit, ChromaDB, and Anthropic's Claude.

## 🎯 What is RAG?

**RAG (Retrieval Augmented Generation)** combines the power of semantic search with large language models:

1. **Upload** your documents (PDFs, Word docs, text files)
2. **Ask** questions in natural language
3. **Get** accurate answers based on your documents with source citations

**Why RAG?**
- LLMs don't know about YOUR specific documents
- RAG retrieves relevant context from your documents
- The LLM generates answers grounded in that context
- Reduces hallucination and provides transparency through citations

## ✨ Features

- 📄 **Multi-format support**: PDF, Microsoft Word (.docx), and plain text files
- 🔍 **Semantic search**: Finds relevant content based on meaning, not just keywords
- 💾 **Persistent storage**: Documents saved between sessions using ChromaDB
- 📚 **Source citations**: See exactly which document and section answers came from
- 🤖 **Powered by Claude**: Uses Anthropic's Claude AI for high-quality answers
- 🎨 **Clean UI**: Intuitive Streamlit interface with chat-style interaction
- 🔒 **Local embeddings**: Uses sentence-transformers (runs locally, no API calls)

## 🏗️ Architecture

```
User Question
     ↓
[1. Generate Query Embedding]
     ↓
[2. Search Vector Database (ChromaDB)]
     ↓
[3. Retrieve Top-K Similar Chunks]
     ↓
[4. Format Context + Question]
     ↓
[5. Send to Claude API]
     ↓
[6. Generate Answer]
     ↓
Display Answer + Sources
```

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Anthropic API Key** - Get one from [console.anthropic.com](https://console.anthropic.com/)
- **Basic command line knowledge**
- **8GB+ RAM recommended** (for embedding model)

## 🚀 Installation

### Step 1: Clone or Download

If you have git:
```bash
git clone <your-repo-url>
cd rag_agent
```

Or download and extract the project files to a folder named `rag_agent`.

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will download several packages:
- `streamlit` - Web interface
- `anthropic` - Claude API client
- `chromadb` - Vector database
- `sentence-transformers` - Embedding model (~80MB download)
- `pypdf`, `python-docx` - Document parsing
- And their dependencies

Installation may take 2-5 minutes depending on your connection.

### Step 4: Configure API Key

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in a text editor and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_actual_api_key_here
   ```

3. Save the file

**Getting an API Key:**
1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy and paste it into your `.env` file

## 🎮 Usage

### Starting the Application

From the project directory, run:

```bash
streamlit run app.py
```

This will:
1. Start the Streamlit server
2. Automatically open your browser to `http://localhost:8501`
3. Display the RAG Agent interface

### Uploading Documents

1. Click **"Browse files"** in the sidebar
2. Select one or more documents (PDF, DOCX, or TXT)
3. Click **"Process Documents"**
4. Wait for processing to complete (you'll see a success message)

**Processing steps:**
- Extracts text from documents
- Splits text into ~1000 character chunks with overlap
- Generates embeddings for each chunk
- Stores in ChromaDB vector database

### Asking Questions

1. Type your question in the chat input at the bottom
2. Press Enter or click Send
3. The system will:
   - Find the most relevant chunks from your documents
   - Send them to Claude as context
   - Display the answer with source citations

**Example questions:**
- "What is the main purpose of this document?"
- "How do I reset my password?"
- "What are the key findings?"
- "Summarize the refund policy"

### Viewing Sources

Click the **"View Sources"** expander below each answer to see:
- Which documents were used
- Which chunks were retrieved
- Relevance scores
- The actual text from each chunk

## 📁 Project Structure

```
rag_agent/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Your API keys (DO NOT COMMIT)
├── .env.example                   # Template for environment variables
├── .gitignore                     # Files to exclude from git
├── README.md                      # This file
│
├── config/
│   └── settings.py                # Configuration management
│
├── modules/
│   ├── __init__.py
│   ├── document_processor.py     # Parse and chunk documents
│   ├── embeddings.py             # Generate vector embeddings
│   ├── vector_store.py           # ChromaDB interface
│   └── rag_query.py              # RAG pipeline + Claude integration
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                # Utility functions
│
└── data/
    ├── uploads/                  # Temporary uploaded files
    └── chroma_db/                # ChromaDB persistent storage
```

## 🧠 How It Works (Educational)

### 1. Document Processing

**Parsing:**
- **PDF**: Uses `pypdf` to extract text from each page
- **DOCX**: Uses `python-docx` to extract paragraphs and tables
- **TXT**: Reads with automatic encoding detection

**Chunking:**
- Splits text into ~1000 character chunks
- Maintains 200 character overlap between chunks
- Preserves sentence boundaries when possible

**Why chunk?**
- LLMs have limited context windows
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost at boundaries

### 2. Embeddings

**What are embeddings?**

Embeddings convert text into numerical vectors that capture semantic meaning:

```
"The cat sat on the mat" → [0.2, -0.5, 0.8, 0.1, ...]
"A feline rested on the rug" → [0.19, -0.48, 0.82, 0.09, ...]
```

Similar texts have similar vectors, enabling semantic search!

**Model:** `all-MiniLM-L6-v2`
- 384-dimensional vectors
- Trained on billions of text pairs
- Runs locally (free, no API calls)

### 3. Vector Database (ChromaDB)

**Why a vector database?**
- Traditional databases can't efficiently search by similarity
- Vector databases are optimized for finding similar vectors
- ChromaDB uses approximate nearest neighbor search (fast!)

**Storage:**
- Each chunk stored with its embedding
- Metadata tracked (filename, chunk ID, upload date)
- Data persists to disk (survives app restarts)

### 4. Retrieval

**When you ask a question:**

1. Question converted to embedding vector
2. ChromaDB finds the 5 most similar chunk vectors
3. Associated text chunks retrieved
4. Chunks sorted by similarity

**Distance metrics:**
- Cosine similarity measures vector "angle"
- Closer vectors = more similar meaning
- Independent of text length

### 5. Generation (Claude)

**Prompt structure:**
```
System: You are a helpful assistant that answers based on context.
Answer ONLY from the provided context...

Context:
[Retrieved chunk 1]
[Retrieved chunk 2]
...

Question: [Your question]

Answer:
```

**Why this works:**
- Explicit grounding instruction reduces hallucination
- Context provides specific information
- Claude synthesizes natural language answer
- Source attribution enables transparency

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Required
ANTHROPIC_API_KEY=your_key_here

# Optional - defaults provided
CHUNK_SIZE=1000                    # Characters per chunk
CHUNK_OVERLAP=200                  # Overlap between chunks
CLAUDE_MODEL=claude-3-5-sonnet-20241022  # Claude model
MAX_UPLOAD_SIZE_MB=10              # Max file size
```

## 💰 Cost Estimation

### Anthropic API Costs

**Claude 3.5 Sonnet:**
- Input: ~$3 per million tokens
- Output: ~$15 per million tokens

**Typical RAG query:**
- Context: 2,000-4,000 tokens (5 chunks)
- Question: ~20 tokens
- Answer: ~200 tokens
- **Cost per query: ~$0.01-0.02**

**Monthly estimate:**
- 100 queries: ~$1-2
- 1,000 queries: ~$10-20

### Other Costs

- **Embeddings**: FREE (runs locally)
- **ChromaDB**: FREE (local storage)
- **Streamlit**: FREE (self-hosted)

**Total: Only Anthropic API usage costs!**

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**2. "API Key not found"**
- Check `.env` file exists in project root
- Verify `ANTHROPIC_API_KEY` is set correctly
- No spaces around the `=` sign
- No quotes around the key

**3. "No documents uploaded"**
- Click "Process Documents" after selecting files
- Check file formats (PDF, DOCX, TXT only)
- Verify file size under 10MB

**4. "ChromaDB collection not persisting"**
- Check `data/chroma_db/` folder exists
- Verify write permissions
- Ensure `CHROMA_PERSIST_DIRECTORY` in `.env` is correct

**5. "Model download stuck"**
- First run downloads ~80MB embedding model
- May take several minutes on slow connections
- Model cached for future runs

**6. "Out of memory"**
- Embedding model needs ~500MB RAM
- Close other applications
- Process fewer documents at once

**7. "Rate limit exceeded"**
- Anthropic has API rate limits
- Wait 60 seconds and try again
- Consider upgrading API tier

## 🎓 Learning Resources

### Understanding RAG

- [Anthropic: RAG Explained](https://www.anthropic.com/index/contextual-retrieval)
- [What is Retrieval Augmented Generation?](https://www.youtube.com/results?search_query=rag+explained)
- [Vector Database Explained](https://www.pinecone.io/learn/vector-database/)

### Documentation

- [Streamlit Docs](https://docs.streamlit.io)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Anthropic Claude API](https://docs.anthropic.com)
- [Sentence Transformers](https://www.sbert.net/)

### Python Concepts Used

- **Classes and OOP**: Organizing code into reusable components
- **Type Hints**: Making code more readable and catching errors
- **Exception Handling**: Graceful error recovery
- **Environment Variables**: Secure configuration management
- **Lazy Loading**: Loading resources only when needed
- **Batch Processing**: Efficient processing of multiple items

## 🚀 Next Steps & Enhancements

Once you're comfortable with the basics, consider adding:

### Easy Enhancements

1. **Delete individual documents** (currently can only clear all)
2. **Search within specific documents** (filter by filename)
3. **Export chat history** (save conversations to file)
4. **Adjust chunk size** via UI (currently only in config)

### Intermediate Enhancements

5. **Multi-turn conversations** (remember previous questions)
6. **Hybrid search** (combine keyword + semantic search)
7. **Response streaming** (show answer as it's generated)
8. **Document preview** (show uploaded document content)

### Advanced Enhancements

9. **Re-ranking** (improve retrieval with cross-encoder)
10. **Query rewriting** (expand questions for better retrieval)
11. **Multi-modal support** (images, tables from PDFs)
12. **User authentication** (multiple users with separate collections)
13. **Deployment** (Docker, cloud hosting)

## 🤝 Contributing

This is an educational project! Contributions welcome:

- 🐛 Bug reports
- 💡 Feature suggestions
- 📖 Documentation improvements
- 🎨 UI enhancements

## 📄 License

This project is for educational purposes. Feel free to use, modify, and learn from it!

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io) - Web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Anthropic Claude](https://www.anthropic.com/) - LLM
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [pypdf](https://github.com/py-pdf/pypdf) - PDF parsing
- [python-docx](https://python-docx.readthedocs.io/) - Word parsing

## 📧 Support

Having issues? Try:

1. Check this README's troubleshooting section
2. Review the code comments (heavily documented for learning!)
3. Check the GitHub issues (if applicable)
4. Verify your API key and dependencies

## 🎉 Success Checklist

You're ready when you can:

- [ ] Install dependencies without errors
- [ ] Start the application with `streamlit run app.py`
- [ ] Upload a PDF document successfully
- [ ] Ask a question and receive an answer
- [ ] View source citations for answers
- [ ] See documents persist after restart

**Congratulations! You've built a functional RAG system!** 🎊

---

**Happy Learning! 📚🤖**

*Remember: The best way to learn is by experimenting. Try different documents, questions, and modifications to the code!*
