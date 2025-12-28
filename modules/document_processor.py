"""
Document Processing Module for RAG Agent.

This module handles:
1. Parsing different document formats (PDF, DOCX, TXT)
2. Intelligent text chunking with overlap
3. Metadata attachment for source tracking

Learning Note:
Document processing is the first step in the RAG pipeline. Good chunking
is crucial because:
- It determines how precisely we can retrieve information
- Poor chunking can split important context
- Proper overlap maintains continuity between chunks
"""

import re
from datetime import datetime
from typing import List, Dict, Any, BinaryIO
from pathlib import Path

# Document parsing libraries
import pypdf
from docx import Document
import chardet

# Configuration
from config import settings


class DocumentParser:
    """
    Parses documents of various formats and extracts text.

    Supported formats:
    - PDF (.pdf) - using pypdf
    - Microsoft Word (.docx) - using python-docx
    - Plain text (.txt) - with automatic encoding detection
    """

    @staticmethod
    def parse_pdf(file: BinaryIO) -> str:
        """
        Extract text from a PDF file.

        Args:
            file: Binary file object (PDF)

        Returns:
            str: Extracted text from all pages

        Learning Note:
            PDF parsing can be tricky because:
            - PDFs can contain images, tables, and complex layouts
            - Text extraction isn't always perfect
            - Some PDFs are scanned images (would need OCR)
            - Password-protected PDFs need special handling
        """
        try:
            # Create PDF reader object
            pdf_reader = pypdf.PdfReader(file)

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()

                if page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)

            # Combine all pages
            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                raise ValueError("PDF appears to be empty or contains only images")

            return full_text

        except pypdf.errors.PdfReadError as e:
            raise ValueError(f"Could not read PDF file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")

    @staticmethod
    def parse_docx(file: BinaryIO) -> str:
        """
        Extract text from a Microsoft Word document.

        Args:
            file: Binary file object (DOCX)

        Returns:
            str: Extracted text from all paragraphs

        Learning Note:
            DOCX files are actually ZIP archives containing XML files.
            The python-docx library handles this complexity for us.
            It extracts text from paragraphs but may miss:
            - Headers and footers
            - Text in text boxes
            - Text in tables (though we extract these)
        """
        try:
            # Load document
            doc = Document(file)

            # Extract text from all paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

            # Also extract text from tables
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        table_texts.append(row_text)

            # Combine paragraphs and tables
            all_text = paragraphs + table_texts
            full_text = "\n\n".join(all_text)

            if not full_text.strip():
                raise ValueError("Document appears to be empty")

            return full_text

        except Exception as e:
            raise ValueError(f"Error parsing DOCX: {str(e)}")

    @staticmethod
    def parse_txt(file: BinaryIO) -> str:
        """
        Read a plain text file with automatic encoding detection.

        Args:
            file: Binary file object (TXT)

        Returns:
            str: Text content

        Learning Note:
            Text files can have different encodings:
            - UTF-8 (most common, supports all languages)
            - ASCII (English only, subset of UTF-8)
            - Latin-1, Windows-1252, etc.

            The chardet library detects encoding automatically,
            preventing errors with non-UTF-8 files.
        """
        try:
            # Read raw bytes
            raw_bytes = file.read()

            # Detect encoding
            detected = chardet.detect(raw_bytes)
            encoding = detected.get('encoding', 'utf-8')

            # Fallback to utf-8 if detection failed
            if not encoding:
                encoding = 'utf-8'

            # Decode with detected encoding
            text = raw_bytes.decode(encoding, errors='replace')

            if not text.strip():
                raise ValueError("Text file appears to be empty")

            return text

        except Exception as e:
            raise ValueError(f"Error parsing text file: {str(e)}")

    @staticmethod
    def parse_document(file: BinaryIO, file_type: str) -> str:
        """
        Route document to appropriate parser based on file type.

        Args:
            file: Binary file object
            file_type: File extension (e.g., '.pdf', '.docx', '.txt')

        Returns:
            str: Extracted text

        Raises:
            ValueError: If file type is not supported or parsing fails
        """
        file_type = file_type.lower()

        if file_type == '.pdf':
            return DocumentParser.parse_pdf(file)
        elif file_type == '.docx':
            return DocumentParser.parse_docx(file)
        elif file_type == '.txt':
            return DocumentParser.parse_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


class TextChunker:
    """
    Intelligently chunks text into smaller pieces with overlap.

    Why chunking matters:
    - LLMs have limited context windows
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    """

    @staticmethod
    def chunk_by_sentences(
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[str]:
        """
        Split text into chunks, trying to preserve sentence boundaries.

        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks

        Learning Note:
            Sentence-aware chunking is better than simple character splitting because:
            - Complete sentences make more semantic sense
            - Easier for embeddings to capture meaning
            - Better for human readability
            - Prevents awkward mid-sentence cuts

        Algorithm:
        1. Split text into sentences
        2. Accumulate sentences until chunk_size is reached
        3. Start new chunk with overlap from previous chunk
        4. Repeat until all text is processed
        """
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE

        if overlap is None:
            overlap = settings.CHUNK_OVERLAP

        # Split text into sentences using regex
        # This pattern looks for sentence-ending punctuation followed by space
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence would exceed chunk_size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) > chunk_size and current_chunk:
                # Current chunk is full, save it and start new one
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap from previous chunk
                # Take the last 'overlap' characters from current chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

                # Find the start of the first complete word in overlap
                # (avoid starting mid-word)
                space_idx = overlap_text.find(' ')
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1:]

                current_chunk = overlap_text + " " + sentence
            else:
                # Add sentence to current chunk
                current_chunk = potential_chunk

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def add_metadata(
        chunks: List[str],
        filename: str,
        file_type: str
    ) -> List[Dict[str, Any]]:
        """
        Attach metadata to each chunk for source tracking.

        Args:
            chunks: List of text chunks
            filename: Source filename
            file_type: File extension (e.g., '.pdf')

        Returns:
            List of dictionaries with 'text' and 'metadata' keys

        Learning Note:
            Metadata is crucial for:
            - Source attribution (which document/chunk did this come from?)
            - Debugging (why was this chunk retrieved?)
            - Filtering (search within specific documents)
            - Timestamps (when was this uploaded?)

        Metadata schema:
        {
            'filename': str - original filename
            'chunk_id': int - position in document (0-indexed)
            'file_type': str - file extension
            'upload_date': str - ISO format timestamp
            'total_chunks': int - how many chunks in this document
        }
        """
        chunks_with_metadata = []

        for idx, chunk_text in enumerate(chunks):
            chunk_dict = {
                'text': chunk_text,
                'metadata': {
                    'filename': filename,
                    'chunk_id': idx,
                    'file_type': file_type,
                    'upload_date': datetime.now().isoformat(),
                    'total_chunks': len(chunks)
                }
            }
            chunks_with_metadata.append(chunk_dict)

        return chunks_with_metadata

    @staticmethod
    def process_document(
        file: BinaryIO,
        filename: str,
        file_type: str
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: Parse document, chunk text, add metadata.

        Args:
            file: Binary file object
            filename: Original filename
            file_type: File extension

        Returns:
            List of chunks with metadata

        This is the main entry point for document processing.
        It orchestrates the entire pipeline:
        1. Parse document to extract text
        2. Chunk text intelligently
        3. Add metadata for tracking

        Example:
            >>> with open('document.pdf', 'rb') as f:
            >>>     chunks = TextChunker.process_document(f, 'document.pdf', '.pdf')
            >>> print(f"Created {len(chunks)} chunks")
            >>> print(f"First chunk: {chunks[0]['text'][:100]}...")
        """
        # Step 1: Parse document to get text
        text = DocumentParser.parse_document(file, file_type)

        # Step 2: Chunk the text
        chunks = TextChunker.chunk_by_sentences(text)

        # Step 3: Add metadata
        chunks_with_metadata = TextChunker.add_metadata(
            chunks,
            filename,
            file_type
        )

        return chunks_with_metadata
