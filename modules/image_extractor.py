"""
Image Extraction Module for RAG Agent.

This module extracts images from documents (PDF, DOCX) for vision processing.

Learning Note - Why Extract Images?
-----------------------------------
Documents often contain valuable visual information:
- Charts and graphs showing data trends
- Diagrams explaining concepts
- Screenshots with important details
- Tables (though text extraction handles most tables)

By extracting and analyzing images, we can:
1. Answer questions about visual content
2. Retrieve relevant images based on their content
3. Provide more comprehensive document understanding
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import base64
from io import BytesIO

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("WARNING: PyMuPDF not available. PDF image extraction will be limited.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow not available. Image processing will be limited.")

try:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import _Cell, Table
    from docx.text.paragraph import Paragraph
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("WARNING: python-docx not available. DOCX image extraction will be disabled.")

from config import settings


class ImageExtractor:
    """
    Extract images from PDF and DOCX files.

    Learning Note - Extraction Strategy:
    -----------------------------------
    Different file formats store images differently:

    PDF: Images are embedded as objects
    - Use PyMuPDF to iterate through pages
    - Extract image objects with metadata (size, position)
    - Filter out small images (icons, bullets)

    DOCX: Images are part of document structure
    - Images stored in document relationships
    - Can be in paragraphs, tables, headers, etc.
    - Extract via python-docx library
    """

    def __init__(self):
        """Initialize image extractor with configuration."""
        self.min_width = settings.MIN_IMAGE_WIDTH
        self.min_height = settings.MIN_IMAGE_HEIGHT
        self.max_images = settings.MAX_IMAGES_PER_DOCUMENT
        self.temp_dir = Path(settings.IMAGE_TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of image dictionaries with metadata

        Learning Note - PDF Image Extraction:
        ------------------------------------
        PyMuPDF (fitz) provides powerful image extraction:
        1. Open PDF document
        2. Iterate through each page
        3. Get list of images on page
        4. Extract image data and metadata
        5. Filter by size to remove decorative elements

        Example:
            >>> extractor = ImageExtractor()
            >>> images = extractor.extract_from_pdf("report.pdf")
            >>> print(f"Found {len(images)} images")
        """
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF not available. Cannot extract images from PDF.")
            return []

        if not PIL_AVAILABLE:
            print("Pillow not available. Cannot process images.")
            return []

        images = []

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            print(f"Extracting images from PDF: {Path(pdf_path).name}")
            print(f"Total pages: {len(pdf_document)}")

            # Iterate through pages
            for page_num in range(len(pdf_document)):
                if len(images) >= self.max_images:
                    print(f"Reached maximum images limit ({self.max_images}). Stopping extraction.")
                    break

                page = pdf_document[page_num]
                image_list = page.get_images()

                # Extract each image on the page
                for img_index, img_info in enumerate(image_list):
                    if len(images) >= self.max_images:
                        break

                    try:
                        # Get image xref (reference)
                        xref = img_info[0]

                        # Extract image
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Load image to check dimensions
                        img = Image.open(BytesIO(image_bytes))
                        width, height = img.size

                        # Filter small images
                        if width < self.min_width or height < self.min_height:
                            continue

                        # Convert to base64 for storage/API calls
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')

                        # Store image info
                        images.append({
                            'image_data': base64_image,
                            'format': image_ext,
                            'width': width,
                            'height': height,
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'source_type': 'pdf'
                        })

                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue

            pdf_document.close()

            print(f"Extracted {len(images)} images from PDF")
            return images

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def extract_from_docx(self, docx_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from a DOCX file.

        Args:
            docx_path: Path to DOCX file

        Returns:
            List of image dictionaries with metadata

        Learning Note - DOCX Image Extraction:
        -------------------------------------
        DOCX files store images in the document's relationship parts.
        Images can appear in:
        - Regular paragraphs
        - Tables
        - Headers/footers
        - Text boxes

        We extract all embedded images and their metadata.

        Example:
            >>> extractor = ImageExtractor()
            >>> images = extractor.extract_from_docx("document.docx")
        """
        if not DOCX_AVAILABLE:
            print("python-docx not available. Cannot extract images from DOCX.")
            return []

        if not PIL_AVAILABLE:
            print("Pillow not available. Cannot process images.")
            return []

        images = []

        try:
            doc = Document(docx_path)

            print(f"Extracting images from DOCX: {Path(docx_path).name}")

            # Get all image parts from document relationships
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    if len(images) >= self.max_images:
                        print(f"Reached maximum images limit ({self.max_images}). Stopping extraction.")
                        break

                    try:
                        # Get image data
                        image_data = rel.target_part.blob

                        # Load image to check dimensions
                        img = Image.open(BytesIO(image_data))
                        width, height = img.size
                        format_type = img.format.lower() if img.format else 'unknown'

                        # Filter small images
                        if width < self.min_width or height < self.min_height:
                            continue

                        # Convert to base64
                        base64_image = base64.b64encode(image_data).decode('utf-8')

                        # Store image info
                        images.append({
                            'image_data': base64_image,
                            'format': format_type,
                            'width': width,
                            'height': height,
                            'source_type': 'docx',
                            'image_index': len(images)
                        })

                    except Exception as e:
                        print(f"Error extracting image: {e}")
                        continue

            print(f"Extracted {len(images)} images from DOCX")
            return images

        except Exception as e:
            print(f"Error processing DOCX: {e}")
            return []

    def extract_images(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Extract images from a document (PDF or DOCX).

        Args:
            file_path: Path to document file
            file_type: Type of file ('pdf' or 'docx')

        Returns:
            List of extracted images with metadata

        This is the main entry point for image extraction.
        Routes to appropriate extraction method based on file type.

        Example:
            >>> extractor = ImageExtractor()
            >>> images = extractor.extract_images("/path/to/doc.pdf", "pdf")
            >>> print(f"Extracted {len(images)} images")
        """
        if not settings.ENABLE_IMAGE_PROCESSING:
            return []

        if file_type.lower() == 'pdf':
            return self.extract_from_pdf(file_path)
        elif file_type.lower() in ['docx', 'doc']:
            return self.extract_from_docx(file_path)
        else:
            print(f"Image extraction not supported for file type: {file_type}")
            return []


# ============================================================================
# Convenience function
# ============================================================================

def extract_images_from_file(file_path: str, file_type: str) -> List[Dict[str, Any]]:
    """
    Convenience function to extract images from a file.

    Args:
        file_path: Path to document
        file_type: File type ('pdf', 'docx', etc.)

    Returns:
        List of extracted images

    Usage:
        >>> from modules.image_extractor import extract_images_from_file
        >>> images = extract_images_from_file("document.pdf", "pdf")
    """
    extractor = ImageExtractor()
    return extractor.extract_images(file_path, file_type)
