"""
Vision Processing Module for RAG Agent.

This module uses OpenAI GPT-4 Vision to analyze images and generate descriptions.

Learning Note - What is Vision Processing?
-----------------------------------------
Vision AI models like GPT-4 Vision can:
1. Describe what's in an image
2. Read text from images (OCR)
3. Analyze charts and graphs
4. Explain diagrams and flowcharts
5. Answer questions about visual content

This enables the RAG agent to understand and retrieve visual information,
not just text!
"""

from typing import List, Dict, Any, Optional
import time
from openai import OpenAI

from config import settings


class VisionProcessor:
    """
    Process images using OpenAI Vision API to generate descriptions.

    Learning Note - How Vision Processing Works:
    -------------------------------------------
    1. Take image (base64 encoded)
    2. Send to GPT-4 Vision with prompt
    3. Get back text description
    4. Store description with image metadata

    The description can then be:
    - Embedded as a text chunk
    - Retrieved based on semantic similarity
    - Shown to user with the original image
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize vision processor.

        Args:
            api_key: OpenAI API key (uses settings if not provided)
            model: Vision model to use (uses settings if not provided)

        Learning Note - Model Selection:
        -------------------------------
        gpt-4o: Latest, fastest, cheapest vision model (RECOMMENDED)
        - $0.01 per image (low detail)
        - $0.03 per image (high detail)
        - Great quality, fast response

        gpt-4-vision-preview: Original vision model
        - Similar pricing
        - Slightly slower

        gpt-4o-mini: Budget option
        - $0.003 per image (low detail)
        - Good for simple images
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.VISION_MODEL
        self.detail_level = settings.IMAGE_DETAIL_LEVEL
        self.client = OpenAI(api_key=self.api_key)

    def analyze_image(
        self,
        image_data: str,
        prompt: str = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze an image using GPT-4 Vision.

        Args:
            image_data: Base64 encoded image data
            prompt: Custom prompt (uses default if not provided)
            max_retries: Maximum retry attempts on failure

        Returns:
            Dictionary with:
            - description: Text description of the image
            - success: Whether analysis succeeded
            - error: Error message if failed

        Learning Note - The Vision Prompt:
        ---------------------------------
        The prompt tells GPT-4 Vision what information to extract.
        Good prompts:
        - Are specific about what to look for
        - Ask for structured information
        - Mention document context

        Example:
            >>> processor = VisionProcessor()
            >>> result = processor.analyze_image(base64_image)
            >>> print(result['description'])
        """
        if not prompt:
            # Default prompt for document images
            prompt = """Analyze this image from a document. Provide a detailed but concise description.

Focus on:
1. **Type**: What kind of image is this? (chart, graph, diagram, screenshot, photo, table, etc.)
2. **Content**: What does it show or explain?
3. **Text**: If there's text in the image, include the key text content
4. **Data**: If it's a chart/graph, describe the data and trends
5. **Purpose**: What information does this convey?

Be specific and factual. If you can read any labels, titles, or captions, include them.
Format your response as clear, readable text suitable for search and retrieval."""

        # Prepare image URL for API
        image_url = f"data:image/jpeg;base64,{image_data}"

        for attempt in range(max_retries):
            try:
                # Call OpenAI Vision API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                        "detail": self.detail_level
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,  # Enough for detailed description
                    temperature=0.3  # Lower temperature for factual descriptions
                )

                # Extract description
                description = response.choices[0].message.content.strip()

                return {
                    'description': description,
                    'success': True,
                    'error': None,
                    'model': self.model,
                    'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0
                }

            except Exception as e:
                error_msg = str(e)
                print(f"Vision API error (attempt {attempt + 1}/{max_retries}): {error_msg}")

                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # All retries failed
                    return {
                        'description': None,
                        'success': False,
                        'error': error_msg,
                        'model': self.model,
                        'tokens_used': 0
                    }

    def process_images_batch(
        self,
        images: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images and generate descriptions.

        Args:
            images: List of image dictionaries from ImageExtractor
            show_progress: Whether to print progress

        Returns:
            List of images with added 'description' field

        Learning Note - Batch Processing:
        --------------------------------
        Processing images one by one allows:
        1. Progress tracking (important for slow API calls)
        2. Individual error handling (one failure doesn't stop all)
        3. Cost estimation (track tokens per image)

        Cost example with 10 images:
        - 10 images × $0.01 = $0.10
        - Takes ~5-10 seconds total

        Example:
            >>> processor = VisionProcessor()
            >>> images_with_descriptions = processor.process_images_batch(images)
        """
        if not images:
            return []

        processed_images = []
        total = len(images)
        total_tokens = 0

        if show_progress:
            print(f"\nAnalyzing {total} images with {self.model}...")

        for idx, image in enumerate(images, 1):
            if show_progress:
                print(f"Processing image {idx}/{total}...", end=" ")

            # Analyze image
            result = self.analyze_image(image['image_data'])

            if result['success']:
                # Add description to image metadata
                image['description'] = result['description']
                image['vision_model'] = result['model']
                image['tokens_used'] = result['tokens_used']
                total_tokens += result['tokens_used']

                if show_progress:
                    print(f"✓ {len(result['description'])} chars")

                processed_images.append(image)
            else:
                if show_progress:
                    print(f"✗ Failed: {result['error']}")

                # Still include image but mark as failed
                image['description'] = f"[Image analysis failed: {result['error']}]"
                image['vision_model'] = result['model']
                image['tokens_used'] = 0
                processed_images.append(image)

        if show_progress:
            print(f"\nCompleted: {len(processed_images)}/{total} images processed")
            print(f"Total tokens used: {total_tokens}")
            estimated_cost = total_tokens * 0.00001  # Rough estimate
            print(f"Estimated cost: ${estimated_cost:.4f}")

        return processed_images


# ============================================================================
# Convenience function
# ============================================================================

def analyze_images(images: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to analyze a list of images.

    Args:
        images: List of image dictionaries from ImageExtractor
        show_progress: Whether to show progress

    Returns:
        List of images with descriptions

    Usage:
        >>> from modules.vision_processor import analyze_images
        >>> from modules.image_extractor import extract_images_from_file
        >>>
        >>> images = extract_images_from_file("document.pdf", "pdf")
        >>> analyzed_images = analyze_images(images)
    """
    processor = VisionProcessor()
    return processor.process_images_batch(images, show_progress)
