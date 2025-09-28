"""
Async interface for Google Gemini models to enable parallel evaluation.
"""

import asyncio
import logging
import base64
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types

from config.model_config import GeminiConfig
from models.async_base import AsyncModelInterface

logger = logging.getLogger(__name__)


class AsyncGeminiModel(AsyncModelInterface):
    """Async interface for Google Gemini models using new google.genai library."""
    
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        
        # Initialize Google Generative AI client with API key
        self.client = genai.Client(api_key=config.api_key)
        
        # Store model name and parameters
        self.model_name = config.name
        self.model_params = config.get_params()
        
        logger.info(f"Initialized async Gemini model: {self.name}")
    
    async def generate_text_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """
        Generate text using Gemini model asynchronously via new google.genai API.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            
        Returns:
            Generated text
        """
        try:
            # Create content parts for the request
            contents = []
            
            # Add images first if available (following Gemini API best practices)
            if images and len(images) > 0:
                for image in images:
                    if isinstance(image, dict):
                        # Handle different image formats
                        if "data" in image and "mime_type" in image:
                            # Format from image processor: {"mime_type": "image/png", "data": bytes}
                            image_part = types.Part.from_bytes(
                                data=image["data"],
                                mime_type=image["mime_type"]
                            )
                            contents.append(image_part)
                        elif "source" in image:
                            # Legacy format support
                            if "data" in image["source"]:
                                # Base64 encoded data
                                image_bytes = base64.b64decode(image["source"]["data"])
                                image_part = types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type="image/png"
                                )
                                contents.append(image_part)
                            elif "path" in image["source"]:
                                # File path
                                with open(image["source"]["path"], "rb") as img_file:
                                    image_bytes = img_file.read()
                                    image_part = types.Part.from_bytes(
                                        data=image_bytes,
                                        mime_type="image/png"
                                    )
                                    contents.append(image_part)
            
            # Add the text prompt after images
            if prompt:
                contents.append(prompt)
            
            # Generate content asynchronously using executor since google.genai doesn't have native async support yet
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents
                )
            )
            
            # Extract and return response text
            if response and hasattr(response, "text"):
                return response.text.strip()
            else:
                logger.warning(f"Empty response from Gemini API")
                return "Error: Empty response from Gemini API"
        
        except Exception as e:
            logger.error(f"Error generating text with {self.name}: {e}")
            raise 