"""
Async interface for vLLM models served via OpenAI-compatible API.
Supports both text-only and multimodal inputs with configurable endpoints.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import openai

from config.model_config import VLLMConfig
from models.async_base import AsyncModelInterface

logger = logging.getLogger(__name__)


class AsyncVLLMModel(AsyncModelInterface):
    """Async interface for vLLM models served via OpenAI-compatible API."""

    def __init__(self, config: VLLMConfig):
        super().__init__(config)

        # Initialize AsyncOpenAI client with custom base URL
        self.client = openai.AsyncOpenAI(
            api_key="vllm",  # vLLM typically doesn't require authentication
            base_url=config.api_base
        )
        self.model_params = config.get_params()

        logger.info(f"Initialized async vLLM model: {self.name} at {config.api_base}")

    async def generate_text_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """
        Generate text using vLLM model asynchronously via OpenAI-compatible API.

        Args:
            prompt: Text prompt
            images: Optional list of images (for multimodal models)

        Returns:
            Generated text
        """
        try:
            # Prepare message content
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})

            # Add images if provided (vLLM supports multimodal through OpenAI format)
            if images and len(images) > 0:
                logger.debug(f"Processing {len(images)} images for vLLM model")

                # Check if this is an Ovis model (needs special image format)
                is_ovis_model = False

                if is_ovis_model:
                    # Ovis models need PIL Image objects, but we need to convert them to base64 for API
                    logger.debug("Using Ovis-compatible image format via base64")
                    for i, image in enumerate(images):
                        try:
                            if isinstance(image, dict):
                                if "data" in image and "mime_type" in image:
                                    # Convert image data to base64
                                    img_data = image["data"]
                                    if isinstance(img_data, bytes):
                                        import base64
                                        b64_data = base64.b64encode(img_data).decode('utf-8')
                                        mime_type = image["mime_type"]
                                        data_url = f"data:{mime_type};base64,{b64_data}"
                                        content.append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": data_url
                                            }
                                        })
                                        logger.debug(f"Converted image {i+1} to base64 for Ovis model")
                                    else:
                                        logger.warning(f"Image {i+1} data is not bytes for Ovis model")
                                elif "type" in image and image["type"] == "image_url":
                                    # OpenAI format - already in correct format
                                    content.append(image)
                                    logger.debug(f"Used image {i+1} in OpenAI format for Ovis model")
                                else:
                                    logger.warning(f"Unsupported image format for Ovis model: {list(image.keys())}")
                            elif hasattr(image, 'convert'):  # PIL Image object
                                # Convert PIL Image to base64
                                from io import BytesIO
                                pil_image = image.convert("RGB")
                                buffer = BytesIO()
                                pil_image.save(buffer, format="PNG")
                                img_data = buffer.getvalue()

                                import base64
                                b64_data = base64.b64encode(img_data).decode('utf-8')
                                data_url = f"data:image/png;base64,{b64_data}"
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                })
                                logger.debug(f"Converted PIL image {i+1} to base64 for Ovis model")
                            else:
                                logger.warning(f"Unknown image type for Ovis model: {type(image)}")
                        except Exception as img_error:
                            logger.error(f"Error processing image {i+1} for Ovis model: {img_error}")
                            continue
                else:
                    # Standard vLLM models use OpenAI-compatible format
                    for i, image in enumerate(images):
                        if isinstance(image, dict):
                            try:
                                # Handle different image formats
                                if "data" in image and "mime_type" in image:
                                    # Standard format: {"mime_type": "image/png", "data": bytes}
                                    if isinstance(image["data"], bytes):
                                        import base64
                                        b64_data = base64.b64encode(image["data"]).decode('utf-8')
                                    else:
                                        b64_data = image["data"]

                                    data_url = f"data:{image['mime_type']};base64,{b64_data}"
                                    content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": data_url
                                        }
                                    })
                                    logger.debug(f"Converted image {i+1} to vLLM format")

                                elif "type" in image and image["type"] == "image_url":
                                    # Already in correct format
                                    content.append(image)
                                    logger.debug(f"Used image {i+1} in OpenAI format")

                                elif "source" in image:
                                    # Claude format support
                                    if "data" in image["source"]:
                                        media_type = image["source"].get("media_type", "image/png")
                                        data_url = f"data:{media_type};base64,{image['source']['data']}"
                                        content.append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": data_url
                                            }
                                        })
                                        logger.debug(f"Converted Claude format image {i+1} to vLLM format")

                                else:
                                    logger.warning(f"Unknown image format for image {i+1}: {list(image.keys())}")

                            except Exception as img_error:
                                logger.error(f"Error processing image {i+1}: {img_error}")
                                continue

            # Create messages
            messages = [{"role": "user", "content": content}]

            logger.debug(f"Making request to vLLM with {len([c for c in content if c.get('type') == 'image_url'])} images")

            # Make the async API call
            response = await self.client.chat.completions.create(
                model=self.model_params["model"],
                messages=messages,
                temperature=self.model_params["temperature"],
                max_tokens=self.model_params["max_tokens"]
            )

            # Extract and return the response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                logger.warning("Empty response from vLLM API")
                return "Error: Empty response from vLLM API"

        except Exception as e:
            logger.error(f"Error generating text with {self.name}: {e}")
            raise
