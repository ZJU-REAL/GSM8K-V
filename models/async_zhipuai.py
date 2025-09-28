"""
Async interface for ZhipuAI GLM models to enable parallel evaluation.
"""

import asyncio
import logging
import json
import aiohttp
import base64
from typing import Dict, List, Any, Optional

from config.model_config import ZhipuAIConfig
from models.async_base import AsyncModelInterface

logger = logging.getLogger(__name__)


class AsyncZhipuAIModel(AsyncModelInterface):
    """Async interface for ZhipuAI GLM models using HTTP API."""
    
    def __init__(self, config: ZhipuAIConfig):
        super().__init__(config)
        
        # ZhipuAI API endpoint
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.api_key = config.api_key
        self.model_params = config.get_params()
        
        logger.info(f"Initialized async ZhipuAI model: {self.name}")
    
    async def generate_text_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """
        Generate text using ZhipuAI GLM model asynchronously.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            
        Returns:
            Generated text
        """
        try:
            # Prepare message content
            content = []
            
            # Add text content
            if prompt:
                content.append({
                    "type": "text",
                    "text": prompt
                })
            
            # Add images if provided
            if images and len(images) > 0:
                logger.debug(f"Processing {len(images)} images for ZhipuAI")
                
                for i, image in enumerate(images):
                    if isinstance(image, dict):
                        try:
                            # Handle different image formats
                            if "data" in image and "mime_type" in image:
                                # Gemini format: {"mime_type": "image/png", "data": bytes}
                                if isinstance(image["data"], bytes):
                                    b64_data = base64.b64encode(image["data"]).decode('utf-8')
                                else:
                                    b64_data = image["data"]
                                
                                # ZhipuAI uses image_url format similar to OpenAI
                                data_url = f"data:{image['mime_type']};base64,{b64_data}"
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                })
                                logger.debug(f"Converted Gemini format image {i+1} to ZhipuAI format")
                            
                            elif "type" in image and image["type"] == "image_url":
                                # Already in OpenAI/ZhipuAI format
                                content.append(image)
                                logger.debug(f"Used image {i+1} in ZhipuAI format")
                            
                            elif "source" in image:
                                # Claude format
                                if "data" in image["source"]:
                                    media_type = image["source"].get("media_type", "image/png")
                                    data_url = f"data:{media_type};base64,{image['source']['data']}"
                                    content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": data_url
                                        }
                                    })
                                    logger.debug(f"Converted Claude format image {i+1} to ZhipuAI format")
                            
                            elif "type" in image and image["type"] == "image" and "data" in image:
                                # Qwen format
                                data_url = f"data:image/png;base64,{image['data']}"
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                })
                                logger.debug(f"Converted Qwen format image {i+1} to ZhipuAI format")
                            
                            else:
                                logger.warning(f"Unknown image format for image {i+1}: {list(image.keys())}")
                        
                        except Exception as img_error:
                            logger.error(f"Error processing image {i+1}: {img_error}")
                            continue
            
            # Prepare the payload according to ZhipuAI API format
            payload = {
                "model": self.model_params["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": self.model_params["temperature"],
                "max_tokens": self.model_params["max_tokens"]
            }
            
            # Add thinking parameter for GLM-4.5V
            if "glm-4.5v" in self.model_params["model"]:
                payload["thinking"] = {"type": "enabled"}
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
                        
            logger.debug(f"Making request to ZhipuAI with {len([c for c in content if c.get('type') == 'image_url'])} images")
            
            # Make the async HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract response text from ZhipuAI format
                        if "choices" in result and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                return choice["message"]["content"].strip()
                        
                        logger.warning(f"Unexpected response format from ZhipuAI API: {result}")
                        return "Error: Unexpected response format"
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"ZhipuAI HTTP error {response.status}: {error_text}")
                        return f"Error: HTTP {response.status} - {error_text}"
        
        except Exception as e:
            logger.error(f"Error generating text with {self.name}: {e}")
            raise
