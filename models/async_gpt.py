"""
Async interface for OpenAI GPT models to enable parallel evaluation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import openai

from config.model_config import OpenAIConfig
from models.async_base import AsyncModelInterface

logger = logging.getLogger(__name__)


class AsyncGPTModel(AsyncModelInterface):
    """Async interface for OpenAI GPT models."""
    
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
        self.model_params = config.get_params()
    
    async def generate_text_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """
        Generate text using GPT model asynchronously.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            
        Returns:
            Generated text
        """
        try:
            # Prepare the message content
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            
            # Add images if provided
            if images:
                for image in images:
                    content.append(image)
            
            # Create the message
            messages = [{"role": "user", "content": content}]
            
            # Make the async API call - check if model supports new parameters
            call_params = {
                "model": self.model_params["model"],
                "messages": messages
            }

            # Handle gpt-5 specific parameters
            if "gpt-5" in self.model_params["model"]:
                # gpt-5 uses max_completion_tokens instead of max_tokens
                call_params["max_completion_tokens"] = self.model_params.get("max_tokens", 2048)
                # gpt-5 doesn't support custom temperature, uses default (1)
                # Add gpt-5 specific parameters
                call_params["reasoning_effort"] = "minimal"
                call_params["verbosity"] = "low"
            else:
                # For other GPT models, use standard parameters
                call_params["max_tokens"] = self.model_params.get("max_tokens", 2048)
                call_params["temperature"] = self.model_params.get("temperature", 0.2)

            response = await self.client.chat.completions.create(**call_params)

            # Extract and return the response text
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating text with {self.name}: {e}")
            raise

