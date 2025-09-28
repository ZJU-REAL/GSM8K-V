"""
Async interface for Qwen models to enable parallel evaluation.
"""

import asyncio
import logging
import os
import random
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
import aiohttp
import time

from config.model_config import QwenConfig
from models.async_base import AsyncModelInterface

logger = logging.getLogger(__name__)


class AsyncQwenModel(AsyncModelInterface):
    """Async interface for Qwen models using AsyncOpenAI client with improved error handling."""
    
    def __init__(self, config: QwenConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.model_params = config.get_params()
        
        # Initialize AsyncOpenAI client with optimized settings for Qwen API
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=180.0,
            max_retries=0   # Disable built-in retries, use custom retry logic
        )
        
        self.model_name = config.name
        logger.info(f"Initialized async Qwen model using OpenAI-compatible API: {self.model_name}")
    
    def _should_enable_thinking(self, use_thinking: bool = False) -> bool:
        """
        Determine if thinking mode should be enabled for qwen-vl-plus model.
        
        Args:
            use_thinking: Whether Chain-of-Thought reasoning is enabled
            
        Returns:
            True if thinking should be enabled, False otherwise
        """
        # Enable thinking only for qwen-vl-plus model when CoT thinking is enabled
        return "qwen-vl-plus" in self.model_name and use_thinking
    
    def _is_qvq_max_model(self) -> bool:
        """
        Check if the current model is QVQ-Max which requires special streaming handling.

        Returns:
            True if model is QVQ-Max, False otherwise
        """
        return "qvq-max" in self.model_name.lower() or "qvq" in self.model_name.lower()
    
    async def _process_qvq_stream_response(self, completion) -> str:
        """
        Process streaming response for QVQ-Max model which includes reasoning content.

        Args:
            completion: Async streaming completion object

        Returns:
            Final answer content (without reasoning)
        """
        reasoning_content = ""
        answer_content = ""
        is_answering = False

        try:
            chunk_count = 0
            async for chunk in completion:
                chunk_count += 1

                # Handle usage information chunk
                if not chunk.choices:
                    logger.debug(f"Token usage for {self.model_name}: {chunk.usage}")
                    continue

                delta = chunk.choices[0].delta
                logger.debug(f"Chunk {chunk_count}: delta attributes: {[attr for attr in dir(delta) if not attr.startswith('_')]}")

                # Check for reasoning content
                if hasattr(delta, 'reasoning_content'):
                    logger.debug(f"Has reasoning_content: {delta.reasoning_content is not None}")
                    if delta.reasoning_content is not None:
                        # Limit reasoning content to avoid context overflow
                        if len(reasoning_content) + len(delta.reasoning_content) > 8000:
                            logger.warning(f"Reasoning content too long ({len(reasoning_content)}), truncating...")
                            # For QVQ-Max, keep more content that might contain image analysis
                            # Keep the beginning (which likely contains image analysis) and recent content
                            reasoning_content = reasoning_content[:4000] + "...[truncated]..." + reasoning_content[-3000:]
                        reasoning_content += delta.reasoning_content
                        logger.debug(f"Reasoning progress: {len(reasoning_content)} chars")
                        continue

                # Check for content
                if hasattr(delta, 'content'):
                    logger.debug(f"Has content: {delta.content is not None}")
                    if delta.content:
                        if not is_answering:
                            is_answering = True
                            logger.info(f"Started receiving answer for {self.model_name}")
                        answer_content += delta.content
                        logger.debug(f"Answer progress: {len(answer_content)} chars")
                else:
                    logger.debug("Delta has no content attribute")

            logger.info(f"Processed {chunk_count} chunks. Reasoning: {len(reasoning_content)} chars, Answer: {len(answer_content)} chars")

            # Log reasoning content preview for debugging
            if reasoning_content:
                # Check if reasoning content mentions images (more comprehensive check)
                has_image_mention = any(keyword in reasoning_content.lower() for keyword in [
                    'image', 'picture', 'visual', 'photo', 'diagram', 'chart', 'graph',
                    'figure', 'illustration', 'drawing', 'scene', 'view'
                ])
                logger.info(f"Reasoning content preview: {reasoning_content[:500]}...")
                logger.info(f"Reasoning mentions visual content: {has_image_mention}")

                # Additional check for QVQ-Max: ensure reasoning content shows image processing
                if self._is_qvq_max_model() and not has_image_mention and len(reasoning_content) > 100:
                    logger.warning("QVQ-Max reasoning content doesn't mention visual content - possible image processing issue")

            if answer_content:
                logger.info(f"Final answer: {answer_content}")
                # Check if answer contradicts reasoning
                if self._is_qvq_max_model() and 'no image' in answer_content.lower() and has_image_mention:
                    logger.error("QVQ-Max answer contradicts reasoning - says no image but reasoning mentions images")

            # Return the final answer content
            if answer_content:
                return answer_content.strip()
            else:
                logger.error(f"No answer content received from QVQ-Max model. Reasoning content: {reasoning_content[:200]}...")
                raise Exception("No answer content received from QVQ-Max model")

        except Exception as e:
            logger.error(f"Error processing QVQ-Max stream: {str(e)}")
            raise

    async def _exponential_backoff_delay(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> None:
        """
        Calculate and apply exponential backoff delay with jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
        """
        # Exponential backoff: base_delay * (2 ^ attempt)
        delay = min(base_delay * (2 ** attempt), max_delay)
        
        # Add jitter to avoid thundering herd problem
        jitter = random.uniform(0.1, 0.5) * delay
        final_delay = delay + jitter
        
        logger.debug(f"Applying backoff delay: {final_delay:.2f}s for attempt {attempt + 1}")
        await asyncio.sleep(final_delay)
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """
        Determine if an error is retryable based on error message.
        
        Args:
            error_msg: Error message string
            
        Returns:
            True if error is retryable, False otherwise
        """
        retryable_keywords = [
            'connection', 'network', 'timeout', 'refused', 'reset',
            'throttling', 'rate limit', 'quota', 'temporary', 'unavailable',
            'service unavailable', '502', '503', '504', '429'
        ]
        
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in retryable_keywords)
    
    async def generate_text_async(self, prompt: str, images: Optional[List[Any]] = None, use_thinking: bool = False) -> str:
        """
        Generate text using Qwen model asynchronously via OpenAI-compatible API with enhanced retry logic.
        Supports special QVQ-Max streaming with reasoning content.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            use_thinking: Whether to use Chain-of-Thought reasoning (affects thinking mode)
            
        Returns:
            Generated text
        """
        max_retries = 5  # Increase max retries
        base_delay = 2.0  # Increase base delay
        
        for attempt in range(max_retries):
            try:
                # Prepare messages for OpenAI-compatible API
                messages = []
                
                # Create user message with content
                if images and len(images) > 0:
                    # Multi-modal content with images
                    logger.info(f"Processing {len(images)} images for {self.model_name}")
                    message_content = []
                    
                    # Add images first - use QVQ-Max compatible format
                    for i, image in enumerate(images):
                        logger.debug(f"Processing image {i+1}/{len(images)} for {self.model_name}")
                        if isinstance(image, dict):
                            if "type" in image and image["type"] == "image" and "data" in image:
                                # Validate base64 data for QVQ-Max
                                base64_data = image['data']
                                if not base64_data or len(base64_data) < 100:
                                    logger.warning(f"Image {i+1} has invalid base64 data (length: {len(base64_data) if base64_data else 0})")
                                    continue

                                # Convert to QVQ-Max compatible format with proper base64 encoding
                                image_content = {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_data}"
                                    }
                                }
                                logger.debug(f"Converted image {i+1} to QVQ-Max format (data length: {len(base64_data)})")
                                message_content.append(image_content)
                            elif "image_url" in image:
                                # Already in correct format
                                logger.debug(f"Image {i+1} already in image_url format")
                                message_content.append(image)
                            elif "url" in image:
                                # Direct URL format
                                logger.debug(f"Image {i+1} in direct URL format")
                                message_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": image["url"]}
                                })
                            else:
                                logger.warning(f"Unknown image format for {self.model_name}: {image.keys()}")
                        else:
                            logger.warning(f"Invalid image format for {self.model_name}: {type(image)}")

                    # QVQ-Max specific: Ensure proper image-text interleaving
                    if self._is_qvq_max_model() and len(images) > 1:
                        logger.info(f"QVQ-Max multi-image setup: {len(images)} images detected")
                        # For multi-image, ensure proper ordering: images first, then text
                    
                    # Add text content
                    message_content.append({
                        "type": "text",
                        "text": prompt
                    })

                    user_message = {
                        "role": "user",
                        "content": message_content
                    }

                    # Debug: Log the final message structure for QVQ-Max
                    if self._is_qvq_max_model():
                        logger.info(f"QVQ-Max message structure: {len(message_content)} content items")
                        image_count = sum(1 for item in message_content if item.get('type') == 'image_url')
                        logger.info(f"QVQ-Max images in message: {image_count}")
                        for i, item in enumerate(message_content):
                            if item.get('type') == 'image_url':
                                url = item['image_url']['url']
                                logger.info(f"Content item {i}: image_url (length: {len(url)})")
                                if url.startswith('data:image/png;base64,'):
                                    data_len = len(url) - len('data:image/png;base64,')
                                    logger.info(f"  Base64 data prefix found, data length: {data_len}")
                                    if data_len < 100:
                                        logger.warning(f"  Base64 data seems too short: {data_len} chars")
                                else:
                                    logger.info(f"  URL: {url[:100]}...")
                            elif item.get('type') == 'text':
                                logger.info(f"Content item {i}: text (length: {len(item['text'])})")
                                logger.info(f"Text content: {item['text'][:200]}...")
                else:
                    # Text-only content
                    user_message = {
                        "role": "user",
                        "content": prompt
                    }
                
                messages.append(user_message)
                
                # Determine if streaming is required (mandatory for QVQ-Max)
                use_streaming = self._is_qvq_max_model()
                
                # Prepare completion parameters with optimized settings
                completion_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.model_params["temperature"],
                    "max_tokens": self.model_params["max_tokens"],
                    "stream": use_streaming  # Enable streaming for QVQ-Max, disable for others
                }

                # QVQ-Max specific optimizations
                if self._is_qvq_max_model():
                    # Ensure proper streaming and reasoning for QVQ-Max
                    completion_params["stream"] = True
                    completion_params["max_tokens"] = min(completion_params.get("max_tokens", 2048), 4096)
                    logger.info(f"QVQ-Max parameters: stream={completion_params['stream']}, max_tokens={completion_params['max_tokens']}")
                
                # Add enable_thinking parameter for qwen-vl-plus when using CoT
                if self._should_enable_thinking(use_thinking):
                    completion_params["extra_body"] = {"enable_thinking": True}
                    logger.debug(f"Enabled thinking mode for {self.model_name} with CoT")
                
                # Make async API call with extended timeout
                completion = await asyncio.wait_for(
                    self.client.chat.completions.create(**completion_params),
                    timeout=180.0  # Match the client timeout
                )
                
                # Handle response based on model type
                if use_streaming:
                    # Process QVQ-Max streaming response with reasoning content
                    logger.debug(f"Processing streaming response for {self.model_name}")
                    return await self._process_qvq_stream_response(completion)
                else:
                    # Process standard non-streaming response
                    if completion and completion.choices and len(completion.choices) > 0:
                        content = completion.choices[0].message.content
                        if content:
                            return content.strip()
                        else:
                            raise Exception("Empty content in API response")
                    else:
                        raise Exception("No choices in API response")
            
            except asyncio.TimeoutError:
                error_msg = f"Request timeout for {self.name} (attempt {attempt + 1}/{max_retries})"
                logger.warning(error_msg)
                
                if attempt == max_retries - 1:
                    raise Exception(f"Request timeout after {max_retries} retries")
                
                await self._exponential_backoff_delay(attempt, base_delay)
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if it's a retryable error
                if self._is_retryable_error(error_msg):
                    logger.warning(f"Connection issue with {self.name} (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    
                    if attempt == max_retries - 1:
                        raise Exception(f"Connection error after {max_retries} retries: {error_msg}")
                    
                    # Apply exponential backoff with jitter
                    await self._exponential_backoff_delay(attempt, base_delay)
                else:
                    # Non-retryable error, raise immediately
                    logger.error(f"Non-retryable error generating text with {self.name}: {error_type}: {error_msg}")
                    raise Exception(f"{error_type}: {error_msg}")
        
        # Should never reach here
        raise Exception("Unexpected error in retry loop") 