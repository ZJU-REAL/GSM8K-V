"""
Asynchronous Model Base Interface for GSM8K-V

This module defines the abstract base class AsyncModelInterface that provides a unified asynchronous interface for all AI models used in GSM8K-V evaluation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class AsyncModelInterface(ABC):
    """Base async interface for all models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        logger.info(f"Initializing async model: {self.name}")
    
    @abstractmethod
    async def generate_text_async(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """
        Generate text from the model asynchronously.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            
        Returns:
            Generated text
        """
        pass
    
    def generate_text(self, prompt: str, images: Optional[List[Any]] = None) -> str:
        """
        Synchronous wrapper for async text generation.
        
        Args:
            prompt: Text prompt
            images: Optional list of images
            
        Returns:
            Generated text
        """
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate_text_async(prompt, images))
    
    async def process_response_async(self, response: Any) -> str:
        """
        Process the model's raw response asynchronously.
        
        Args:
            response: Raw response from the model
            
        Returns:
            Processed text response
        """
        # Default implementation, override if needed
        return str(response)
    
    def __str__(self) -> str:
        return self.name 