import asyncio
import logging
import random
from typing import Any, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class EvaluationRetryConfig:
    """Configuration for evaluation retry logic"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter_factor: float = 0.3
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter_factor = jitter_factor

def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry based on error type and message.
    
    Args:
        error: Exception that occurred
        
    Returns:
        True if error is retryable, False otherwise
    """
    error_msg = str(error).lower()
    
    # Network and connection related errors
    retryable_patterns = [
        'connection', 'network', 'timeout', 'refused', 'reset',
        'disconnected', 'unavailable', 'server error',
        'rate limit', 'throttling', 'quota exceeded',
        'temporary', 'service unavailable', 'gateway',
        '429', '500', '502', '503', '504', '520', '521', '522', '523', '524'
    ]
    
    # Check if any retryable pattern is in the error message
    is_retryable = any(pattern in error_msg for pattern in retryable_patterns)
    
    # Log the decision for debugging
    if is_retryable:
        logger.debug(f"Retryable error detected: {error}")
    else:
        logger.debug(f"Non-retryable error: {error}")
    
    return is_retryable

async def apply_exponential_backoff(
    attempt: int, 
    config: EvaluationRetryConfig
) -> None:
    """
    Apply exponential backoff with jitter for retry delays.
    
    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration
    """
    if attempt == 0:
        return
    
    # Calculate exponential delay
    delay = min(
        config.base_delay * (config.exponential_base ** (attempt - 1)),
        config.max_delay
    )
    
    # Add jitter to prevent thundering herd
    jitter = random.uniform(-config.jitter_factor, config.jitter_factor) * delay
    final_delay = max(0, delay + jitter)
    
    logger.info(f"Applying retry backoff: {final_delay:.2f}s (attempt {attempt})")
    await asyncio.sleep(final_delay)

def async_retry_on_failure(config: Optional[EvaluationRetryConfig] = None):
    """
    Decorator for async functions to add retry logic with exponential backoff.
    
    Args:
        config: Retry configuration, uses default if None
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = EvaluationRetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    # Apply backoff delay before retry (not on first attempt)
                    await apply_exponential_backoff(attempt, config)
                    
                    # Execute the function
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if attempt < config.max_retries and is_retryable_error(e):
                        logger.warning(
                            f"Retryable error in {func.__name__} "
                            f"(attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                        )
                        continue
                    else:
                        # Either non-retryable error or max retries reached
                        if attempt >= config.max_retries:
                            logger.error(
                                f"Max retries ({config.max_retries}) reached for {func.__name__}: {e}"
                            )
                        else:
                            logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator