"""
Image processing utilities for benchmark evaluation.
"""

import asyncio
import os
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional
from PIL import Image


def resize_image(image_data: bytes, max_size: int = 2048) -> bytes:
    """
    Resize an image to a maximum dimension while preserving aspect ratio.
    
    Args:
        image_data: Image data as bytes
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized image data as bytes
    """
    img = Image.open(BytesIO(image_data))
    width, height = img.size
    
    # Resize if needed
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert back to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_image_base64(image_data: bytes) -> str:
    """
    Encode image data as base64 string.

    Args:
        image_data: Image data as bytes

    Returns:
        Base64-encoded image string
    """
    return base64.b64encode(image_data).decode('utf-8')


def resize_image_half(image_data: bytes) -> bytes:
    """
    Resize an image to half its original width and height for maximum token reduction.

    Args:
        image_data: Image data as bytes

    Returns:
        Resized image data as bytes
    """
    img = Image.open(BytesIO(image_data))
    width, height = img.size

    # Resize to eighth the dimensions for maximum token reduction
    new_width = width // 2
    new_height = height // 2

    # Ensure minimum size of 32 pixels for basic usability
    new_width = max(256, new_width)
    new_height = max(256, new_height)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Convert back to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_images_for_model(image_paths: List[str], model_type: str) -> List[Any]:
    """
    Prepare images for a specific model.

    Args:
        image_paths: List of image file paths
        model_type: Type of model (openai, claude, gemini, qwen, local)

    Returns:
        List of processed images in the format required by the model
    """
    processed_images = []

    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            img_data = f.read()

        # Different resize strategies based on model type
        if model_type == "vllm":
            # For vLLM models, use maximum aggressive resizing to reduce token count
            # print(f"[DEBUG] vLLM image processing: {img_path}")
            img_data = resize_image_half(img_data)
            # print(f"[DEBUG] Image resized to 1/8 size for vLLM")
        else:
            # For other models, use standard resizing
            img_data = resize_image(img_data)

        if model_type == "openai":
            processed_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_base64(img_data)}"
                }
            })
        elif model_type == "gemini":
            processed_images.append({
                "mime_type": "image/png",
                "data": img_data
            })
        elif model_type == "qwen":
            # Add Qwen-specific image processing
            processed_images.append({
                "type": "image",
                "data": encode_image_base64(img_data)
            })
        elif model_type == "zhipuai":
            # ZhipuAI uses OpenAI-compatible format
            processed_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_base64(img_data)}"
                }
            })
        elif model_type == "vllm":
            # vLLM models already resized above
            # vLLM uses OpenAI-compatible format for multimodal models
            # Return the final format expected by async_vllm.py
            processed_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_base64(img_data)}"
                }
            })
        elif model_type == "local":
            # For local models, return PIL Image objects directly
            pil_image = Image.open(BytesIO(img_data))
            processed_images.append(pil_image)

    return processed_images


async def prepare_single_image_for_model_async(img_path: str, model_type: str) -> Any:
    """
    Asynchronously prepare a single image for a specific model.

    Args:
        img_path: Path to the image file
        model_type: Type of model (openai, claude, gemini, qwen, local)

    Returns:
        Processed image in the format required by the model
    """
    # Read image file asynchronously
    def read_image():
        with open(img_path, 'rb') as f:
            return f.read()

    # Run file reading in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    img_data = await loop.run_in_executor(None, read_image)

    # Different resize strategies based on model type
    if model_type == "vllm":
        # For vLLM models, use maximum aggressive resizing to reduce token count
        img_data = resize_image_half(img_data)
    else:
        # For other models, use standard resizing
        img_data = resize_image(img_data)

    if model_type == "openai":
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_base64(img_data)}"
            }
        }
    elif model_type == "gemini":
        return {
            "mime_type": "image/png",
            "data": img_data
        }
    elif model_type == "qwen":
        # Add Qwen-specific image processing
        return {
            "type": "image",
            "data": encode_image_base64(img_data)
        }
    elif model_type == "zhipuai":
        # ZhipuAI uses OpenAI-compatible format
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_base64(img_data)}"
            }
        }
    elif model_type == "claude":
        # Claude uses base64 encoded data
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encode_image_base64(img_data)
            }
        }
    elif model_type == "vllm":
        # vLLM models use OpenAI-compatible format for multimodal models
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_base64(img_data)}"
            }
        }
    elif model_type == "local":
        # For local models, return PIL Image objects directly
        pil_image = Image.open(BytesIO(img_data))
        return pil_image
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


async def prepare_images_for_model_async(image_paths: List[str], model_type: str) -> List[Any]:
    """
    Asynchronously prepare images for a specific model.

    Args:
        image_paths: List of image file paths
        model_type: Type of model (openai, claude, gemini, qwen, local)

    Returns:
        List of processed images in the format required by the model
    """
    if not image_paths:
        return []

    # Process all images concurrently
    tasks = [prepare_single_image_for_model_async(img_path, model_type) for img_path in image_paths]
    processed_images = await asyncio.gather(*tasks)

    return list(processed_images)