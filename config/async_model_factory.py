import logging
from typing import Dict, List, Any, Optional

from config.model_config import (
    get_model_config,
    MODEL_REGISTRY,
    OpenAIConfig,
    GeminiConfig,
    QwenConfig,
    ZhipuAIConfig,
    LocalModelConfig,
    VLLMConfig
)
from models.async_base import AsyncModelInterface

logger = logging.getLogger(__name__)


def initialize_async_model(model_config) -> AsyncModelInterface:
    """
    Initialize an async model based on its configuration.
    Imports are done at runtime to avoid loading unnecessary dependencies.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Initialized async model interface
    """
    try:
        if isinstance(model_config, OpenAIConfig):
            from models.async_gpt import AsyncGPTModel
            return AsyncGPTModel(model_config)
        elif isinstance(model_config, GeminiConfig):
            from models.async_gemini import AsyncGeminiModel
            return AsyncGeminiModel(model_config)
        elif isinstance(model_config, QwenConfig):
            from models.async_qwen import AsyncQwenModel
            return AsyncQwenModel(model_config)
        elif isinstance(model_config, ZhipuAIConfig):
            from models.async_zhipuai import AsyncZhipuAIModel
            return AsyncZhipuAIModel(model_config)
        elif isinstance(model_config, VLLMConfig):
            from models.async_vllm import AsyncVLLMModel
            return AsyncVLLMModel(model_config)
        else:
            raise ValueError(f"Unknown model type: {type(model_config)}")
    except ImportError as e:
        logger.error(f"Failed to initialize async {model_config.name} due to missing dependencies: {e}")
        raise


def get_async_models_for_evaluation(
    model_names: Optional[List[str]] = None,
    skip_key_validation: bool = False,
    model_configs: Optional[Dict[str, Dict]] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> List[AsyncModelInterface]:
    """
    Get list of async models for evaluation with support for custom configurations.

    Args:
        model_names: Optional list of specific model names to evaluate
        skip_key_validation: Whether to skip API key validation
        model_configs: Optional dictionary of model configurations with custom paths
        api_keys: Optional dictionary mapping model names to API keys

    Returns:
        List of initialized async model interfaces
    """
    async_models = []
    
    if model_names:
        # Only evaluate specified models
        for model_name in model_names:
            try:
                # Check if custom configuration is provided in model_configs
                custom_config = None
                if model_configs and model_name in model_configs:
                    custom_config = model_configs[model_name]

                # Handle vLLM models with custom api_base
                custom_path = custom_config.get("model_path") if custom_config else None
                if custom_config and custom_config.get("api_base"):
                    # Create dynamic vLLM config
                    from config.model_config import create_dynamic_vllm_model_config
                    model_config = create_dynamic_vllm_model_config(
                        model_name,
                        custom_config["api_base"],
                        custom_config.get("model_path", ""),  # For local path if needed
                        custom_config.get("model_id", model_name),  # Use model_id if available, otherwise model_name
                        skip_key_validation
                    )
                elif custom_config and custom_config.get("model_path") and model_name not in MODEL_REGISTRY:
                    # Create dynamic local config for models not in registry
                    from config.model_config import create_dynamic_local_model_config
                    model_config = create_dynamic_local_model_config(
                        model_name, custom_config["model_path"], skip_key_validation
                    )
                else:
                    # Use standard registry with API key if provided
                    api_key = api_keys.get(model_name) if api_keys else None
                    model_config = get_model_config(
                        model_name,
                        skip_key_validation=skip_key_validation,
                        custom_path=custom_path,
                        api_key=api_key
                    )

                async_model = initialize_async_model(model_config)
                if async_model is not None:
                    async_models.append(async_model)
                    logger.info(f"Initialized async model for evaluation: {model_name}")
                    if custom_path:
                        logger.info(f"Using custom path for {model_name}: {custom_path}")
            except Exception as e:
                logger.error(f"Failed to initialize async model {model_name}: {e}")
    else:
        for model_name in MODEL_REGISTRY:
            try:
                model_config = get_model_config(model_name, skip_key_validation=skip_key_validation)
                async_model = initialize_async_model(model_config)
                if async_model is not None:
                    async_models.append(async_model)
                    logger.info(f"Initialized async model for evaluation: {model_name}")
            except Exception as e:
                logger.warning(f"Skipping async model {model_name}: {e}")
    
    return async_models


def filter_available_models(model_names: List[str], model_configs: Optional[Dict[str, Dict]] = None) -> List[str]:
    """
    Filter model names to only include those available in the registry or have custom configurations.

    Args:
        model_names: List of model names
        model_configs: Optional dictionary of model configurations (used for vLLM models)

    Returns:
        List of available model names (both API and local models)
    """
    available_models = []
    for model_name in model_names:
        # Check if model is in registry
        if model_name in MODEL_REGISTRY:
            available_models.append(model_name)
            logger.info(f"Model {model_name} is available for async evaluation")
        # Check if model has custom configuration (for vLLM models)
        elif model_configs and model_name in model_configs:
            available_models.append(model_name)
            logger.info(f"Model {model_name} has custom configuration, available for async evaluation")
        else:
            logger.warning(f"Unknown model {model_name}, skipping")

    return available_models 