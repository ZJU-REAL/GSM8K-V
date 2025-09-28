"""
Model Configuration for GSM8K-V
"""

import os
from typing import Dict, List, Any, Optional

from config.model_parameter_manager import get_parameter_manager


class ModelConfig:
    """Base configuration class for models."""
    
    def __init__(self, api_key: Optional[str] = None, skip_key_validation: bool = False):
        self.api_key = api_key or os.environ.get(self.env_var_name, "")
        if not self.api_key and self.require_key and not skip_key_validation:
            raise ValueError(f"API key for {self.name} is required. Set it via {self.env_var_name} environment variable.")
    
    @property
    def name(self) -> str:
        """Model name."""
        raise NotImplementedError
    
    @property
    def env_var_name(self) -> str:
        """Environment variable name for API key."""
        raise NotImplementedError
    
    @property
    def require_key(self) -> bool:
        """Whether this model requires an API key."""
        return True
    
    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters from centralized configuration."""
        try:
            param_manager = get_parameter_manager()
            return param_manager.get_model_parameters(self.name)
        except Exception as e:
            raise RuntimeError(f"Failed to load parameters for model '{self.name}': {e}. "
                             "Please ensure config/model_parameters.json exists and contains "
                             "parameters for this model.")


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, skip_key_validation: bool = False):
        self.model_name = model_name
        super().__init__(api_key, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return "OPENAI_API_KEY"

    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters including model name."""
        base_params = super().get_params()
        base_params["model"] = self.model_name
        return base_params


class ClaudeConfig(ModelConfig):
    """Configuration for Anthropic Claude models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, skip_key_validation: bool = False):
        self.model_name = model_name
        super().__init__(api_key, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return "ANTHROPIC_API_KEY"

    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters including model name."""
        base_params = super().get_params()
        base_params["model"] = self.model_name
        return base_params


class GeminiConfig(ModelConfig):
    """Configuration for Google Gemini models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, skip_key_validation: bool = False):
        self.model_name = model_name
        super().__init__(api_key, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return "GOOGLE_API_KEY"

    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters including model name."""
        base_params = super().get_params()
        base_params["model"] = self.model_name
        return base_params

class QwenConfig(ModelConfig):
    """Configuration for Qwen models via API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, skip_key_validation: bool = False):
        self.model_name = model_name
        super().__init__(api_key, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return "QWEN_API_KEY"

    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters including model name."""
        base_params = super().get_params()
        base_params["model"] = self.model_name
        return base_params

class ZhipuAIConfig(ModelConfig):
    """Configuration for ZhipuAI models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, skip_key_validation: bool = False):
        self.model_name = model_name
        super().__init__(api_key, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return "ZHIPUAI_API_KEY"

    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters including model name."""
        base_params = super().get_params()
        base_params["model"] = self.model_name
        return base_params

class LocalModelConfig(ModelConfig):
    """Configuration for local models like InternVL, LLaVA."""

    def __init__(self, model_name: str, model_path: str, skip_key_validation: bool = False):
        self.model_name = model_name
        self.model_path = model_path
        super().__init__(None, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return ""

    @property
    def require_key(self) -> bool:
        return False


class VLLMConfig(ModelConfig):
    """Configuration for vLLM models served via OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        api_base: str,
        model_path: str = "",
        model_id: str = "",
        skip_key_validation: bool = False
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.model_path = model_path
        self.model_id = model_id or model_name  # Use model_id if provided, otherwise fallback to model_name
        # vLLM typically doesn't require API key for local serving
        super().__init__(None, skip_key_validation)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def env_var_name(self) -> str:
        return ""

    @property
    def require_key(self) -> bool:
        return False

    def get_params(self) -> Dict[str, Any]:
        """Get model-specific parameters including model name."""
        base_params = super().get_params()
        # Use model_id for API requests, but keep model_name for identification
        base_params["model"] = self.model_name
        return base_params


# Configuration registry - stores model specifications without initializing
MODEL_REGISTRY = {
    "gpt-4o": {"type": "openai", "name": "gpt-4o"},
    "gpt-4o-mini": {"type": "openai", "name": "gpt-4o-mini"},
    "gpt-5": {"type": "openai", "name": "gpt-5"},
    "claude-3-7-sonnet": {"type": "claude", "name": "claude-opus-4-20250514"},
    "claude": {"type": "claude", "name": "claude-opus-4-20250514"},
    "gemini-2.0-flash": {"type": "gemini", "name": "gemini-2.0-flash"},
    "gemini": {"type": "gemini", "name": "gemini-2.0-flash"},
    "gemini-2.5-flash-preview": {"type": "gemini", "name": "gemini-2.5-flash-preview-05-20"},
    "gemini-2.5-pro-preview": {"type": "gemini", "name": "gemini-2.5-pro-preview-06-05"},
    "gemini-2.5-pro": {"type": "gemini", "name": "gemini-2.5-pro"},
    "gemini-2.5-flash": {"type": "gemini", "name": "gemini-2.5-flash"},
    "qwen-vl-plus": {"type": "qwen", "name": "qwen-vl-plus-2025-05-07"},
    "qwen-vl-max": {"type": "qwen", "name": "qwen-vl-max"},
    "qwen2.5-vl-72b-instruct": {"type": "qwen", "name": "qwen2.5-vl-72b-instruct"},
    "qwen2.5-vl-7b-instruct": {"type": "qwen", "name": "qwen2.5-vl-7b-instruct"},
    "qvq-max-latest": {"type": "qwen", "name": "qvq-max-latest"},
    "qwen-long-latest": {"type": "qwen", "name": "qwen-long-latest"},
    "qwen2.5-vl-32b-instruct": {"type": "qwen", "name": "qwen2.5-vl-32b-instruct"},
    "qwen2.5-omni-7b": {"type": "qwen", "name": "qwen2.5-omni-7b"},
    "glm-4.5v": {"type": "zhipuai", "name": "glm-4.5v"},
    

    # vLLM models
    "llama-4-17b-128e-instruct": {"type": "vllm", "api_base": "http://localhost:8000/v1", "path": "meta-llama/Llama-4-Maverick-17B-128E-Instruct"},
    "llama-4-17b-16e-instruct": {"type": "vllm", "api_base": "http://localhost:8001/v1", "path": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    "internvl3.5-38b": {"type": "vllm", "api_base": "http://localhost:8002/v1", "path": "OpenGVLab/InternVL3_5-38B-Pretrained"},
    "internvl3.5-8b": {"type": "vllm", "api_base": "http://localhost:8006/v1", "path": "OpenGVLab/InternVL3_5-8B-Pretrained"},
    "internvl3.5-30b-a3b": {"type": "vllm", "api_base": "http://localhost:8003/v1", "path": "OpenGVLab/InternVL3_5-30B-A3B"},
    "internvl3.5-241b-a28b": {"type": "vllm", "api_base": "http://localhost:8004/v1", "path": "OpenGVLab/InternVL3_5-241B-A28B"},
    "Ovis2.5-2B": {"type": "vllm", "api_base": "http://localhost:8000/v1", "path": "AIDC-AI/Ovis2.5-2B"},
    "ovis": {"type": "vllm", "api_base": "http://localhost:8006/v1", "path": "AIDC-AI/Ovis2.5-9B"},
    "minicpm-v-4_5": {"type": "vllm", "api_base": "http://localhost:8007/v1", "path": "OpenBMB/MiniCPM-V-4.5"}
}


def get_model_config(model_name: str, skip_key_validation: bool = False, custom_path: str = None, api_key: Optional[str] = None) -> ModelConfig:
    """
    Factory function to create a model configuration instance.

    Args:
        model_name: Name of the model to create configuration for
        skip_key_validation: Whether to skip API key validation
        custom_path: Custom path for local models (overrides registry path)
        api_key: Optional API key to use instead of environment variable

    Returns:
        A ModelConfig instance for the specified model

    Raises:
        ValueError: If the model name is not found in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(MODEL_REGISTRY.keys())}")
    
    config_spec = MODEL_REGISTRY[model_name]
    
    if config_spec["type"] == "openai":
        return OpenAIConfig(config_spec["name"], api_key=api_key, skip_key_validation=skip_key_validation)
    elif config_spec["type"] == "gemini":
        return GeminiConfig(config_spec["name"], api_key=api_key, skip_key_validation=skip_key_validation)
    elif config_spec["type"] == "qwen":
        return QwenConfig(config_spec["name"], api_key=api_key, skip_key_validation=skip_key_validation)
    elif config_spec["type"] == "zhipuai":
        return ZhipuAIConfig(config_spec["name"], api_key=api_key, skip_key_validation=skip_key_validation)
    elif config_spec["type"] == "vllm":
        # vLLM models use api_base for serving
        api_base = config_spec.get("api_base", "http://localhost:8000/v1")
        model_path = config_spec.get("path", "")
        return VLLMConfig(model_name, api_base, model_path, skip_key_validation=skip_key_validation)
    else:
        raise ValueError(f"Unsupported model type: {config_spec['type']}")


def create_dynamic_vllm_model_config(
    model_name: str,
    api_base: str,
    model_path: str = "",
    model_id: str = "",
    skip_key_validation: bool = False
) -> VLLMConfig:
    """
    Create a vLLM model configuration with custom api_base, model_path and model_id.
    This allows for vLLM models not in the registry or with custom endpoints.

    Args:
        model_name: Name identifier for the model
        api_base: API base URL for the vLLM server
        model_path: Path to the model files (optional)
        model_id: Actual model ID used by vLLM server (optional)
        skip_key_validation: Whether to skip validation

    Returns:
        VLLMConfig instance
    """
    return VLLMConfig(model_name, api_base, model_path, model_id, skip_key_validation=skip_key_validation)

