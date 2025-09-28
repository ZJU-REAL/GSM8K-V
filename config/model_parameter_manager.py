"""
Model Parameter Manager - Centralized configuration for all model parameters.
Manages temperature and max_tokens settings for all supported models.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelParameterManager:
    """Centralized manager for model parameters (temperature, max_tokens, etc.)"""

    _instance = None
    _parameters = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._parameters is None:
            self._load_parameters()

    def _load_parameters(self):
        """Load parameters from the configuration file"""
        config_file = os.path.join(os.path.dirname(__file__), "model_parameters.json")

        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Model parameters configuration file not found: {config_file}. "
                "Please ensure config/model_parameters.json exists."
            )

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._parameters = json.load(f)
            logger.info(f"Loaded model parameters from {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model parameters file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model parameters: {e}")

    def get_model_parameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary containing temperature and max_tokens

        Raises:
            ValueError: If model parameters are not found
        """
        if not self._parameters:
            raise RuntimeError("Model parameters not loaded")

        # Try to find exact model match first
        if model_name in self._parameters.get("models", {}):
            params = self._parameters["models"][model_name].copy()
            logger.debug(f"Found exact parameters for model: {model_name}")
            return params

        # Try to find by model type (for dynamic models)
        model_type = self._detect_model_type(model_name)
        if model_type and model_type in self._parameters.get("model_type_defaults", {}):
            params = self._parameters["model_type_defaults"][model_type].copy()
            logger.debug(f"Using {model_type} defaults for model: {model_name}")
            return params

        # Fall back to global defaults
        if "model_defaults" in self._parameters:
            params = self._parameters["model_defaults"].copy()
            logger.warning(f"Using global defaults for unknown model: {model_name}")
            return params

        raise ValueError(
            f"No parameters found for model '{model_name}' and no defaults available. "
            f"Please add model parameters to config/model_parameters.json"
        )

    def _detect_model_type(self, model_name: str) -> Optional[str]:
        """Detect model type from model name"""
        model_name_lower = model_name.lower()

        if "gpt" in model_name_lower:
            return "openai"
        elif "claude" in model_name_lower:
            return "claude"
        elif "gemini" in model_name_lower:
            return "gemini"
        elif ("qwen" in model_name_lower or "qvq" in model_name_lower) and "local" not in model_name_lower:
            return "qwen"
        elif "glm" in model_name_lower or "zhipuai" in model_name_lower:
            return "zhipuai"
        elif any(keyword in model_name_lower for keyword in ["vllm", "server", "api"]):
            return "vllm"

        return None

    def get_all_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all models with their parameters"""
        if not self._parameters:
            raise RuntimeError("Model parameters not loaded")

        return self._parameters.get("models", {})

    def validate_model_parameters(self, model_name: str, temperature: Optional[float] = None,
                                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate and get model parameters, allowing optional overrides for validation.

        Args:
            model_name: Name of the model
            temperature: Optional temperature override (for validation only)
            max_tokens: Optional max_tokens override (for validation only)

        Returns:
            Validated parameters dictionary
        """
        params = self.get_model_parameters(model_name)

        # Validate temperature range
        temp = temperature if temperature is not None else params.get("temperature", 0.2)
        if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 2.0):
            raise ValueError(f"Invalid temperature value: {temp}. Must be between 0.0 and 2.0")

        # Validate max_tokens range
        tokens = max_tokens if max_tokens is not None else params.get("max_tokens", 2048)
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"Invalid max_tokens value: {tokens}. Must be a positive integer")

        validated_params = {
            "temperature": temp,
            "max_tokens": tokens
        }

        # Add any additional parameters from the config
        for key, value in params.items():
            if key not in validated_params:
                validated_params[key] = value

        return validated_params

    def reload_parameters(self):
        """Reload parameters from the configuration file"""
        self._parameters = None
        self._load_parameters()
        logger.info("Reloaded model parameters from configuration file")


# Global instance for easy access
_parameter_manager = None

def get_parameter_manager() -> ModelParameterManager:
    """Get the global parameter manager instance"""
    global _parameter_manager
    if _parameter_manager is None:
        _parameter_manager = ModelParameterManager()
    return _parameter_manager
