"""
Evaluation Configuration for GSM8K-V
"""

from enum import Enum
from typing import Dict, List, Any, Optional

from config.data_category_config import DataCategoryConfig


class EvaluationMode(Enum):
    """Evaluation modes."""
    TEXT_ONLY = "text_only"
    VISUAL = "visual"
    SCENE = "scene"


# Removed ScoreWeights - scene evaluation not needed


class EvaluationConfig:
    """Configuration for evaluation settings."""
    
    def __init__(
        self,
        data_path: str = "meta.json",
        image_dir: str = "images",
        results_dir: str = "results",
        modes: Optional[List[EvaluationMode]] = None,
        num_samples: Optional[int] = None,
        seed: Optional[int] = 42,
        category_config: Optional[DataCategoryConfig] = None
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.results_dir = results_dir
        self.modes = modes or list(EvaluationMode)
        self.num_samples = num_samples  # If None, evaluate all samples
        self.seed = seed  # Random seed for reproducibility
        self.category_config = category_config  # Optional category filtering
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {
            "data_path": self.data_path,
            "image_dir": self.image_dir,
            "results_dir": self.results_dir,
            "modes": [mode.value for mode in self.modes],
            "num_samples": self.num_samples,
            "seed": self.seed
        }
        
        # Add category configuration if present
        if self.category_config:
            result["category_config"] = self.category_config.to_dict()
        
        return result


# Default evaluation configuration
DEFAULT_EVAL_CONFIG = EvaluationConfig()