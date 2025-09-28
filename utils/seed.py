import random
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: The random seed to use
    """
    if seed is None:
        logger.info("No random seed specified, using random initialization")
        return
    
    logger.info(f"Setting global random seed to {seed}")
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set environment variable for potential subprocesses
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("PyTorch random seed set")
    except ImportError:
        logger.debug("PyTorch not available, skipping its seed setting")
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info("TensorFlow random seed set")
    except ImportError:
        logger.debug("TensorFlow not available, skipping its seed setting")
    
    logger.info("Global random seed setting completed")