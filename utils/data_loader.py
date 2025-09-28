import json
import os
import logging
from typing import Dict, List, Any, Optional, Set

from config.data_category_config import DataCategoryConfig, DataSubCategory

logger = logging.getLogger(__name__)


def _build_question_category_map(metadata_dir: str = "data/metadata") -> Dict[str, tuple]:
    """
    Build a mapping from question_id to (category, subcategory) based on metadata files.

    Returns:
        Dictionary mapping question_id to (category, subcategory) tuples
    """
    category_map = {}
    config = DataCategoryConfig(metadata_dir=metadata_dir)

    for subcategory, relative_path in config.SUBCATEGORY_FILE_MAP.items():
        file_path = os.path.join(metadata_dir, relative_path)
        if not os.path.exists(file_path):
            logger.warning(f"Category file not found: {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                category_data = json.load(f)

            category = config._get_parent_category(subcategory).value
            subcategory_name = subcategory.value

            for sample in category_data:
                question_id = sample.get('question_id')
                if question_id:
                    category_map[question_id] = (category, subcategory_name)

        except Exception as e:
            logger.warning(f"Failed to load category mapping from {file_path}: {e}")
            continue

    return category_map


def load_benchmark_data(
    file_path: str,
    num_samples: Optional[int] = None,
    category_config: Optional[DataCategoryConfig] = None,
    metadata_dir: str = "data/metadata"
) -> List[Dict[str, Any]]:
    """
    Load benchmark data from JSON file with optional category filtering.
    
    Args:
        file_path: Path to the JSON data file
        num_samples: Optional number of samples to load (for testing)
        category_config: Optional configuration for category-based filtering
        
    Returns:
        List of benchmark examples
    """
    logger.info(f"Loading benchmark data from {file_path}")
    
    # If category filtering is enabled, load from category-specific files
    if category_config is not None:
        return load_category_filtered_data(category_config, num_samples)
    
    # Default behavior: load from single file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if num_samples is not None:
            data = data[:num_samples]
            logger.info(f"Loaded {len(data)} samples (limited by config)")
        else:
            logger.info(f"Loaded {len(data)} samples")

        # Add category information to samples
        category_map = _build_question_category_map(metadata_dir)
        for sample in data:
            question_id = sample.get('question_id')
            if question_id and question_id in category_map:
                category, subcategory = category_map[question_id]
                sample['data_category'] = category
                sample['data_subcategory'] = subcategory
            else:
                sample['data_category'] = 'unknown'
                sample['data_subcategory'] = 'unknown'

        return data
    except Exception as e:
        logger.error(f"Failed to load benchmark data: {e}")
        raise


def load_category_filtered_data(
    category_config: DataCategoryConfig,
    num_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load benchmark data filtered by specific categories.
    
    Args:
        category_config: Configuration specifying which categories to load
        num_samples: Optional number of samples to load per category
        
    Returns:
        List of benchmark examples from specified categories
    """
    logger.info("Loading category-filtered benchmark data")
    
    all_data = []
    enabled_files = category_config.get_enabled_data_files()
    
    if not enabled_files:
        logger.warning("No enabled data files found in category configuration")
        return []
    
    for subcategory, file_path in enabled_files.items():
        try:
            logger.info(f"Loading data from {subcategory.value}: {file_path}")
            
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found: {file_path}")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                category_data = json.load(f)
            
            # Add category metadata to each sample
            for sample in category_data:
                sample['data_category'] = category_config._get_parent_category(subcategory).value
                sample['data_subcategory'] = subcategory.value
            
            # Apply sample limit per category if specified
            if num_samples is not None:
                category_data = category_data[:num_samples]
            
            all_data.extend(category_data)
            logger.info(f"Loaded {len(category_data)} samples from {subcategory.value}")
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            continue
    
    # Shuffle data to mix categories
    import random
    random.shuffle(all_data)
    
    logger.info(f"Total loaded samples across all categories: {len(all_data)}")
    return all_data


def get_category_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about data categories in the loaded dataset.
    
    Args:
        data: List of loaded benchmark examples
        
    Returns:
        Dictionary containing category distribution statistics
    """
    if not data:
        return {}
    
    # Count samples by category and subcategory
    category_counts = {}
    subcategory_counts = {}
    
    for sample in data:
        category = sample.get('data_category', 'unknown')
        subcategory = sample.get('data_subcategory', 'unknown')
        
        category_counts[category] = category_counts.get(category, 0) + 1
        subcategory_counts[subcategory] = subcategory_counts.get(subcategory, 0) + 1
    
    total_samples = len(data)
    
    statistics = {
        'total_samples': total_samples,
        'category_distribution': {
            cat: {
                'count': count,
                'percentage': round(count / total_samples * 100, 2)
            }
            for cat, count in category_counts.items()
        },
        'subcategory_distribution': {
            subcat: {
                'count': count,
                'percentage': round(count / total_samples * 100, 2)
            }
            for subcat, count in subcategory_counts.items()
        }
    }
    
    return statistics


def load_image(image_path: str) -> bytes:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image data as bytes
    """
    try:
        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def get_image_paths(question_data: Dict[str, Any], image_dir: str, mode: str) -> List[str]:
    """
    Get image paths for a question based on the evaluation mode.
    
    Args:
        question_data: Question data from the benchmark
        image_dir: Directory containing images
        mode: Evaluation mode (single_scene or multi_scene)
        
    Returns:
        List of image paths
    """
    if mode == "text_only":
        return []
    
    pic_ids = question_data.get("pic_ids", [])
    if not pic_ids:
        logger.warning(f"No pic_ids found for question {question_data.get('question_id', 'unknown')}")
        return []
    
    if mode == "single_scene":
        # For single scene mode, use question_id.png format
        question_id = question_data.get("question_id", "")
        image_path = os.path.join(image_dir, f"{question_id}")
        return [image_path] if os.path.exists(image_path) else []
    
    elif mode == "multi_scene":
        image_paths = [os.path.join(image_dir, f"{pic_id}") for pic_id in pic_ids]
    else:
        # Fallback for any other modes
        image_paths = [os.path.join(image_dir, f"{pic_id}") for pic_id in pic_ids]
    
    # Verify images exist
    valid_paths = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            valid_paths.append(img_path)
        else:
            logger.warning(f"Image not found: {img_path}")
    
    return valid_paths