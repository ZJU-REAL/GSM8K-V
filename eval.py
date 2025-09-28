"""
GSM8K-V Evaluation Script

Usage:
1. Config file mode (default): python eval.py
2. Direct model config: python eval.py --type vllm --model_name <name> --api_base <base> --concurrency <int> --image_dir <path>
3. API model config: python eval.py --type api --model_name <name> --api_key <key> --concurrency <int> --image_dir <path>

Examples:
- Default config: python eval.py
- vLLM model: python eval.py --type vllm --model_name minicpm-v-4_5 --api_base http://localhost:8007/v1 --concurrency 32 --image_dir /path/to/images
- API model: python eval.py --type api --model_name gpt-4 --api_key sk-xxx --concurrency 10 --image_dir /path/to/images
"""

import os
import argparse
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm.asyncio import tqdm

from config.model_config import MODEL_REGISTRY
from config.evaluation_config import DEFAULT_EVAL_CONFIG, EvaluationMode
from config.async_model_factory import get_async_models_for_evaluation, filter_available_models
from config.data_category_config import DataCategoryConfig, create_category_config_from_names

from prompts.eval_prompt import get_text_only_math_prompt, get_image_math_prompt

from utils.math_evaluator import MathEvaluator
from utils.async_evaluator import AsyncEvaluationEngine
from utils.data_loader import load_benchmark_data, get_image_paths, get_category_statistics
from utils.image_processor import prepare_images_for_model
from utils.async_result_writer import AsyncResultWriter
from utils.seed import set_global_seed
from utils.retry_decorator import async_retry_on_failure, EvaluationRetryConfig

# Configure logging
def setup_logging():
    """
    Setup logging configuration to save logs in logs directory.
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Configure logging with file in logs directory
    log_file = os.path.join(logs_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Setup logging
logger = setup_logging()


def build_direct_model_config(args) -> Dict[str, Any]:
    """
    Build model configuration from direct command line parameters.

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration dictionary with model settings
    """
    if not args.type or not args.model_name:
        return {}

    config = {}

    # Build model configuration based on type
    if args.type == "vllm":
        if not args.api_base:
            raise ValueError("--api_base is required for vLLM models")

        config["vllm_models"] = {
            args.model_name: {
                "enabled": True,
                "concurrency": args.concurrency,
                "api_base": args.api_base,
                "model_id": args.model_name,  # Use model_name as model_id
                "description": f"{args.model_name} via vLLM"
            }
        }
        config["evaluation_type"] = "vllm"

    elif args.type == "api":
        if not args.api_key:
            raise ValueError("--api_key is required for API models")

        config["api_models"] = {
            args.model_name: {
                "enabled": True,
                "concurrency": args.concurrency,
                "description": f"{args.model_name} API model",
                "api_key": args.api_key
            }
        }
        config["evaluation_type"] = "api"

        # Set API key as environment variable
        os.environ[f"{args.model_name.upper().replace('-', '_')}_API_KEY"] = args.api_key

    return config

def get_config_key_from_model_name(model_name: str) -> str:
    """
    Get the original config key from the actual model name.
    
    Args:
        model_name: The actual model name (e.g., "qwen-vl-plus-2025-05-07")
        
    Returns:
        The original config key (e.g., "qwen-vl-plus")
    """
    # Create reverse mapping from actual names to config keys
    reverse_mapping = {}
    for config_key, model_spec in MODEL_REGISTRY.items():
        # Handle different model spec formats
        if "name" in model_spec:
            # API models use "name" key
            reverse_mapping[model_spec["name"]] = config_key
        elif "path" in model_spec:
            reverse_mapping[config_key] = config_key
    
    # Return the config key if found, otherwise return the model name itself
    return reverse_mapping.get(model_name, model_name)


async def run_async_evaluation(args):
    """
    Run the benchmark evaluation with enhanced error handling and retry logic.
    """
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Set global random seed if specified
        seed = args.seed or config.get("seed")
        set_global_seed(seed)
        
        # Load evaluation configuration
        eval_config = DEFAULT_EVAL_CONFIG
        eval_config.data_path = config.get("data_path", eval_config.data_path)
        eval_config.image_dir = config.get("image_dir", eval_config.image_dir)
        eval_config.results_dir = config.get("results_dir", eval_config.results_dir)
        eval_config.num_samples = args.num_samples or config.get("num_samples", eval_config.num_samples)
        
        # Get evaluation modes
        modes = [EvaluationMode(m) for m in config.get("modes", [m.value for m in eval_config.modes])]
        
        # Get prompt modes for image evaluations
        prompt_modes = config.get("prompt_modes", ["implicit", "explicit"])
        
        # Initialize category configuration if specified
        category_config = None
        if config.get("data_categories") or config.get("data_subcategories"):
            try:
                category_config = create_category_config_from_names(
                    category_names=config.get("data_categories"),
                    subcategory_names=config.get("data_subcategories"),
                    metadata_dir=config.get("metadata_dir", "data/metadata")
                )
                logger.info(f"Category filtering enabled: {category_config.to_dict()}")
            except ValueError as e:
                logger.error(f"Invalid category configuration: {e}")
                raise
        
        # Load data with optional category filtering
        metadata_dir = config.get("metadata_dir", "data/metadata")
        data = load_benchmark_data(
            eval_config.data_path,
            eval_config.num_samples,
            category_config,
            metadata_dir
        )
        
        # Log category statistics if category data is available
        if data and any('data_category' in sample for sample in data):
            category_stats = get_category_statistics(data)
            logger.info(f"Loaded data statistics: {category_stats}")
        
        # Initialize async result writer
        async_result_writer = AsyncResultWriter(eval_config.results_dir)
        
        # Initialize math evaluator without judger model
        math_evaluator = MathEvaluator()
        
        # Initialize async evaluation engine with retry configuration
        retry_config = EvaluationRetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter_factor=0.3
        )
        
        async_engine = AsyncEvaluationEngine(
            math_evaluator, 
            retry_config=retry_config
        )
        
        # Get model configurations from config file with support for model type separation
        evaluation_type = config.get("evaluation_type", "both")  # "api", "vllm", or "both"
        
        
        # Combine API and vLLM models based on evaluation type
        all_models_config = {}
        if evaluation_type in ["api", "both"]:
            all_models_config.update(config.get("api_models", {}))
        if evaluation_type in ["vllm", "both"]:
            all_models_config.update(config.get("vllm_models", {}))
        
        # Fallback to legacy "models" key for backward compatibility
        if not all_models_config:
            all_models_config = config.get("models", {})
            logger.info("Using legacy 'models' configuration")
        
        # Use enabled models from config (no command-line model selection for now)
        model_names_to_evaluate = [
            model_name for model_name, model_config in all_models_config.items()
            if model_config.get("enabled", False)
        ]
        
        logger.info(f"Evaluation type: {evaluation_type}")
        logger.info(f"Models to evaluate: {model_names_to_evaluate}")
        
        if not model_names_to_evaluate:
            logger.error("No models enabled for evaluation. Check configuration.")
            return
        
        # Filter available models for async evaluation
        available_model_names = filter_available_models(model_names_to_evaluate, all_models_config)
        if len(available_model_names) != len(model_names_to_evaluate):
            excluded_models = [m for m in model_names_to_evaluate if m not in available_model_names]
            logger.info(f"Excluded unavailable models from async evaluation: {excluded_models}")

        # Extract API keys from model configurations
        api_keys = {}
        for model_name in available_model_names:
            if model_name in all_models_config:
                model_config = all_models_config[model_name]
                if "api_key" in model_config:
                    api_keys[model_name] = model_config["api_key"]
                    logger.info(f"Using API key from config for model: {model_name}")

        # Initialize async models for evaluation with custom configurations
        async_models = get_async_models_for_evaluation(
            available_model_names,
            skip_key_validation=True,
            model_configs=all_models_config,
            api_keys=api_keys
        )
        
        if not async_models:
            logger.error("No valid async models found for evaluation")
            return
        
        logger.info(f"Starting async evaluation with {len(async_models)} models: {[m.name for m in async_models]}")
        
        # Create evaluation tasks for all models (models run in batch)
        model_evaluation_tasks = []
        for model in async_models:
            # Get concurrency setting for this model using original config key
            config_key = get_config_key_from_model_name(model.name)
            model_config = all_models_config.get(config_key, {})
            concurrency = model_config.get("concurrency", 128)
            
            logger.info(f"Model {model.name} will use concurrency={concurrency}")
            
            task = async_engine.evaluate_single_model_async(
                model,
                data,
                [mode.value for mode in modes],
                prompt_modes,
                eval_config.image_dir,
                concurrency=concurrency,
                save_intermediate_every=50
            )
            model_evaluation_tasks.append((model.name, task))
        
        # Execute all model evaluations in batch with progress tracking
        logger.info("Executing batch model evaluations...")
        
        # Create progress bar for model evaluation
        model_progress = tqdm(
            total=len(model_evaluation_tasks),
            desc="Evaluating models",
            unit="model",
            position=0,
            leave=True
        )
        
        async def evaluate_model_with_progress(model_name, task):
            """Wrapper to update progress bar after each model completes"""
            try:
                result = await task
                model_progress.set_description(f"Completed: {model_name}")
                model_progress.update(1)
                return result
            except Exception as e:
                model_progress.set_description(f"Failed: {model_name}")
                model_progress.update(1)
                raise e
        
        # Execute all evaluations with progress tracking
        all_results = await asyncio.gather(
            *[evaluate_model_with_progress(model_name, task) for model_name, task in model_evaluation_tasks], 
            return_exceptions=True
        )
        
        model_progress.close()
        
        # Process results and handle exceptions
        all_model_results = {}
        for i, ((model_name, _), result) in enumerate(zip(model_evaluation_tasks, all_results)):
            if isinstance(result, Exception):
                logger.error(f"Exception in evaluation for {model_name}: {result}")
                all_model_results[model_name] = {"error": str(result)}
            else:
                all_model_results[model_name] = result
                logger.info(f"Completed evaluation for {model_name}")
        
        # Save all results concurrently (each model in its own file)
        logger.info("Saving results for all models...")
        config_summary = {
            "data_path": eval_config.data_path,
            "image_dir": eval_config.image_dir,
            "results_dir": eval_config.results_dir,
            "num_samples": eval_config.num_samples,
            "modes": [mode.value for mode in modes],
            "prompt_modes": prompt_modes,
            "seed": seed,
            "models_evaluated": list(all_model_results.keys()),
            "evaluation_mode": "async_batch",
            "model_concurrency_settings": {
                model_name: all_models_config.get(get_config_key_from_model_name(model_name), {}).get("concurrency", 1)
                for model_name in all_model_results.keys()
            },
            "evaluation_type": evaluation_type
        }
        
        saved_files = await async_result_writer.save_all_model_results_async(
            all_model_results,
            config_summary
        )
        
        logger.info(f"Async evaluation completed successfully. Results saved to {len(saved_files)} files:")
        for file_path in saved_files:
            logger.info(f"  - {file_path}")
        
    except Exception as e:
        logger.error(f"Error in enhanced async evaluation: {e}")
        raise


# Removed synchronous run_evaluation - only async evaluation is supported


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run multimodal model benchmark evaluation",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)

    # Configuration mode - mutually exclusive group
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", type=str, default="config.json",
                             help="Path to configuration file (default: config.json)")

    # Direct model configuration parameters
    parser.add_argument("--type", choices=["api", "vllm"],
                        help="Model type for direct configuration")
    parser.add_argument("--model_name", type=str,
                        help="Model name for direct configuration")
    parser.add_argument("--api_base", type=str,
                        help="API base URL for vLLM models")
    parser.add_argument("--api_key", type=str,
                        help="API key for API models")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrency level for the model (default: 10)")

    # Core evaluation parameters
    parser.add_argument("--image_dir", type=str,
                        help="Directory containing evaluation images")
    parser.add_argument("--data_path", type=str,
                        help="Path to evaluation data file")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results (default: results)")
    parser.add_argument("--modes", nargs="+", choices=["text_only", "visual", "scene"],
                        help="Evaluation modes to run")
    parser.add_argument("--prompt_modes", nargs="+", choices=["implicit", "explicit"],
                        help="Prompt modes for image evaluations")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-samples", type=int,
                        help="Number of samples to evaluate")

    # Data category filtering arguments
    parser.add_argument("--data-categories", nargs="+",
                        choices=["measurement", "physical_metric", "ratio_percentage",
                                "signboard_and_icon", "temporal", "other"],
                        help="Specific data categories to evaluate")
    parser.add_argument("--data-subcategories", nargs="+",
                        choices=["distance", "length_area_volume", "speed", "weight",
                                "graph", "statistics", "group", "price",
                                "calendar_age", "clock", "count", "dialogue", "label"],
                        help="Specific data subcategories to evaluate")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if direct model configuration is provided
    direct_config = build_direct_model_config(args)

    if direct_config:
        # Use direct model configuration
        logger.info(f"Using direct model configuration: {args.type} model {args.model_name}")

        # Load base config and merge with direct config
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Override with direct model config
        config.update(direct_config)

        # Override other parameters if provided
        if args.image_dir:
            config["image_dir"] = args.image_dir
        if args.data_path:
            config["data_path"] = args.data_path
        if args.results_dir:
            config["results_dir"] = args.results_dir
        if args.modes:
            config["modes"] = args.modes
        if args.prompt_modes:
            config["prompt_modes"] = args.prompt_modes
        if args.seed is not None:
            config["seed"] = args.seed
        if args.num_samples:
            config["num_samples"] = args.num_samples

        # Override category settings if provided
        if args.data_categories:
            config["data_categories"] = args.data_categories
        if args.data_subcategories:
            config["data_subcategories"] = args.data_subcategories

        # Save merged config temporarily
        temp_config_path = "temp_direct_config.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Update args to use temporary config
        args.config = temp_config_path

        # Clean up temporary config after evaluation
        def cleanup_temp_config():
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        import atexit
        atexit.register(cleanup_temp_config)

    else:
        # Use standard config file mode with overrides
        if args.data_categories or args.data_subcategories or args.image_dir or args.data_path or args.results_dir or args.modes or args.prompt_modes or args.seed is not None or args.num_samples:
            # Load existing config
            with open(args.config, 'r') as f:
                config = json.load(f)

            # Override settings
            if args.image_dir:
                config["image_dir"] = args.image_dir
            if args.data_path:
                config["data_path"] = args.data_path
            if args.results_dir:
                config["results_dir"] = args.results_dir
            if args.modes:
                config["modes"] = args.modes
            if args.prompt_modes:
                config["prompt_modes"] = args.prompt_modes
            if args.seed is not None:
                config["seed"] = args.seed
            if args.num_samples:
                config["num_samples"] = args.num_samples
            if args.data_categories:
                config["data_categories"] = args.data_categories
            if args.data_subcategories:
                config["data_subcategories"] = args.data_subcategories

            # Save temporary config
            temp_config_path = "temp_override_config.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Update args to use temporary config
            args.config = temp_config_path

            # Clean up temporary config after evaluation
            def cleanup_temp_config():
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

            import atexit
            atexit.register(cleanup_temp_config)

    # Always use async evaluation
    logger.info("Running async evaluation")

    # Debug: Print evaluation modes for verification
    if hasattr(args, 'modes') and args.modes:
        logger.info(f"Requested modes: {args.modes}")

    asyncio.run(run_async_evaluation(args))