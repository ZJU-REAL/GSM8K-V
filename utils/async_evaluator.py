import asyncio
import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm.asyncio import tqdm

from models.async_base import AsyncModelInterface
from utils.math_evaluator import MathEvaluator
from utils.data_loader import get_image_paths
from utils.image_processor import prepare_images_for_model, prepare_images_for_model_async
from utils.retry_decorator import async_retry_on_failure, EvaluationRetryConfig
from prompts.eval_prompt import get_text_only_math_prompt, get_image_math_prompt, get_scene_math_prompt

logger = logging.getLogger(__name__)


class AsyncEvaluationEngine:
    """Engine for running async parallel evaluations."""
    
    def __init__(
        self, 
        math_evaluator: MathEvaluator, 
        retry_config: Optional[EvaluationRetryConfig] = None
    ):
        """
        Initialize the async evaluation engine.
        
        Args:
            math_evaluator: Math evaluator instance
            retry_config: Configuration for retry logic
        """
        self.math_evaluator = math_evaluator
        self.retry_config = retry_config or EvaluationRetryConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0
        )
    
    @async_retry_on_failure()
    async def _evaluate_math_with_retry(
        self,
        model: AsyncModelInterface,
        question_data: Dict[str, Any],
        mode: str,
        image_dir: str,
        prompt_mode: str
    ) -> Dict[str, Any]:
        """
        Core math evaluation logic with automatic retry capability.
        
        This method is wrapped with retry decorator to handle transient failures.
        """
        question_id = question_data["question_id"]
        original_question = question_data.get("original_question", "")
        modify_scene_related_question = question_data.get("modify_scene_related_question", "")
        ground_truth = question_data["math_ground_truth"]
        
        # Prepare inputs based on mode
        if mode == "text_only":
            prompt = get_text_only_math_prompt(original_question)
            images = None
        elif mode == "scene":
            # For scene mode, use the scene description directly
            prompt = get_scene_math_prompt(original_question)
            images = None  # Scene mode typically doesn't use images directly
        else:
            # For image modes, determine which question to use based on prompt_mode
            if prompt_mode == "explicit":
                question_to_use = modify_scene_related_question
            else:  # implicit mode
                question_to_use = ""
            
            prompt = get_image_math_prompt(question_to_use, prompt_mode)
            image_paths = get_image_paths(question_data, image_dir, mode)
            
            if not image_paths:
                raise ValueError(f"No images found for question {question_id} in {mode} mode")
            
            # Prepare images for model
            # Check if it's an API model or local model
            if "gpt" in model.name.lower():
                model_type = "openai"
            elif "claude" in model.name.lower():
                model_type = "claude"
            elif "gemini" in model.name.lower():
                model_type = "gemini"
            elif "glm" in model.name.lower() or "zhipuai" in model.name.lower():
                # ZhipuAI models (GLM series)
                model_type = "zhipuai"
            elif ("qwen" in model.name.lower() or "qvq" in model.name.lower()) and "local" not in model.name.lower():
                # API Qwen models (including QVQ-Max)
                model_type = "qwen"
            elif hasattr(model, 'api_base') and model.api_base:
                # VLLM models have api_base attribute
                model_type = "vllm"
            else:
                # Local models (including local Qwen models)
                model_type = "vllm"

            logger.info(f"Using model_type '{model_type}' for model '{model.name}'")
            images = await prepare_images_for_model_async(image_paths, model_type)
        
        # Get model response asynchronously
        if 'qwen' in model.name.lower():
            # For Qwen models, pass use_thinking parameter
            use_thinking = (model.name == "qwen-vl-plus")
            model_response = await model.generate_text_async(prompt, images, use_thinking=use_thinking)
        else:
            # For other models, use standard interface
            model_response = await model.generate_text_async(prompt, images)
        
        if not model_response or not model_response.strip():
            raise ValueError(f"Empty response from model {model.name}")
        
        return {
            "model_response": model_response,
            "question_id": question_id,
            "mode": mode,
            "prompt_mode": prompt_mode,
            "ground_truth": ground_truth
        }
    
    async def evaluate_math_problem_async(
        self,
        model: AsyncModelInterface,
        question_data: Dict[str, Any],
        mode: str,
        image_dir: str,
        prompt_mode: str = "implicit"
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a math problem with comprehensive error handling.
        
        This method ensures that network errors and transient failures
        do not contaminate the accuracy calculations.
        """
        question_id = question_data["question_id"]
        ground_truth = question_data["math_ground_truth"]
        
        logger.info(f"Evaluating {model.name} on question {question_id} ({mode}, {prompt_mode})")
        
        result = {
            "question_id": question_id,
            "mode": mode,
            "prompt_mode": prompt_mode,
            "model": model.name,
            "math_ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat(),
            "data_category": question_data.get("data_category", "unknown"),
            "data_subcategory": question_data.get("data_subcategory", "unknown")
        }
        
        try:
            # Attempt evaluation with retry logic
            eval_result = await self._evaluate_math_with_retry(
                model, question_data, mode, image_dir, prompt_mode
            )
            
            # Extract response and evaluate
            model_response = eval_result["model_response"]
            result["model_response"] = model_response
            
            # Extract answer (first stage only, no post-processing)
            extracted_answer = self.math_evaluator.extract_answer(model_response)
            result["extracted_answer"] = extracted_answer
            
            # Check correctness
            is_correct = self.math_evaluator.is_correct(extracted_answer, ground_truth)
            result["math_correct"] = is_correct
            
            logger.info(
                f"Math evaluation completed: {model.name}, {question_id}, "
                f"{mode}, {prompt_mode}, correct={is_correct}"
            )
            
        except Exception as e:
            # Log the error but mark this evaluation as failed
            logger.error(
                f"Failed to evaluate question {question_id} with {model.name} "
                f"after {self.retry_config.max_retries} retries: {e}"
            )
            
            result["error"] = str(e)
            result["evaluation_failed"] = True
            # Do NOT set math_correct to False - this will be excluded from accuracy calculation
            
        return result

    async def _save_intermediate_results_async(
        self,
        model_name: str,
        results: List[Dict[str, Any]],
        results_dir: str,
        suffix: str
    ) -> None:
        """
        Save intermediate evaluation results asynchronously.

        Args:
            model_name: Name of the model
            results: List of evaluation results to save
            results_dir: Directory to save results
            suffix: Suffix for the filename
        """
        try:
            # Create intermediate results structure
            intermediate_results = {
                model_name: {
                    "math_results": results
                }
            }

            # Ensure results directory exists
            os.makedirs(results_dir, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"intermediate_{model_name}_{suffix}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(intermediate_results, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved intermediate results to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    # Removed scene prediction evaluation - not needed

    async def evaluate_single_model_async(
        self,
        model: AsyncModelInterface,
        data: List[Dict[str, Any]],
        modes: List[str],
        prompt_modes: List[str],
        image_dir: str,
        concurrency: int = 128,
        save_intermediate_every: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate a single model across all specified modes asynchronously with concurrency control.
        
        Args:
            model: Async model interface
            data: List of questions to evaluate
            modes: List of evaluation modes
            prompt_modes: List of prompt modes for image evaluations
            image_dir: Directory with images
            concurrency: Maximum number of concurrent API calls for this model
            
        Returns:
            Dictionary containing all results for this model
        """
        logger.info(f"Starting async evaluation for model: {model.name} with concurrency={concurrency}")
        model_results = {}
        
        # Note: Concurrency control is now handled within each evaluation function using semaphores
        
        for mode in modes:
            if mode == "text_only":
                # Text-only mode - single evaluation
                key = f"{mode}"

                # Execute with true async concurrency using semaphore
                logger.info(f"Processing {len(data)} {mode} questions for {model.name} with max {concurrency} concurrent requests")

                # Create progress bar for this model's evaluation
                progress_bar = tqdm(
                    total=len(data),
                    desc=f"{model.name} {mode}",
                    unit="question",
                    position=1,
                    leave=False
                )

                # Execute with true async concurrency using semaphore
                math_results = []
                semaphore = asyncio.Semaphore(concurrency)

                # Track concurrent task count for debugging
                active_tasks = 0
                max_concurrent_seen = 0

                async def evaluate_single_question(question_data):
                    """Evaluate a single question with semaphore control"""
                    nonlocal active_tasks, max_concurrent_seen
                    active_tasks += 1
                    max_concurrent_seen = max(max_concurrent_seen, active_tasks)

                    try:
                        async with semaphore:
                            result = await self.evaluate_math_problem_async(
                                model, question_data, mode, image_dir, "text_only"
                            )
                            return result
                    except Exception as e:
                        logger.error(f"Error evaluating question {question_data['question_id']}: {e}")
                        # Return error result instead of raising exception
                        return {
                            "question_id": question_data["question_id"],
                            "mode": mode,
                            "model": model.name,
                            "error": str(e),
                            "evaluation_failed": True
                        }
                    finally:
                        active_tasks -= 1

                # Create all tasks immediately - true async concurrency
                all_tasks = []
                for question_data in data:
                    task = asyncio.create_task(evaluate_single_question(question_data))
                    all_tasks.append(task)

                logger.info(f"Created {len(all_tasks)} concurrent tasks with semaphore limit {concurrency}")

                # Process completed tasks as they finish without blocking
                completed_count = 0
                for completed_task in asyncio.as_completed(all_tasks):
                    result = await completed_task
                    math_results.append(result)
                    completed_count += 1

                    # Update progress bar (non-blocking)
                    progress_bar.update(1)

                    # Save intermediate results every save_intermediate_every questions (non-blocking)
                    if completed_count % save_intermediate_every == 0:
                        processed_results = [r for r in math_results[-save_intermediate_every:] if not isinstance(r, dict) or not r.get("evaluation_failed", False)]
                        if processed_results:
                            # Create background task for saving intermediate results
                            asyncio.create_task(self._save_intermediate_results_async(
                                model.name, processed_results,
                                "intermediate_results", f"{key}_intermediate"
                            ))

                progress_bar.close()

                logger.info(f"Text-only evaluation completed. Max concurrent tasks: {max_concurrent_seen}, Total tasks: {len(math_results)}")

                # Final processing - filter out failed evaluations for the final results
                processed_math_results = [r for r in math_results if not isinstance(r, dict) or not r.get("evaluation_failed", False)]

                model_results[key] = {
                    "math_results": processed_math_results
                }

            elif mode == "scene":
                # Scene mode - single evaluation per scene
                key = f"{mode}"

                # Execute with true async concurrency using semaphore
                logger.info(f"Processing {len(data)} {mode} scenes for {model.name} with max {concurrency} concurrent requests")

                # Create progress bar for this model's evaluation
                progress_bar = tqdm(
                    total=len(data),
                    desc=f"{model.name} {mode}",
                    unit="scene",
                    position=1,
                    leave=False
                )

                # Execute with true async concurrency using semaphore
                scene_results = []
                semaphore = asyncio.Semaphore(concurrency)

                # Track concurrent task count for debugging
                active_scene_tasks = 0
                max_scene_concurrent_seen = 0

                async def evaluate_single_scene(question_data):
                    """Evaluate a single scene with semaphore control"""
                    nonlocal active_scene_tasks, max_scene_concurrent_seen
                    active_scene_tasks += 1
                    max_scene_concurrent_seen = max(max_scene_concurrent_seen, active_scene_tasks)

                    try:
                        async with semaphore:
                            result = await self.evaluate_math_problem_async(
                                model, question_data, mode, image_dir, "implicit"
                            )
                            return result
                    except Exception as e:
                        logger.error(f"Error evaluating scene {question_data['question_id']}: {e}")
                        # Return error result instead of raising exception
                        return {
                            "question_id": question_data["question_id"],
                            "mode": mode,
                            "model": model.name,
                            "error": str(e),
                            "evaluation_failed": True
                        }
                    finally:
                        active_scene_tasks -= 1

                # Create all scene tasks immediately - true async concurrency
                all_scene_tasks = []
                for question_data in data:
                    task = asyncio.create_task(evaluate_single_scene(question_data))
                    all_scene_tasks.append(task)

                logger.info(f"Created {len(all_scene_tasks)} concurrent scene tasks with semaphore limit {concurrency}")

                # Process completed scene tasks as they finish without blocking
                completed_count = 0
                for completed_task in asyncio.as_completed(all_scene_tasks):
                    result = await completed_task
                    scene_results.append(result)
                    completed_count += 1

                    # Update progress bar (non-blocking)
                    progress_bar.update(1)

                    # Save intermediate results every save_intermediate_every scenes (non-blocking)
                    if completed_count % save_intermediate_every == 0:
                        processed_results = [r for r in scene_results[-save_intermediate_every:] if not isinstance(r, dict) or not r.get("evaluation_failed", False)]
                        if processed_results:
                            # Create background task for saving intermediate results
                            asyncio.create_task(self._save_intermediate_results_async(
                                model.name, processed_results,
                                "intermediate_results", f"{key}_intermediate"
                            ))

                progress_bar.close()

                logger.info(f"Scene evaluation completed. Max concurrent tasks: {max_scene_concurrent_seen}, Total tasks: {len(scene_results)}")

                # Final processing - filter out failed evaluations for the final results
                processed_scene_results = [r for r in scene_results if not isinstance(r, dict) or not r.get("evaluation_failed", False)]

                model_results[key] = {
                    "math_results": processed_scene_results
                }
                
            else:
                # Image modes - evaluate with both prompt modes
                for prompt_mode in prompt_modes:
                    key = f"{mode}_{prompt_mode}"

                    # Execute with true async concurrency using semaphore
                    logger.info(f"Processing {len(data)} {mode}_{prompt_mode} tasks for {model.name} with max {concurrency} concurrent requests")

                    # Create progress bar for this model's evaluation
                    progress_bar = tqdm(
                        total=len(data),
                        desc=f"{model.name} {mode}_{prompt_mode}",
                        unit="question",
                        position=1,
                        leave=False
                    )

                    # Execute with true async concurrency using semaphore
                    math_results = []
                    semaphore = asyncio.Semaphore(concurrency)

                    # Track concurrent task count for debugging
                    active_visual_tasks = 0
                    max_visual_concurrent_seen = 0

                    async def evaluate_single_question(question_data):
                        """Evaluate a single question with semaphore control"""
                        nonlocal active_visual_tasks, max_visual_concurrent_seen
                        active_visual_tasks += 1
                        max_visual_concurrent_seen = max(max_visual_concurrent_seen, active_visual_tasks)

                        try:
                            async with semaphore:
                                result = await self.evaluate_math_problem_async(
                                    model, question_data, mode, image_dir, prompt_mode
                                )
                                return result
                        except Exception as e:
                            logger.error(f"Error evaluating question {question_data['question_id']}: {e}")
                            # Return error result instead of raising exception
                            return {
                                "question_id": question_data["question_id"],
                                "mode": mode,
                                "prompt_mode": prompt_mode,
                                "model": model.name,
                                "error": str(e),
                                "evaluation_failed": True
                            }
                        finally:
                            active_visual_tasks -= 1

                    # Create all visual tasks immediately - true async concurrency
                    all_visual_tasks = []
                    for question_data in data:
                        task = asyncio.create_task(evaluate_single_question(question_data))
                        all_visual_tasks.append(task)

                    logger.info(f"Created {len(all_visual_tasks)} concurrent visual tasks with semaphore limit {concurrency}")

                    # Process completed visual tasks as they finish without blocking
                    completed_count = 0
                    for completed_task in asyncio.as_completed(all_visual_tasks):
                        result = await completed_task
                        math_results.append(result)
                        completed_count += 1

                        # Update progress bar (non-blocking)
                        progress_bar.update(1)

                        # Save intermediate results every save_intermediate_every questions (non-blocking)
                        if completed_count % save_intermediate_every == 0:
                            processed_results = [r for r in math_results[-save_intermediate_every:] if not isinstance(r, dict) or not r.get("evaluation_failed", False)]
                            if processed_results:
                                # Create background task for saving intermediate results
                                asyncio.create_task(self._save_intermediate_results_async(
                                    model.name, processed_results,
                                    "intermediate_results", f"{key}_intermediate"
                                ))

                    progress_bar.close()

                    logger.info(f"Visual evaluation completed. Max concurrent tasks: {max_visual_concurrent_seen}, Total tasks: {len(math_results)}")

                    # Final processing - filter out failed evaluations for the final results
                    processed_math_results = [r for r in math_results if not isinstance(r, dict) or not r.get("evaluation_failed", False)]

                    model_results[key] = {
                        "math_results": processed_math_results
                    }
    
        logger.info(f"Completed async evaluation for model: {model.name}")
        return model_results 