"""
Async utilities for saving and managing evaluation results with separate files per model.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class AsyncResultWriter:
    """Async manager for evaluation results with separate files per model."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the async result writer.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ensure_dirs()
    
    def ensure_dirs(self):
        """Create necessary directories."""
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def save_model_results_async(
        self, 
        model_name: str, 
        model_results: Dict[str, Any]
    ) -> str:
        """
        Save all results for a specific model in a single JSON file asynchronously.
        
        Args:
            model_name: Name of the model
            model_results: Dictionary containing all evaluation results for this model
            
        Returns:
            Path to the saved file
        """
        # Create a safe filename
        safe_model_name = model_name.replace('/', '_').replace(':', '_').replace('.', '_')
        
        # Create output file path for this model
        out_file = os.path.join(
            self.results_dir,
            f"{self.timestamp}_{safe_model_name}_complete_results.json"
        )
        
        # Prepare the complete model data
        model_data = {
            "model_name": model_name,
            "timestamp": self.timestamp,
            "evaluation_timestamp": datetime.now().isoformat(),
            "results": model_results
        }
        
        # Add computed metrics
        model_data["metrics"] = await self._compute_model_metrics_async(model_results)
        
        try:
            # Write file asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_json_file,
                out_file,
                model_data
            )
            
            logger.info(f"Saved complete results for {model_name} to {out_file}")
            return out_file
        except Exception as e:
            logger.error(f"Failed to save results for {model_name}: {e}")
            raise
    
    def _write_json_file(self, file_path: str, data: Dict[str, Any]):
        """
        Write JSON data to file synchronously (for use with executor).
        
        Args:
            file_path: Path to write the file
            data: Data to write
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def _compute_model_metrics_async(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute metrics for a model's results asynchronously.
        
        Args:
            model_results: Dictionary containing all results for this model
            
        Returns:
            Dictionary of computed metrics
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._compute_model_metrics,
            model_results
        )
    
    def _compute_model_metrics(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute metrics for a model's results.
        
        Args:
            model_results: Dictionary containing all results for this model
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        for key, results_data in model_results.items():
            # Process math results
            if "math_results" in results_data:
                math_results = results_data["math_results"]
                math_accuracy = self.compute_math_accuracy(math_results)
                metrics[f"{key}_math_accuracy"] = math_accuracy
                
                # Compute category-specific accuracy if category data is available
                category_accuracy = self.compute_category_accuracy(math_results)
                if category_accuracy:
                    metrics[f"{key}_category_accuracy"] = category_accuracy
            
            # Scene evaluation removed
        
        return metrics
    
    def compute_math_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute math accuracy excluding failed evaluations.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with accuracy metrics and failure statistics
        """
        if not results:
            return {
                "accuracy": 0.0,
                "total_questions": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "failure_rate": 0.0
            }
        
        # Separate successful and failed evaluations
        successful_results = []
        failed_results = []
        
        for result in results:
            if result.get("evaluation_failed", False) or "error" in result:
                failed_results.append(result)
            else:
                successful_results.append(result)
        
        total_questions = len(results)
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        
        if successful_count == 0:
            logger.warning("No successful evaluations found for accuracy calculation")
            return {
                "accuracy": 0.0,
                "total_questions": total_questions,
                "successful_evaluations": 0,
                "failed_evaluations": failed_count,
                "failure_rate": 1.0,
                "error_details": [r.get("error", "Unknown error") for r in failed_results[:5]]
            }
        
        # Calculate accuracy based only on successful evaluations
        correct_count = sum(1 for r in successful_results if r.get("math_correct", False) is True)
        accuracy = correct_count / successful_count
        failure_rate = failed_count / total_questions
        
        logger.info(
            f"Accuracy calculation: {correct_count}/{successful_count} correct "
            f"({successful_count}/{total_questions} successful evaluations)"
        )
        
        return {
            "accuracy": accuracy,
            "total_questions": total_questions,
            "successful_evaluations": successful_count,
            "failed_evaluations": failed_count,
            "failure_rate": failure_rate,
            "correct_answers": correct_count,
            "error_details": [r.get("error", "Unknown error") for r in failed_results[:5]] if failed_results else []
        }
    
    def compute_category_accuracy(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute accuracy broken down by data categories and subcategories.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with category and subcategory accuracy breakdowns
        """
        if not results:
            return {}
        
        # Filter out failed evaluations
        successful_results = [
            r for r in results 
            if not r.get("evaluation_failed", False) and "error" not in r
        ]
        
        if not successful_results:
            return {}
        
        # Group results by category and subcategory
        category_results = {}
        subcategory_results = {}
        
        for result in successful_results:
            category = result.get('data_category', 'unknown')
            subcategory = result.get('data_subcategory', 'unknown')
            is_correct = result.get("math_correct", False) is True
            
            # Group by category
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(is_correct)
            
            # Group by subcategory
            if subcategory not in subcategory_results:
                subcategory_results[subcategory] = []
            subcategory_results[subcategory].append(is_correct)
        
        # Calculate accuracies
        category_accuracy = {}
        for category, correct_list in category_results.items():
            accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
            category_accuracy[category] = {
                'accuracy': accuracy,
                'correct': sum(correct_list),
                'total': len(correct_list)
            }
        
        subcategory_accuracy = {}
        for subcategory, correct_list in subcategory_results.items():
            accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
            subcategory_accuracy[subcategory] = {
                'accuracy': accuracy,
                'correct': sum(correct_list),
                'total': len(correct_list)
            }
        
        return {
            'by_category': category_accuracy,
            'by_subcategory': subcategory_accuracy
        }
    
    # Removed compute_scene_scores - scene evaluation not needed
    
    async def save_evaluation_summary_async(
        self, 
        summary_data: Dict[str, Any]
    ) -> str:
        """
        Save overall evaluation summary asynchronously.
        
        Args:
            summary_data: Summary data
            
        Returns:
            Path to the saved file
        """
        # Ensure seed information is included
        if "config" in summary_data and "seed" in summary_data["config"]:
            logger.info(f"Evaluation used random seed: {summary_data['config']['seed']}")
        
        out_file = os.path.join(
            self.results_dir,
            f"{self.timestamp}_evaluation_summary.json"
        )
        
        try:
            # Write file asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_json_file,
                out_file,
                summary_data
            )
            
            logger.info(f"Saved evaluation summary to {out_file}")
            return out_file
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            raise
    
    async def save_all_model_results_async(
        self,
        all_model_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[str]:
        """
        Save results for all models concurrently.
        
        Args:
            all_model_results: Dictionary mapping model names to their results
            config: Evaluation configuration
            
        Returns:
            List of file paths where results were saved
        """
        # Create tasks for saving each model's results
        save_tasks = []
        for model_name, model_results in all_model_results.items():
            task = self.save_model_results_async(model_name, model_results)
            save_tasks.append(task)
        
        # Execute all save operations concurrently
        saved_files = await asyncio.gather(*save_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        successful_saves = []
        for i, result in enumerate(saved_files):
            if isinstance(result, Exception):
                model_name = list(all_model_results.keys())[i]
                logger.error(f"Failed to save results for {model_name}: {result}")
            else:
                successful_saves.append(result)
        
        # Create and save evaluation summary
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "models_evaluated": list(all_model_results.keys()),
            "total_models": len(all_model_results),
            "results_files": successful_saves,
            "evaluation_completed": True
        }
        
        summary_file = await self.save_evaluation_summary_async(summary_data)
        successful_saves.append(summary_file)
        
        return successful_saves 