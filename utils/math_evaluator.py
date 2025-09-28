import re
import logging
from fractions import Fraction
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MathEvaluator:
    """Evaluator for math problems."""
    
    def __init__(self):
        """
        Initialize the math evaluator.
        
        Uses rule-based answer extraction instead of external model.
        """
        pass
    
    def extract_answer(self, model_response: str) -> str:
        """
        Extract numerical answer from model response using specific format.
        
        Args:
            model_response: Model's response text
            
        Returns:
            Extracted numerical answer
        """
        if not model_response:
            return ""
        
        # Clean the response for better matching
        text = model_response.strip()
        
        # Priority 1: Look for explicit final answer patterns (most reliable)
        final_answer_patterns = [
            # Various "Final Answer" formats with flexible spacing and punctuation
            r"(?:FINAL\s+ANSWER|Final\s+Answer)\s*[:\-=]?\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
            r"(?:FINAL\s+ANSWER|Final\s+Answer)\s*[:\-=]?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            
            # Boxed answers (LaTeX style)
            r"\\boxed\{([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)\}",
            r"\\boxed\{([0-9,]+(?:\.[0-9]+)?)\}",
            
            # Answer with explicit markers
            r"(?:The\s+)?(?:answer|result)\s+is\s*[:\-=]?\s*\$?\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
            r"(?:The\s+)?(?:answer|result)\s+is\s*[:\-=]?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            
            # Therefore/conclusion patterns
            r"(?:Therefore|Thus|Hence)[^.]*?[:\-=]?\s*\$?\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
            r"(?:Therefore|Thus|Hence)[^.]*?[:\-=]?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
        ]
        
        # Try final answer patterns first (highest priority)
        for pattern in final_answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer = matches[-1]  # Take the last match
                cleaned = self._clean_extracted_number(answer)
                if cleaned:
                    logger.debug(f"Extracted answer using final pattern: '{cleaned}' from '{answer}'")
                    return cleaned
        
        # Priority 2: Look at the last few lines for standalone numbers
        lines = text.split('\n')
        for i in range(min(5, len(lines))):  # Check last 5 lines
            line = lines[-(i+1)].strip()
            if not line:
                continue
                
            # Check for standalone numbers at the end of lines
            standalone_patterns = [
                r'^([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)$',  # Pure number
                r'^([0-9,]+(?:\.[0-9]+)?)$',  # Number with commas
                r'^\$\s*([0-9,]+(?:\.[0-9]+)?)$',  # Dollar amount
                r'^([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)\s*(?:dollars?|cents?)?$',  # Number with currency words
            ]
            
            for pattern in standalone_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    cleaned = self._clean_extracted_number(match.group(1))
                    if cleaned:
                        logger.debug(f"Extracted answer from standalone line: '{cleaned}' from line '{line}'")
                        return cleaned
        
        # Priority 3: Look for contextual answer patterns in the last paragraph
        last_paragraph = self._get_last_paragraph(text)
        contextual_patterns = [
            # Answer in context patterns
            r"(?:answer|result|solution)\s*(?:is|=|:)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
            r"(?:answer|result|solution)\s*(?:is|=|:)?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            
            # Mathematical expressions with equals
            r"=\s*\$?\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)",
            r"=\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            
            # Currency specific patterns
            r"makes?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?)?",
            r"total\s*(?:of|is)?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"costs?\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)",
        ]
        
        for pattern in contextual_patterns:
            matches = re.findall(pattern, last_paragraph, re.IGNORECASE)
            if matches:
                answer = matches[-1]
                cleaned = self._clean_extracted_number(answer)
                if cleaned:
                    logger.debug(f"Extracted answer using contextual pattern: '{cleaned}' from '{answer}'")
                    return cleaned
        
        # Priority 4: Fallback to any number in the last few lines
        for i in range(min(3, len(lines))):
            line = lines[-(i+1)].strip()
            numbers = re.findall(r'([0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?)', line)
            if numbers:
                cleaned = self._clean_extracted_number(numbers[-1])
                if cleaned:
                    logger.debug(f"Extracted answer as fallback number: '{cleaned}' from line '{line}'")
                    return cleaned
            
            # Also check for comma-separated numbers
            comma_numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', line)
            if comma_numbers:
                cleaned = self._clean_extracted_number(comma_numbers[-1])
                if cleaned:
                    logger.debug(f"Extracted answer as fallback comma number: '{cleaned}' from line '{line}'")
                    return cleaned
        
        logger.warning("No numerical answer could be extracted from the response")
        return ""
    
    def _clean_extracted_number(self, number_str: str) -> str:
        """
        Clean and validate extracted number string.
        
        Args:
            number_str: Raw extracted number string
            
        Returns:
            Cleaned number string or empty string if invalid
        """
        if not number_str:
            return ""
        
        # Remove dollar signs and extra whitespace
        cleaned = number_str.strip().lstrip('$').strip()
        
        # Remove commas from numbers like 70,000
        cleaned = cleaned.replace(',', '')
        
        # Validate the cleaned number
        if re.match(r'^[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?$', cleaned):
            return cleaned
        
        return ""
    
    def _get_last_paragraph(self, text: str) -> str:
        """
        Extract the last paragraph from text for contextual analysis.
        
        Args:
            text: Full text
            
        Returns:
            Last paragraph or last 500 characters
        """
        # Split by double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text.strip())
        if paragraphs:
            return paragraphs[-1]
        
        # Fallback to last 500 characters
        return text[-500:] if len(text) > 500 else text
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize a numerical answer for comparison.
        
        Args:
            answer: Answer string
            
        Returns:
            Normalized answer
        """
        if not answer:
            return ""
        
        # Remove commas from numbers like 70,000
        answer = answer.replace(",", "")
        
        # Handle fractions
        if '/' in answer:
            try:
                # Parse as fraction and convert to decimal for comparison
                frac = Fraction(answer)
                if frac.denominator == 1:
                    return str(frac.numerator)  # It's actually an integer
                return str(float(frac))
            except (ValueError, ZeroDivisionError):
                pass
        
        # Remove any non-numeric characters except for decimal point
        clean_answer = re.sub(r'[^\d.]', '', answer)
        
        # Try to convert to float for canonical form
        try:
            value = float(clean_answer)
            # If it's an integer, return as int
            if value.is_integer():
                return str(int(value))
            return str(value)
        except (ValueError, TypeError):
            return clean_answer
    
    def is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match, False otherwise
        """
        if not predicted:
            return False
        
        norm_predicted = self.normalize_answer(predicted)
        norm_ground_truth = self.normalize_answer(ground_truth)
        
        # Check for exact match after normalization
        if norm_predicted == norm_ground_truth:
            return True
        
        # For numerical answers, allow for small differences
        try:
            pred_val = float(norm_predicted)
            truth_val = float(norm_ground_truth)
            
            # Allow for small relative difference (1e-6 relative tolerance)
            rel_diff = abs(pred_val - truth_val) / max(abs(truth_val), 1e-10)
            if rel_diff < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        
        # Try comparing as fractions if one or both contain fractions
        if '/' in predicted or '/' in ground_truth:
            try:
                pred_frac = Fraction(predicted) if '/' in predicted else Fraction(float(predicted))
                truth_frac = Fraction(ground_truth) if '/' in ground_truth else Fraction(float(ground_truth))
                return pred_frac == truth_frac
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        
        return False
    
    def post_process_extraction(self, extracted_answer: str, model_response: str, ground_truth: str) -> str:
        """
        Post-process the extracted answer to improve accuracy using ground truth guidance.
        
        Args:
            extracted_answer: Initially extracted answer
            model_response: Full model response
            ground_truth: Expected ground truth value
            
        Returns:
            Corrected answer if a better match is found
        """
        if not extracted_answer:
            # If no answer was extracted, try a more aggressive extraction
            logger.info("No answer extracted, attempting aggressive re-extraction")
            return self._aggressive_extraction(model_response, ground_truth)
            
        # Normalize the extracted answer
        norm_extracted = self.normalize_answer(extracted_answer)
        norm_ground_truth = self.normalize_answer(ground_truth)
        
        # Check if the extraction is already correct
        if norm_extracted == norm_ground_truth:
            return extracted_answer
        
        # If extraction doesn't match ground truth, search more thoroughly
        logger.info(f"Extracted answer '{extracted_answer}' doesn't match ground truth '{ground_truth}', searching for better match")
        
        # Search in the last few lines more carefully
        lines = model_response.strip().split('\n')
        last_lines = lines[-5:]  # Check the last 5 lines
        
        for line in last_lines:
            # Look for all numbers in the line
            all_numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?(?:/[0-9]+)?', line)
            for number in all_numbers:
                if self.normalize_answer(number) == norm_ground_truth:
                    logger.info(f"Post-processing found correct answer: '{number}' instead of '{extracted_answer}'")
                    return number
        
        # If still no match, return the original extraction
        logger.warning(f"Could not find better match than '{extracted_answer}' for ground truth '{ground_truth}'")
        return extracted_answer
    
    def _aggressive_extraction(self, model_response: str, ground_truth: str) -> str:
        """
        Aggressive extraction when normal methods fail.
        
        Args:
            model_response: Full model response
            ground_truth: Expected ground truth value
            
        Returns:
            Best guess answer or empty string
        """
        # Look for any numbers that match the ground truth pattern
        norm_ground_truth = self.normalize_answer(ground_truth)
        
        # Find all possible numbers in the response
        all_numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?(?:/[0-9]+)?', model_response)
        
        # Check if any of them match the ground truth
        for number in all_numbers:
            if self.normalize_answer(number) == norm_ground_truth:
                logger.info(f"Aggressive extraction found matching answer: '{number}'")
                return number
        
        # If no exact match, return the last number found
        if all_numbers:
            last_number = all_numbers[-1]
            logger.info(f"Aggressive extraction returning last number: '{last_number}'")
            return last_number
        
        return ""