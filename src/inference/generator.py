"""
Inference engine for generating structured predictions.

Handles:
- Batched generation for efficiency
- Generation parameters (temperature, top_p, etc.)
- Post-processing of model outputs
"""

import torch
import logging
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles model inference with batched generation.
    
    Key features:
    - Batched processing for efficiency
    - Configurable generation parameters
    - Progress tracking
    - Memory management
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
        device: str = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer (with padding_side='left' for batched generation)
            batch_size: Number of examples per batch
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            do_sample: Enable sampling (vs greedy decoding)
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Auto-detect device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
        
        # Verify tokenizer padding side
        if self.tokenizer.padding_side != "left":
            logger.warning(
                "Tokenizer padding_side should be 'left' for batched generation. "
                f"Current: {self.tokenizer.padding_side}"
            )
        
        logger.info(f"Inference engine initialized on device: {self.device}")
        logger.info(f"Generation params: temp={temperature}, top_p={top_p}, "
                   f"max_new_tokens={max_new_tokens}")
    
    def generate_batch(
        self,
        prompts: List[str],
        return_full_text: bool = False
    ) -> List[str]:
        """
        Generate outputs for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            return_full_text: If True, return prompt + generation
            
        Returns:
            List of generated texts
            
        Note:
            With padding_side='left', all prompts are padded on the left,
            ensuring they align at the end. This allows the model to generate
            from the actual prompt end for all sequences simultaneously.
        """
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        if return_full_text:
            # Return full sequence (prompt + generation)
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
        else:
            # Return only the generated portion
            # Calculate where the new tokens start
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            generated_texts = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
        
        return generated_texts
    
    def generate_all(
        self,
        prompts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate outputs for all prompts with batching.
        
        Args:
            prompts: List of all input prompts
            show_progress: Show progress bar
            
        Returns:
            List of generated texts (same length as prompts)
        """
        all_outputs = []
        
        # Create batches
        num_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(prompts), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Generating",
                total=num_batches,
                unit="batch"
            )
        
        for i in iterator:
            batch_prompts = prompts[i:i + self.batch_size]
            batch_outputs = self.generate_batch(batch_prompts)
            all_outputs.extend(batch_outputs)
            
            # Clear CUDA cache periodically to prevent OOM
            if i % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"Generated {len(all_outputs)} outputs")
        return all_outputs
    
    def predict_with_prompts(
        self,
        prompts: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generate and return structured outputs.
        
        Args:
            prompts: List of input prompts
            show_progress: Show progress bar
            
        Returns:
            List of dicts with {prompt, generated_text}
        """
        generated_texts = self.generate_all(prompts, show_progress)
        
        results = []
        for prompt, text in zip(prompts, generated_texts):
            results.append({
                "prompt": prompt,
                "generated_text": text
            })
        
        return results


class PredictionPostProcessor:
    """
    Post-process model predictions for evaluation.
    
    Combines:
    - Raw generation
    - JSON parsing
    - Principle mapping
    - Ground truth comparison
    """
    
    def __init__(
        self,
        json_parser,
        principle_mapper,
    ):
        """
        Initialize post-processor.
        
        Args:
            json_parser: ResilientJSONParser instance
            principle_mapper: PrincipleMapper instance
        """
        self.json_parser = json_parser
        self.principle_mapper = principle_mapper
    
    def process_predictions(
        self,
        generated_texts: List[str],
        ground_truth_labels: List[int] = None,
        input_texts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process all predictions and compute results.
        
        Args:
            generated_texts: Raw model outputs
            ground_truth_labels: True labels (for evaluation)
            input_texts: Original input texts (for logging)
            
        Returns:
            Dictionary with:
                - parsed_outputs: List of parsed JSON dicts
                - predicted_labels: List of predicted binary labels
                - ground_truth_labels: List of true labels (if provided)
                - parse_failures: Count of parsing failures
        """
        # Parse all outputs
        parsed_outputs = self.json_parser.batch_parse(generated_texts)
        
        # Extract principle IDs
        principle_ids = [output["principle_id"] for output in parsed_outputs]
        
        # Map to binary labels
        predicted_labels = self.principle_mapper.batch_map(principle_ids)
        
        # Count parse failures
        parse_failures = sum(
            1 for output in parsed_outputs
            if output.get("parse_failed", False)
        )
        
        results = {
            "parsed_outputs": parsed_outputs,
            "predicted_labels": predicted_labels,
            "parse_failures": parse_failures,
            "total_predictions": len(generated_texts)
        }
        
        if ground_truth_labels is not None:
            results["ground_truth_labels"] = ground_truth_labels
        
        if input_texts is not None:
            results["input_texts"] = input_texts
        
        logger.info(
            f"Processed {len(generated_texts)} predictions. "
            f"Parse failures: {parse_failures} ({100*parse_failures/len(generated_texts):.1f}%)"
        )
        
        return results
    
    def create_disagreement_records(
        self,
        processed_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create records for cases where prediction != ground truth.
        
        Args:
            processed_results: Output from process_predictions
            
        Returns:
            List of disagreement records for qualitative analysis
        """
        if "ground_truth_labels" not in processed_results:
            logger.warning("No ground truth labels provided")
            return []
        
        predicted_labels = processed_results["predicted_labels"]
        ground_truth_labels = processed_results["ground_truth_labels"]
        parsed_outputs = processed_results["parsed_outputs"]
        input_texts = processed_results.get("input_texts", [None] * len(predicted_labels))
        
        disagreements = []
        
        for i, (pred, truth) in enumerate(zip(predicted_labels, ground_truth_labels)):
            if pred != truth:
                record = {
                    "index": i,
                    "input_text": input_texts[i] if input_texts[i] else f"Example {i}",
                    "ground_truth_label": truth,
                    "predicted_label": pred,
                    "principle_id": parsed_outputs[i]["principle_id"],
                    "justification": parsed_outputs[i]["justification"],
                    "parse_failed": parsed_outputs[i].get("parse_failed", False)
                }
                disagreements.append(record)
        
        logger.info(f"Found {len(disagreements)} disagreements")
        return disagreements
