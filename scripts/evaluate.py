"""
Evaluation script for JDC Framework (Step 4: Evaluation).

Usage:
    python scripts/evaluate.py --config configs/eval_config.yaml
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from utils.config import EvaluationConfig, load_config_from_yaml
from modeling.model_loader import ModelLoader
from data.dataset import JDCDataset
from inference.generator import InferenceEngine, PredictionPostProcessor
from inference.parser import ResilientJSONParser, PrincipleMapper
from utils.metrics import MetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate(config: EvaluationConfig):
    """
    Main evaluation function.
    
    Args:
        config: Evaluation configuration
    """
    logger.info("=" * 60)
    logger.info("JDC FRAMEWORK - STEP 4: EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Adapter: {config.adapter_path}")
    logger.info(f"Test data: {config.test_data_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load model and tokenizer
    logger.info("Loading fine-tuned model...")
    model, tokenizer = ModelLoader.load_model_for_inference(
        model_name=config.model_name,
        adapter_path=config.adapter_path,
        load_in_4bit=config.load_in_4bit,
        device_map=config.device_map,
    )
    
    # 2. Load test data
    logger.info("Loading test data...")
    dataset_handler = JDCDataset(
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        use_chat_template=True
    )
    
    test_data = dataset_handler.load_data(config.test_data_path)
    test_dataset = dataset_handler.prepare_dataset(test_data, for_training=False)
    
    logger.info(f"Test examples: {len(test_dataset)}")
    logger.info("")
    
    # Extract components
    input_prompts = [example["text"] for example in test_dataset]
    input_texts = [example["input_prompt"] for example in test_dataset]
    
    # Extract ground truth labels if available
    ground_truth_labels = None
    if "label" in test_dataset.features:
        ground_truth_labels = test_dataset["label"]
        logger.info(f"Ground truth labels found: {len(ground_truth_labels)}")
    else:
        logger.warning("No ground truth labels found in test data")
    
    # 3. Setup inference engine
    logger.info("Setting up inference engine...")
    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
    )
    
    # 4. Generate predictions
    logger.info("Generating predictions...")
    logger.info("-" * 60)
    generated_texts = inference_engine.generate_all(
        prompts=input_prompts,
        show_progress=True
    )
    logger.info("-" * 60)
    logger.info(f"Generated {len(generated_texts)} predictions")
    logger.info("")
    
    # 5. Setup parsers
    logger.info("Setting up JSON parser and principle mapper...")
    json_parser = ResilientJSONParser(
        max_retries=config.max_parse_retries,
        fallback_principle=config.fallback_principle,
        expected_keys=["principle_id", "justification"]
    )
    
    principle_mapper = PrincipleMapper(
        ableist_principles=config.ableist_principles,
        non_ableist_principles=config.non_ableist_principles,
        error_principle=config.fallback_principle
    )
    
    # 6. Process predictions
    logger.info("Processing predictions...")
    post_processor = PredictionPostProcessor(
        json_parser=json_parser,
        principle_mapper=principle_mapper
    )
    
    processed_results = post_processor.process_predictions(
        generated_texts=generated_texts,
        ground_truth_labels=ground_truth_labels,
        input_texts=input_texts
    )
    logger.info("")
    
    # 7. Compute metrics (if ground truth available)
    metrics = None
    if ground_truth_labels is not None:
        logger.info("Computing evaluation metrics...")
        metrics_calculator = MetricsCalculator(exclude_errors=True)
        
        metrics = metrics_calculator.compute_metrics(
            y_true=processed_results["ground_truth_labels"],
            y_pred=processed_results["predicted_labels"]
        )
        
        # Print report
        report = metrics_calculator.format_metrics_report(metrics)
        print("\n" + report)
        logger.info("")
    
    # 8. Save results
    logger.info("Saving results...")
    
    # Save metrics
    if metrics:
        metrics_path = os.path.join(config.output_dir, f"results_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save full predictions
    if config.save_predictions:
        predictions_data = []
        for i, (input_text, generated_text, parsed_output, pred_label) in enumerate(
            zip(input_texts, generated_texts, 
                processed_results["parsed_outputs"],
                processed_results["predicted_labels"])
        ):
            record = {
                "index": i,
                "input_text": input_text,
                "generated_text": generated_text,
                "principle_id": parsed_output["principle_id"],
                "justification": parsed_output["justification"],
                "predicted_label": pred_label,
                "parse_failed": parsed_output.get("parse_failed", False)
            }
            
            if ground_truth_labels:
                record["ground_truth_label"] = ground_truth_labels[i]
            
            predictions_data.append(record)
        
        predictions_path = os.path.join(
            config.output_dir, 
            f"predictions_{timestamp}.json"
        )
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        logger.info(f"Predictions saved to: {predictions_path}")
    
    # Save disagreements for qualitative analysis
    if config.save_disagreements and ground_truth_labels:
        disagreements = post_processor.create_disagreement_records(processed_results)
        
        if disagreements:
            disagreements_df = pd.DataFrame(disagreements)
            disagreements_path = os.path.join(
                config.output_dir,
                f"disagreements_{timestamp}.csv"
            )
            disagreements_df.to_csv(disagreements_path, index=False)
            logger.info(f"Disagreements saved to: {disagreements_path}")
            logger.info(f"Total disagreements: {len(disagreements)}")
        else:
            logger.info("No disagreements found (perfect predictions!)")
    
    # 9. Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total predictions: {processed_results['total_predictions']}")
    logger.info(f"Parse failures: {processed_results['parse_failures']}")
    
    if metrics:
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    logger.info("")
    logger.info("Output files:")
    if metrics:
        logger.info(f"  - {os.path.basename(metrics_path)}")
    if config.save_predictions:
        logger.info(f"  - {os.path.basename(predictions_path)}")
    if config.save_disagreements and ground_truth_labels and disagreements:
        logger.info(f"  - {os.path.basename(disagreements_path)}")
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned JDC model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config_from_yaml(args.config, EvaluationConfig)
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(config.model_dump_json(indent=2))
    logger.info("")
    
    # Start evaluation
    evaluate(config)


if __name__ == "__main__":
    main()
