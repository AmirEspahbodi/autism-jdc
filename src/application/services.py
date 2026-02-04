"""
Application services module - Use cases for the JDC system.
Refactored to support Pre-formatted SFT pipeline.
"""

from typing import Protocol

from src.config import KnowledgeBaseConfig, SystemConfig
from src.domain import (
    DataLoader,
    EvaluationMetrics,
    InferenceEngine,
    JustificationParser,
    LabeledExample,
    LLMTrainer,
    MetricsRepository,
    ParsingError,
    PredictionResult,
    ReportGenerator,
)


class FineTuneModelUseCase:
    """Use case for fine-tuning the LLM on labeled examples."""

    def __init__(
        self,
        trainer: LLMTrainer,
        data_loader: DataLoader,
        config: SystemConfig,
    ) -> None:
        self.trainer = trainer
        self.data_loader = data_loader
        self.config = config

    def execute(self, use_validation: bool = True) -> None:
        """Execute the fine-tuning pipeline."""
        print("=" * 60)
        print("STAGE 3: FINE-TUNING MODEL (SFT PRE-FORMATTED)")
        print("=" * 60)

        # Load training data
        print("\n[1/4] Loading training data...")
        # Now loads from dataset.json via PreformattedDataLoader
        training_examples = self.data_loader.load_training_data()
        print(f"✓ Loaded {len(training_examples)} training examples")

        # Optionally load validation data
        validation_examples = None
        if use_validation:
            try:
                print("\n[2/4] Loading validation data...")
                validation_examples = self.data_loader.load_validation_data()
                print(f"✓ Loaded {len(validation_examples)} validation examples")
            except Exception as e:
                print(f"⚠ Warning: Could not load validation data: {e}")
                print("  Continuing without validation...")

        # Execute training
        print("\n[3/4] Starting fine-tuning...")
        print(f"  Model: {self.config.model_type.value}")

        # Note: We no longer inject KB here. The Trainer adapter handles the
        # preformatted data directly.
        self.trainer.train(
            training_examples=training_examples,
            validation_examples=validation_examples,
        )
        print("✓ Training completed successfully")

        # Save model
        print("\n[4/4] Saving fine-tuned model...")
        output_path = str(self.config.output_dir / "fine_tuned_model")
        self.trainer.save_model(output_path)
        print(f"✓ Model saved to: {output_path}")


class EvaluateModelUseCase:
    """Use case for evaluating the fine-tuned model.
    Refactored to support Pre-formatted SFT data.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        parser: JustificationParser,
        metrics_repo: MetricsRepository,
        report_generator: ReportGenerator,
        data_loader: DataLoader,
        kb_config: KnowledgeBaseConfig,
    ) -> None:
        self.inference_engine = inference_engine
        self.parser = parser
        self.metrics_repo = metrics_repo
        self.report_generator = report_generator
        self.data_loader = data_loader
        self.kb_config = kb_config

    def execute(self, output_path: str) -> EvaluationMetrics:
        print("=" * 60)
        print("STAGE 4: EVALUATING MODEL")
        print("=" * 60)

        # 1. Load Test Data
        print("\n[1/4] Loading test data...")
        test_examples = self.data_loader.load_test_data()
        if not test_examples:
            print("⚠ No test examples found. Aborting evaluation.")
            return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        print(f"✓ Loaded {len(test_examples)} test examples")

        # 2. Run Inference
        print("\n[2/4] Running inference...")
        # Note: kb_text is unused in SFT inference as context is in the prompt,
        # but kept for interface compliance.
        kb_text = self.kb_config.get_all_principles_text()

        try:
            raw_outputs = self.inference_engine.batch_generate(test_examples, kb_text)
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            raise

        print(f"✓ Generated {len(raw_outputs)} predictions")

        # 3. Process Results (Parse & Compare)
        print("\n[3/4] Processing results...")
        prediction_results = []

        for i, (example, raw_output) in enumerate(zip(test_examples, raw_outputs)):
            # 3a. Parse Ground Truth (from model_output)
            try:
                # The SFT dataset contains the expected JSON in model_output
                ground_truth_justification = self.parser.parse(example.model_output)
                ground_truth_label = ground_truth_justification.to_label()
            except ParsingError:
                print(
                    f"⚠ Warning: Could not parse ground truth for example {i}. Skipping."
                )
                continue

            # 3b. Parse Prediction
            predicted_justification = None
            predicted_label = None
            parsing_error = None

            try:
                predicted_justification = self.parser.parse(raw_output)
                predicted_label = predicted_justification.to_label()
            except ParsingError as e:
                parsing_error = str(e)

            # 3c. Create Result Object
            # For SFT, 'sentence' is buried in input_prompt. We use a truncated prompt
            # or the full prompt for the report.
            display_sentence = example.input_prompt if example.input_prompt else "N/A"
            if len(display_sentence) > 200:
                display_sentence = display_sentence[:197] + "..."

            result = PredictionResult(
                sentence=display_sentence,
                ground_truth_justification=ground_truth_justification,
                ground_truth_label=ground_truth_label,
                raw_output=raw_output,
                predicted_justification=predicted_justification,
                predicted_label=predicted_label,
                parsing_error=parsing_error,
            )
            prediction_results.append(result)

        # 4. Compute Metrics & Generate Report
        print("\n[4/4] Calculating metrics and generating report...")
        metrics = self.metrics_repo.compute_metrics(prediction_results)

        # Save detailed report
        report_file = f"{output_path}_report.json"
        self.report_generator.generate_report(prediction_results, metrics, report_file)

        # Save metrics
        metrics_file = f"{output_path}_metrics.json"
        self.metrics_repo.save_metrics(metrics, metrics_file)

        return metrics
