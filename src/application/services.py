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
    (Kept largely the same, assumes test data might still be structured for detailed metric analysis)
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
        # Implementation remains similar to original for evaluation logic
        # ... (omitted to focus on Training Refactor) ...
        return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # Placeholder
