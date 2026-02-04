"""
Application services module - Use cases for the JDC system.

This module contains the application-layer use cases that orchestrate
domain entities and infrastructure adapters to accomplish business goals.
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
    """Use case for fine-tuning the LLM on labeled examples.

    This orchestrates the entire training pipeline:
    1. Load training data
    2. Optionally load validation data
    3. Execute fine-tuning via the LLMTrainer adapter
    4. Save the trained model

    Attributes:
        trainer: The LLM trainer adapter.
        data_loader: The data loading adapter.
        config: System configuration.
    """

    def __init__(
        self,
        trainer: LLMTrainer,
        data_loader: DataLoader,
        config: SystemConfig,
    ) -> None:
        """Initialize the fine-tuning use case.

        Args:
            trainer: LLM trainer implementation.
            data_loader: Data loader implementation.
            config: System configuration.
        """
        self.trainer = trainer
        self.data_loader = data_loader
        self.config = config

    def execute(self, use_validation: bool = True) -> None:
        """Execute the fine-tuning pipeline.

        Args:
            use_validation: Whether to use validation data during training.

        Raises:
            TrainingError: If training fails.
            DataLoadError: If data loading fails.
        """
        print("=" * 60)
        print("STAGE 3: FINE-TUNING MODEL")
        print("=" * 60)

        # Load training data
        print("\n[1/4] Loading training data...")
        training_examples = self.data_loader.load_training_data()
        print(f"✓ Loaded {len(training_examples)} training examples")

        # Optionally load validation data
        validation_examples = None
        if use_validation:
            try:
                print("\n[2/4] Loading validation data...")
                # FIX: PREVENT DATA LEAKAGE
                # Previously used load_test_data(), which contaminated the model with test set.
                # Now explicitly loads distinct validation set.
                validation_examples = self.data_loader.load_validation_data()
                print(f"✓ Loaded {len(validation_examples)} validation examples")
            except Exception as e:
                print(f"⚠ Warning: Could not load validation data: {e}")
                print("  Continuing without validation...")
        else:
            print("\n[2/4] Skipping validation data (use_validation=False)")

        # Execute training
        print("\n[3/4] Starting fine-tuning...")
        print(f"  Model: {self.config.model_type.value}")
        print(f"  Epochs: {self.config.training_hyperparameters.num_epochs}")
        print(f"  Learning Rate: {self.config.training_hyperparameters.learning_rate}")
        print(f"  Batch Size: {self.config.training_hyperparameters.batch_size}")
        print(f"  LoRA Rank: {self.config.lora_config.r}")
        print(
            f"  Quantization: {self.config.quantization_config.quantization_type.value}"
        )

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

        print("\n" + "=" * 60)
        print("FINE-TUNING COMPLETE")
        print("=" * 60)


class EvaluateModelUseCase:
    """Use case for evaluating the fine-tuned model.

    This orchestrates the entire evaluation pipeline:
    1. Load test data
    2. Generate justifications via the inference engine
    3. Parse outputs via the justification parser
    4. Map justifications to labels (deterministic mapping)
    5. Compute metrics
    6. Generate evaluation report

    Attributes:
        inference_engine: The inference engine adapter.
        parser: The justification parser adapter.
        metrics_repo: The metrics repository adapter.
        report_generator: The report generator adapter.
        data_loader: The data loading adapter.
        kb_config: Knowledge base configuration.
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
        """Initialize the evaluation use case.

        Args:
            inference_engine: Inference engine implementation.
            parser: Justification parser implementation.
            metrics_repo: Metrics repository implementation.
            report_generator: Report generator implementation.
            data_loader: Data loader implementation.
            kb_config: Knowledge base configuration.
        """
        self.inference_engine = inference_engine
        self.parser = parser
        self.metrics_repo = metrics_repo
        self.report_generator = report_generator
        self.data_loader = data_loader
        self.kb_config = kb_config

    def execute(self, output_path: str) -> EvaluationMetrics:
        """Execute the evaluation pipeline.

        Args:
            output_path: Base path for saving outputs (metrics and report).

        Returns:
            Computed evaluation metrics.

        Raises:
            DataLoadError: If test data loading fails.
            InferenceError: If inference fails.
        """
        print("=" * 60)
        print("STAGE 4: MODEL EVALUATION")
        print("=" * 60)

        # Step 1: Load test data
        print("\n[1/6] Loading test data...")
        test_examples = self.data_loader.load_test_data()
        print(f"✓ Loaded {len(test_examples)} test examples")

        # Step 2: Prepare knowledge base
        print("\n[2/6] Preparing knowledge base...")
        kb_text = self.kb_config.get_all_principles_text()
        print(f"✓ Knowledge base contains {len(self.kb_config.principles)} principles")

        # Step 3: Generate justifications
        print("\n[3/6] Generating justifications...")
        raw_outputs = self.inference_engine.batch_generate(test_examples, kb_text)
        print(f"✓ Generated {len(raw_outputs)} outputs")

        # Step 4: Parse outputs and create predictions
        print("\n[4/6] Parsing outputs...")
        predictions: list[PredictionResult] = []
        parsing_errors = 0

        for example, raw_output in zip(test_examples, raw_outputs):
            try:
                # Parse the justification
                predicted_justification = self.parser.parse(raw_output)

                # Deterministic mapping: justification -> label
                predicted_label = predicted_justification.to_label()

                prediction = PredictionResult(
                    sentence=example.sentence,
                    ground_truth_justification=example.ground_truth_justification,
                    ground_truth_label=example.ground_truth_label,
                    raw_output=raw_output,
                    predicted_justification=predicted_justification,
                    predicted_label=predicted_label,
                    parsing_error=None,
                )
            except Exception as e:
                # Handle parsing failures gracefully
                parsing_errors += 1
                prediction = PredictionResult(
                    sentence=example.sentence,
                    ground_truth_justification=example.ground_truth_justification,
                    ground_truth_label=example.ground_truth_label,
                    raw_output=raw_output,
                    predicted_justification=None,
                    predicted_label=None,
                    parsing_error=str(e),
                )

            predictions.append(prediction)

        print(
            f"✓ Successfully parsed {len(predictions) - parsing_errors}/{len(predictions)}"
        )
        if parsing_errors > 0:
            print(f"⚠ Warning: {parsing_errors} parsing failures")

        # Step 5: Compute metrics
        print("\n[5/6] Computing metrics...")
        metrics = self.metrics_repo.compute_metrics(predictions)
        print(f"✓ Metrics computed:")
        print(f"  F1 Score:  {metrics.f1_score:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")

        # Save metrics
        metrics_path = f"{output_path}_metrics.json"
        self.metrics_repo.save_metrics(metrics, metrics_path)
        print(f"✓ Metrics saved to: {metrics_path}")

        # Step 6: Generate detailed report
        print("\n[6/6] Generating evaluation report...")
        report_path = f"{output_path}_report.json"
        self.report_generator.generate_report(predictions, metrics, report_path)
        print(f"✓ Report saved to: {report_path}")

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return metrics


class PromptTemplateBuilder(Protocol):
    """Protocol for building prompts from examples.

    This is used by infrastructure adapters to format examples
    into the correct prompt format for the model.
    """

    def build_training_prompt(
        self,
        example: LabeledExample,
        kb_text: str,
    ) -> str:
        """Build a training prompt from a labeled example."""
        ...

    def build_inference_prompt(
        self,
        sentence: str,
        kb_text: str,
        context_before: str | None = None,
        context_after: str | None = None,
    ) -> str:
        """Build an inference prompt from a sentence."""
        ...
