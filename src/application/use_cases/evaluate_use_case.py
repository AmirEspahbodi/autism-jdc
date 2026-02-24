"""Evaluate use case: inference over dataset splits and metric computation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.application.ports.evaluator_port import IEvaluatorService
from src.application.ports.model_port import IModelService
from src.application.ports.repository_port import IDatasetRepository
from src.domain.entities import EvaluationResult, JDCSample, ParsedOutput
from src.domain.exceptions import InferenceError
from src.domain.value_objects import derive_label


class EvaluateUseCase:
    """Orchestrates inference, metric computation, and report generation.

    For each requested split the use case:
      1. Loads the dataset split.
      2. Loads the fine-tuned model from a checkpoint.
      3. Runs per-sample inference.
      4. Computes quantitative metrics.
      5. Generates a disagreement analysis report.
      6. Saves all outputs to the configured output directory.

    Args:
        repository: Concrete implementation of IDatasetRepository.
        model_service: Concrete implementation of IModelService.
        evaluator_service: Concrete implementation of IEvaluatorService.
        checkpoint_path: Path to the saved adapter weights directory.
        output_dir: Directory for evaluation JSON reports.
    """

    def __init__(
        self,
        repository: IDatasetRepository,
        model_service: IModelService,
        evaluator_service: IEvaluatorService,
        checkpoint_path: Path,
        output_dir: Path,
    ) -> None:
        self._repository = repository
        self._model_service = model_service
        self._evaluator_service = evaluator_service
        self._checkpoint_path = checkpoint_path
        self._output_dir = output_dir

    def execute(self, splits: list[str] | None = None) -> dict[str, EvaluationResult]:
        """Run evaluation on the specified splits.

        Args:
            splits: Which splits to evaluate. Defaults to both
                    ``["validation", "test"]`` when None.

        Returns:
            A mapping from split name to EvaluationResult.

        Raises:
            DatasetLoadError: If a dataset split cannot be read.
            ModelLoadError: If checkpoint loading fails.
        """
        if splits is None:
            splits = ["validation", "test"]

        logger.info(f"EvaluateUseCase: loading model from {self._checkpoint_path} …")
        self._model_service.load_model()
        self._model_service.load_from_checkpoint(self._checkpoint_path)

        results: dict[str, EvaluationResult] = {}

        for split in splits:
            logger.info(f"EvaluateUseCase: evaluating split='{split}' …")
            samples = self._load_split(split)
            predictions = self._run_inference(samples, split)
            result = self._evaluator_service.evaluate(samples, predictions, split)
            results[split] = result

            self._save_evaluation_result(result)
            self._save_disagreement_report(samples, predictions, split)
            self._log_summary(result)

        return results

    def _load_split(self, split: str) -> list[JDCSample]:
        """Load dataset samples for the given split name.

        Args:
            split: "validation" or "test".

        Returns:
            List of JDCSample objects.

        Raises:
            ValueError: If split is not "validation" or "test".
            DatasetLoadError: If the file cannot be read.
        """
        if split == "validation":
            return self._repository.load_validation()
        elif split == "test":
            return self._repository.load_test()
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'validation' or 'test'.")

    def _run_inference(
        self,
        samples: list[JDCSample],
        split: str,
    ) -> list[ParsedOutput]:
        """Run model inference on every sample in the split.

        Failed inference attempts are replaced with a fallback ParsedOutput
        predicting P0 (not ableist) and the error is logged.

        Args:
            samples: Dataset samples to run inference on.
            split: Split name (used for logging only).

        Returns:
            List of ParsedOutput objects, one per sample (same order).
        """
        predictions: list[ParsedOutput] = []
        failed = 0

        for idx, sample in enumerate(samples):
            try:
                pred: ParsedOutput = self._model_service.predict(sample.input_prompt)
                # Always override is_ableist with deterministic derivation
                pred = ParsedOutput(
                    justification_reasoning=pred.justification_reasoning,
                    evidence_quote=pred.evidence_quote,
                    principle_id=pred.principle_id,
                    principle_name=pred.principle_name,
                    is_ableist=derive_label(pred.principle_id),
                )
                predictions.append(pred)
            except InferenceError as exc:
                logger.warning(
                    f"[{split}] Inference failed for sample id='{sample.id}' "
                    f"(idx={idx}): {exc}. Using fallback prediction P0."
                )
                fallback = ParsedOutput(
                    justification_reasoning="INFERENCE_FAILED",
                    evidence_quote="",
                    principle_id="P0",
                    principle_name="Not Ableist",
                    is_ableist=False,
                )
                predictions.append(fallback)
                failed += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"[{split}] Inference progress: {idx + 1}/{len(samples)}")

        logger.info(
            f"[{split}] Inference complete. "
            f"Total={len(samples)}, Failed={failed}."
        )
        return predictions

    def _save_evaluation_result(self, result: EvaluationResult) -> None:
        """Serialize and save an EvaluationResult to disk.

        Args:
            result: The EvaluationResult to save.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{result.split}_{timestamp}.json"
        out_path = self._output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved evaluation result to {out_path}")

    def _save_disagreement_report(
        self,
        samples: list[JDCSample],
        predictions: list[ParsedOutput],
        split: str,
    ) -> None:
        """Build and save a disagreement analysis report.

        A disagreement occurs when the predicted is_ableist label differs
        from the ground-truth derived_label.

        Args:
            samples: Ground-truth JDCSample objects.
            predictions: Corresponding model predictions.
            split: Name of the evaluated split.
        """
        disagreements: list[dict[str, object]] = []
        fp_count = 0
        fn_count = 0
        principle_pair_counts: dict[str, int] = {}

        for sample, pred in zip(samples, predictions):
            gt_label = sample.derived_label
            pred_label = pred.is_ableist

            if gt_label == pred_label:
                continue

            if pred_label and not gt_label:
                error_type = "FALSE_POSITIVE"
                fp_count += 1
            else:
                error_type = "FALSE_NEGATIVE"
                fn_count += 1

            pair_key = (
                f"{sample.parsed_output.principle_id} -> {pred.principle_id}"
            )
            principle_pair_counts[pair_key] = (
                principle_pair_counts.get(pair_key, 0) + 1
            )

            disagreements.append(
                {
                    "id": sample.id,
                    "input_prompt": sample.input_prompt[:500],
                    "ground_truth_principle": sample.parsed_output.principle_id,
                    "ground_truth_label": gt_label,
                    "predicted_principle": pred.principle_id,
                    "predicted_label": pred_label,
                    "generated_justification": pred.justification_reasoning,
                    "generated_evidence": pred.evidence_quote,
                    "error_type": error_type,
                }
            )

        summary = {
            "split": split,
            "total_samples": len(samples),
            "total_disagreements": len(disagreements),
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "principle_pair_frequency": dict(
                sorted(
                    principle_pair_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
            "disagreements": disagreements,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"disagreement_report_{split}_{timestamp}.json"
        out_path = self._output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info(
            f"[{split}] Disagreement report saved to {out_path}. "
            f"Total={len(disagreements)}, FP={fp_count}, FN={fn_count}."
        )
        logger.info(
            f"[{split}] Most common misclassified pairs: "
            f"{list(principle_pair_counts.items())[:5]}"
        )

    def _log_summary(self, result: EvaluationResult) -> None:
        """Log a formatted metric summary table to stdout.

        Args:
            result: The EvaluationResult to summarise.
        """
        separator = "=" * 60
        logger.info(separator)
        logger.info(f"  EVALUATION SUMMARY — split: {result.split.upper()}")
        logger.info(separator)
        logger.info(f"  F1  (macro)          : {result.f1:.4f}")
        logger.info(f"  Precision (macro)    : {result.precision:.4f}")
        logger.info(f"  Recall (macro)       : {result.recall:.4f}")
        logger.info(f"  Accuracy             : {result.accuracy:.4f}")
        logger.info(f"  Principle Accuracy   : {result.principle_accuracy:.4f}")
        logger.info(separator)
        logger.info("  Classification Report:")
        for line in result.classification_report.splitlines():
            logger.info(f"    {line}")
        logger.info(separator)
