"""
Infrastructure metrics module - Metrics computation and storage.

This module contains adapters for computing and storing evaluation metrics
such as F1 score, precision, recall, and accuracy.
"""

import json
from pathlib import Path
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.domain import EvaluationMetrics, MetricsRepository, PredictionResult


class StandardMetricsRepository(MetricsRepository):
    """Standard implementation of metrics computation using scikit-learn.

    This adapter computes binary classification metrics and handles
    cases where some predictions failed to parse.
    """

    def compute_metrics(
        self,
        predictions: list[PredictionResult],
    ) -> EvaluationMetrics:
        """Compute evaluation metrics from prediction results.

        Args:
            predictions: List of prediction results.

        Returns:
            Computed evaluation metrics.
        """
        # Filter out predictions with parsing errors
        valid_predictions = [p for p in predictions if p.predicted_label is not None]

        if not valid_predictions:
            # All predictions failed - return zeros
            return EvaluationMetrics(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                total_examples=len(predictions),
                parsing_failures=len(predictions),
            )

        # Extract labels
        y_true = [p.ground_truth_label for p in valid_predictions]
        y_pred = [p.predicted_label for p in valid_predictions]

        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0.0)
        recall = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)
        accuracy = accuracy_score(y_true, y_pred)

        # Compute confusion matrix values
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        parsing_failures = len(predictions) - len(valid_predictions)

        return EvaluationMetrics(
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            accuracy=float(accuracy),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_examples=len(predictions),
            parsing_failures=parsing_failures,
        )

    def save_metrics(
        self,
        metrics: EvaluationMetrics,
        output_path: str,
    ) -> None:
        """Save metrics to a JSON file.

        Args:
            metrics: Metrics to save.
            output_path: Path where metrics should be saved.

        Raises:
            IOError: If saving fails.
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)

        except Exception as e:
            raise IOError(f"Failed to save metrics: {str(e)}") from e


class DetailedReportGenerator:
    """Generator for detailed evaluation reports.

    This creates comprehensive JSON reports containing all predictions,
    errors, and metrics for qualitative analysis.
    """

    def generate_report(
        self,
        predictions: list[PredictionResult],
        metrics: EvaluationMetrics,
        output_path: str,
    ) -> None:
        """Generate a detailed evaluation report.

        Args:
            predictions: All prediction results.
            metrics: Computed metrics.
            output_path: Path where report should be saved.

        Raises:
            IOError: If report generation fails.
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Build report structure
            report = {
                "summary": {
                    "total_examples": metrics.total_examples,
                    "parsing_failures": metrics.parsing_failures,
                    "successful_predictions": metrics.total_examples
                    - metrics.parsing_failures,
                    "metrics": metrics.to_dict(),
                },
                "predictions": [],
            }

            # Add detailed predictions
            for i, pred in enumerate(predictions):
                pred_dict = {
                    "index": i,
                    "sentence": pred.sentence,
                    "ground_truth": {
                        "label": pred.ground_truth_label,
                        "principle_id": pred.ground_truth_justification.principle_id,
                        "justification": pred.ground_truth_justification.justification_text,
                        "evidence": pred.ground_truth_justification.evidence_quote,
                    },
                    "raw_output": pred.raw_output,
                }

                if pred.predicted_justification:
                    pred_dict["prediction"] = {
                        "label": pred.predicted_label,
                        "principle_id": pred.predicted_justification.principle_id,
                        "justification": pred.predicted_justification.justification_text,
                        "evidence": pred.predicted_justification.evidence_quote,
                        "is_correct": pred.is_correct,
                    }
                else:
                    pred_dict["prediction"] = None
                    pred_dict["parsing_error"] = pred.parsing_error

                report["predictions"].append(pred_dict)

            # Write report
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)

        except Exception as e:
            raise IOError(f"Failed to generate report: {str(e)}") from e


class ConsoleReportGenerator:
    """Generator for console-friendly reports.

    This creates human-readable summaries for terminal output.
    """

    def generate_report(
        self,
        predictions: list[PredictionResult],
        metrics: EvaluationMetrics,
        output_path: Optional[str] = None,
    ) -> None:
        """Generate a console report.

        Args:
            predictions: All prediction results.
            metrics: Computed metrics.
            output_path: Optional path to save the report (ignored for console).
        """
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print(f"\nTotal Examples: {metrics.total_examples}")
        print(
            f"Successful Predictions: {metrics.total_examples - metrics.parsing_failures}"
        )
        print(f"Parsing Failures: {metrics.parsing_failures}")

        print("\nMetrics:")
        print(f"  F1 Score:  {metrics.f1_score:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")

        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics.true_positives}")
        print(f"  False Positives: {metrics.false_positives}")
        print(f"  True Negatives:  {metrics.true_negatives}")
        print(f"  False Negatives: {metrics.false_negatives}")

        # Show some example errors
        errors = [
            p for p in predictions if not p.is_correct and p.predicted_label is not None
        ]
        if errors:
            print(f"\nExample Errors (showing up to 3 of {len(errors)}):")
            for i, error in enumerate(errors[:3]):
                print(f"\n  Error {i + 1}:")
                print(f"    Sentence: {error.sentence[:100]}...")
                print(
                    f"    Ground Truth: {error.ground_truth_label} ({error.ground_truth_justification.principle_id})"
                )
                print(
                    f"    Predicted: {error.predicted_label} ({error.predicted_justification.principle_id})"
                )

        # Show parsing failures
        failures = [p for p in predictions if p.has_parsing_error]
        if failures:
            print(f"\nParsing Failures (showing up to 3 of {len(failures)}):")
            for i, failure in enumerate(failures[:3]):
                print(f"\n  Failure {i + 1}:")
                print(f"    Sentence: {failure.sentence[:100]}...")
                print(f"    Error: {failure.parsing_error}")
                print(f"    Raw Output: {failure.raw_output[:200]}...")
