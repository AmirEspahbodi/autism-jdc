"""scikit-learn based evaluator service.

Computes all quantitative metrics for both binary classification (is_ableist)
and multi-class principle classification (P0..P4).
"""
from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.application.ports.evaluator_port import IEvaluatorService
from src.domain.entities import EvaluationResult, JDCSample, ParsedOutput
from src.domain.value_objects import derive_label


class SklearnEvaluatorService(IEvaluatorService):
    """Computes JDC evaluation metrics using scikit-learn.

    Binary metrics are computed over the is_ableist label.
    Principle accuracy is computed over the exact principle_id string.
    """

    def evaluate(
        self,
        samples: list[JDCSample],
        predictions: list[ParsedOutput],
        split: str,
    ) -> EvaluationResult:
        """Compute all evaluation metrics for a dataset split.

        Args:
            samples: Ground-truth JDCSample objects.
            predictions: Model-predicted ParsedOutput objects (same order).
            split: Name of the split ("validation" or "test").

        Returns:
            A fully populated EvaluationResult.

        Raises:
            ValueError: If samples and predictions have different lengths.
        """
        if len(samples) != len(predictions):
            raise ValueError(
                f"Mismatch: {len(samples)} ground-truth samples but "
                f"{len(predictions)} predictions for split='{split}'."
            )

        if len(samples) == 0:
            logger.warning(f"Evaluator received empty sample list for split='{split}'.")
            return EvaluationResult(
                split=split,
                f1=0.0,
                precision=0.0,
                recall=0.0,
                accuracy=0.0,
                principle_accuracy=0.0,
                confusion_matrix=[[0, 0], [0, 0]],
                classification_report="No samples to evaluate.",
            )

        # --- Ground truth ---
        # Ground truth labels ALWAYS come from derive_label(ground_truth_principle_id)
        # to guarantee consistency, not from parsed_output.is_ableist directly.
        y_true_binary: list[int] = [
            int(derive_label(s.parsed_output.principle_id)) for s in samples
        ]
        y_true_principle: list[str] = [
            s.parsed_output.principle_id for s in samples
        ]

        # --- Predictions ---
        # Predicted labels come from derive_label(predicted_principle_id).
        y_pred_binary: list[int] = [
            int(derive_label(p.principle_id)) for p in predictions
        ]
        y_pred_principle: list[str] = [p.principle_id for p in predictions]

        # --- Binary classification metrics ---
        f1 = float(
            f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
        )
        precision = float(
            precision_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
        )
        recall = float(
            recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
        )
        accuracy = float(accuracy_score(y_true_binary, y_pred_binary))

        cm: list[list[int]] = confusion_matrix(
            y_true_binary, y_pred_binary, labels=[0, 1]
        ).tolist()

        clf_report: str = classification_report(
            y_true_binary,
            y_pred_binary,
            target_names=["not_ableist", "ableist"],
            zero_division=0,
        )

        # --- Principle accuracy (multi-class) ---
        principle_acc = float(accuracy_score(y_true_principle, y_pred_principle))

        principle_f1 = float(
            f1_score(
                y_true_principle,
                y_pred_principle,
                average="macro",
                zero_division=0,
                labels=["P0", "P1", "P2", "P3", "P4"],
            )
        )

        principle_report: str = classification_report(
            y_true_principle,
            y_pred_principle,
            labels=["P0", "P1", "P2", "P3", "P4"],
            zero_division=0,
        )

        combined_report = (
            f"=== Binary Classification (is_ableist) ===\n{clf_report}\n"
            f"=== Principle Classification (P0..P4) ===\n{principle_report}\n"
            f"Principle Macro F1: {principle_f1:.4f}"
        )

        logger.info(
            f"[{split}] Binary  — F1={f1:.4f}, P={precision:.4f}, "
            f"R={recall:.4f}, Acc={accuracy:.4f}"
        )
        logger.info(
            f"[{split}] Principle — Acc={principle_acc:.4f}, F1={principle_f1:.4f}"
        )

        return EvaluationResult(
            split=split,
            f1=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            principle_accuracy=principle_acc,
            confusion_matrix=cm,
            classification_report=combined_report,
        )
