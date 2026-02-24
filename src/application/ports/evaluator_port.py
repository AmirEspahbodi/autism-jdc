"""Abstract evaluator service port.

Defines the interface for computing quantitative evaluation metrics
over a list of predicted and ground-truth JDC samples.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import EvaluationResult, JDCSample
from src.domain.entities import ParsedOutput


class IEvaluatorService(ABC):
    """Port (interface) for evaluation metric computation.

    Implementations receive ground-truth JDCSample objects together with
    model predictions and return a fully populated EvaluationResult.
    """

    @abstractmethod
    def evaluate(
        self,
        samples: list[JDCSample],
        predictions: list[ParsedOutput],
        split: str,
    ) -> EvaluationResult:
        """Compute evaluation metrics for a dataset split.

        Args:
            samples: Ground-truth JDCSample objects (from the dataset).
            predictions: Model-generated ParsedOutput objects, one per sample.
                         Must be the same length and order as ``samples``.
            split: Name of the evaluated split, e.g. "validation" or "test".

        Returns:
            A fully populated EvaluationResult instance.

        Raises:
            ValueError: If ``samples`` and ``predictions`` have different lengths.
        """
        ...
