"""
Domain layer - Core business logic and entities.

This package contains the pure domain logic with no dependencies
on external frameworks or libraries.
"""

from src.domain.interfaces import (
    DataLoader,
    DataLoadError,
    InferenceEngine,
    InferenceError,
    JustificationParser,
    LLMTrainer,
    MetricsRepository,
    ParsingError,
    ReportGenerator,
    TrainingError,
)
from src.domain.types import (
    EvaluationMetrics,
    Justification,
    LabeledExample,
    PredictionResult,
    Principle,
)

__all__ = [
    # Interfaces
    "DataLoader",
    "InferenceEngine",
    "JustificationParser",
    "LLMTrainer",
    "MetricsRepository",
    "ReportGenerator",
    # Types
    "EvaluationMetrics",
    "Justification",
    "LabeledExample",
    "PredictionResult",
    "Principle",
    # Exceptions
    "DataLoadError",
    "InferenceError",
    "ParsingError",
    "TrainingError",
]
