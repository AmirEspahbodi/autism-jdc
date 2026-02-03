"""
Infrastructure layer - Concrete implementations and adapters.

This package contains all the external framework dependencies and
concrete implementations of the domain interfaces.
"""

from src.infrastructure.data_loader import FileBasedDataLoader, MockDataLoader
from src.infrastructure.llm import (
    HuggingFaceInferenceAdapter,
    LoRAAdapter,
    PromptTemplate,
)
from src.infrastructure.metrics import (
    ConsoleReportGenerator,
    DetailedReportGenerator,
    StandardMetricsRepository,
)
from src.infrastructure.parsing import (
    LenientJSONParser,
    RobustJSONParser,
    StrictJSONParser,
)

__all__ = [
    # Data Loaders
    "FileBasedDataLoader",
    "MockDataLoader",
    # LLM Adapters
    "HuggingFaceInferenceAdapter",
    "LoRAAdapter",
    "PromptTemplate",
    # Metrics and Reporting
    "ConsoleReportGenerator",
    "DetailedReportGenerator",
    "StandardMetricsRepository",
    # Parsing
    "LenientJSONParser",
    "RobustJSONParser",
    "StrictJSONParser",
]
