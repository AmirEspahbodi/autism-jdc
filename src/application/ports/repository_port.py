"""Abstract repository port for dataset access.

Defines the interface that all dataset adapters must implement.
The application layer depends only on this interface, never on a
concrete implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities import JDCSample


class IDatasetRepository(ABC):
    """Port (interface) for loading JDC dataset splits.

    Implementations are responsible for reading raw data files and
    converting them into validated JDCSample objects.
    """

    @abstractmethod
    def load_train(self) -> list[JDCSample]:
        """Load and return all samples from the training split.

        Returns:
            List of validated JDCSample objects for training.

        Raises:
            DatasetLoadError: If the training file cannot be read.
        """
        ...

    @abstractmethod
    def load_validation(self) -> list[JDCSample]:
        """Load and return all samples from the validation split.

        Returns:
            List of validated JDCSample objects for validation.

        Raises:
            DatasetLoadError: If the validation file cannot be read.
        """
        ...

    @abstractmethod
    def load_test(self) -> list[JDCSample]:
        """Load and return all samples from the test split.

        Returns:
            List of validated JDCSample objects for testing.

        Raises:
            DatasetLoadError: If the test file cannot be read.
        """
        ...
