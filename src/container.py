"""Dependency injection container for the JDC project.

Manually wires concrete implementations to abstract interfaces and
assembles the three use cases.  No DI framework is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from src.application.use_cases.evaluate_use_case import EvaluateUseCase
from src.application.use_cases.inference_use_case import InferenceUseCase
from src.application.use_cases.train_use_case import TrainUseCase
from src.domain.exceptions import ConfigurationError
from src.infrastructure.adapters.json_dataset_repository import JsonDatasetRepository
from src.infrastructure.adapters.sklearn_evaluator_service import SklearnEvaluatorService
from src.infrastructure.adapters.unsloth_model_service import UnslothModelService
from src.infrastructure.config.config_loader import load_config


class Container:
    """Central wiring point that assembles the full application.

    Usage::

        container = Container()
        container.wire()
        use_case = container.get_train_use_case()
        use_case.execute()

    Args:
        config_path: Optional explicit path to config.yaml.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path
        self._config: DictConfig | None = None
        self._repository: JsonDatasetRepository | None = None
        self._model_service: UnslothModelService | None = None
        self._evaluator_service: SklearnEvaluatorService | None = None

    def wire(self) -> None:
        """Load config, configure loguru, and instantiate all services.

        Raises:
            ConfigurationError: If config loading or validation fails.
        """
        self._config = load_config(self._config_path)
        self._configure_logging(self._config)
        self._repository = JsonDatasetRepository(self._config)
        self._model_service = UnslothModelService(self._config)
        self._evaluator_service = SklearnEvaluatorService()

    def get_train_use_case(self) -> TrainUseCase:
        """Construct and return the TrainUseCase.

        Returns:
            Fully wired TrainUseCase instance.

        Raises:
            ConfigurationError: If wire() has not been called.
        """
        self._check_wired()
        assert self._config is not None
        checkpoint_path = Path(str(self._config.training.output_dir))
        return TrainUseCase(
            repository=self._repository,  # type: ignore[arg-type]
            model_service=self._model_service,  # type: ignore[arg-type]
            checkpoint_path=checkpoint_path,
        )

    def get_evaluate_use_case(self) -> EvaluateUseCase:
        """Construct and return the EvaluateUseCase.

        Returns:
            Fully wired EvaluateUseCase instance.

        Raises:
            ConfigurationError: If wire() has not been called.
        """
        self._check_wired()
        assert self._config is not None
        checkpoint_path = Path(str(self._config.evaluation.checkpoint_path))
        output_dir = Path(str(self._config.evaluation.output_dir))
        return EvaluateUseCase(
            repository=self._repository,  # type: ignore[arg-type]
            model_service=self._model_service,  # type: ignore[arg-type]
            evaluator_service=self._evaluator_service,  # type: ignore[arg-type]
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )

    def get_inference_use_case(self) -> InferenceUseCase:
        """Construct and return the InferenceUseCase.

        Returns:
            Fully wired InferenceUseCase instance.

        Raises:
            ConfigurationError: If wire() has not been called.
        """
        self._check_wired()
        assert self._config is not None
        checkpoint_path = Path(str(self._config.evaluation.checkpoint_path))
        return InferenceUseCase(
            model_service=self._model_service,  # type: ignore[arg-type]
            checkpoint_path=checkpoint_path,
        )

    def _check_wired(self) -> None:
        """Raise ConfigurationError if wire() has not been called.

        Raises:
            ConfigurationError: If container is not wired.
        """
        if self._config is None:
            raise ConfigurationError(
                "Container.wire() must be called before accessing use cases."
            )

    @staticmethod
    def _configure_logging(config: DictConfig) -> None:
        """Configure loguru with file and stderr sinks.

        Removes the default loguru handler and adds:
          - A stderr sink at the configured log level.
          - A rotating file sink writing to the configured log file.

        Args:
            config: The full application DictConfig.
        """
        logger.remove()  # Remove default handler

        log_level = str(config.logging.level)
        log_file = Path(str(config.logging.log_file))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Stderr sink (human-readable with colors)
        logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
            ),
            colorize=True,
        )

        # File sink (structured, rotation every 50 MB)
        logger.add(
            str(log_file),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
            rotation="50 MB",
            retention="14 days",
            enqueue=True,  # Thread-safe async writing
        )

        logger.info(
            f"Logging configured. Level={log_level}, File={log_file}"
        )
