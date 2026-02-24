"""Train use case: orchestrates data loading, model loading, training, and saving."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.application.ports.model_port import IModelService
from src.application.ports.repository_port import IDatasetRepository
from src.domain.entities import JDCSample


class TrainUseCase:
    """Orchestrates the full supervised fine-tuning workflow.

    This use case coordinates:
      1. Loading training and validation data via the dataset repository.
      2. Loading the base model via the model service.
      3. Running the SFT training loop.
      4. Saving the fine-tuned adapter weights to a checkpoint directory.

    Args:
        repository: Concrete implementation of IDatasetRepository.
        model_service: Concrete implementation of IModelService.
        checkpoint_path: Directory where the trained model is saved.
    """

    def __init__(
        self,
        repository: IDatasetRepository,
        model_service: IModelService,
        checkpoint_path: Path,
    ) -> None:
        self._repository = repository
        self._model_service = model_service
        self._checkpoint_path = checkpoint_path

    def execute(self) -> None:
        """Run the complete training pipeline.

        Steps:
            1. Load training split.
            2. Load validation split.
            3. Load (and quantize) the base model with LoRA adapters.
            4. Train via SFTTrainer.
            5. Save the final model to ``self._checkpoint_path``.

        Raises:
            DatasetLoadError: If training or validation data cannot be read.
            ModelLoadError: If model initialisation fails.
            RuntimeError: If the training loop fails unexpectedly.
        """
        logger.info("TrainUseCase: loading training split …")
        train_data: list[JDCSample] = self._repository.load_train()
        logger.info(f"Loaded {len(train_data)} training samples.")

        logger.info("TrainUseCase: loading validation split …")
        val_data: list[JDCSample] = self._repository.load_validation()
        logger.info(f"Loaded {len(val_data)} validation samples.")

        logger.info("TrainUseCase: loading model …")
        self._model_service.load_model()

        logger.info("TrainUseCase: starting training …")
        self._model_service.train(train_data, val_data)

        logger.info(f"TrainUseCase: saving checkpoint to {self._checkpoint_path} …")
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._model_service.save(self._checkpoint_path)

        logger.info("TrainUseCase: training complete.")
