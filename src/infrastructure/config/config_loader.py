"""OmegaConf-based configuration loader.

Reads ``config/config.yaml``, validates required top-level sections,
and returns a DictConfig instance consumed by the DI container.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.domain.exceptions import ConfigurationError

# Required top-level sections in config.yaml
_REQUIRED_SECTIONS = ("model", "lora", "training", "data", "evaluation", "logging")


def load_config(config_path: Path | None = None) -> DictConfig:
    """Load and validate the OmegaConf configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. Defaults to
                     ``<project_root>/config/config.yaml`` if None.

    Returns:
        A validated OmegaConf DictConfig object.

    Raises:
        ConfigurationError: If the file does not exist, cannot be parsed,
                            or is missing required sections.

    Example:
        >>> cfg = load_config()
        >>> print(cfg.model.name)
        unsloth/Meta-Llama-3-8B-Instruct
    """
    if config_path is None:
        # Assume the project root is two levels above this file:
        # src/infrastructure/config/config_loader.py → project_root/
        config_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"
        )

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}. "
            "Create config/config.yaml before running the project."
        )

    logger.info(f"Loading configuration from {config_path} …")

    try:
        cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore[assignment]
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to parse {config_path}: {exc}"
        ) from exc

    _validate_config(cfg)
    logger.info("Configuration loaded and validated successfully.")
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    """Check that all required top-level sections are present.

    Args:
        cfg: The loaded DictConfig to validate.

    Raises:
        ConfigurationError: If any required section is missing.
    """
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise ConfigurationError(
                f"Configuration is missing required section '{section}'. "
                f"Expected sections: {_REQUIRED_SECTIONS}."
            )
