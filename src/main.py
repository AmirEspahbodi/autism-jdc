# CRITICAL: unsloth must be imported before transformers (or any library that
# triggers a transformers import) so that all kernel patches are applied.
# Do NOT move this import below any other ML library import.
from __future__ import annotations

import unsloth

"""CLI entry point for the JDC project.

Commands:
    train    — Run supervised fine-tuning.
    evaluate — Run inference + metric computation on validation and/or test splits.
    infer    — Run single-sample inference from a prompt file or stdin.

CUDA availability is asserted before any other operation.
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from src.container import Container
from src.domain.exceptions import (
    ConfigurationError,
    CUDANotAvailableError,
    JDCException,
)


def _assert_cuda() -> None:
    """Assert CUDA is available; raise CUDANotAvailableError if not.

    Raises:
        CUDANotAvailableError: If no CUDA-capable GPU is detected.
    """
    if not torch.cuda.is_available():
        raise CUDANotAvailableError(
            "No CUDA device found. JDC requires an NVIDIA GPU with CUDA 12.6+. "
            f"torch version: {torch.__version__}. "
            "Set CUDA_VISIBLE_DEVICES appropriately and retry."
        )
    logger.info(f"CUDA OK. Device: {torch.cuda.get_device_name(0)}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="jdc",
        description=(
            "Justification-Driven Classification (JDC) — Training, Evaluation, and Inference CLI"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    subparsers.add_parser(
        "train",
        help="Fine-tune the model on the training split.",
    )

    # --- evaluate subcommand ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run inference and compute metrics on validation/test splits.",
    )
    eval_parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        choices=["validation", "test"],
        help="Which splits to evaluate (default: both validation and test).",
    )

    # --- infer subcommand ---
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run single-sample inference on a prompt.",
    )
    infer_input = infer_parser.add_mutually_exclusive_group(required=True)
    infer_input.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing the input_prompt.",
    )
    infer_input.add_argument(
        "--prompt",
        type=str,
        help="The input_prompt string directly on the command line.",
    )

    return parser


def _cmd_train(container: Container) -> None:
    """Execute the training use case.

    Args:
        container: Wired DI container.
    """
    logger.info("Command: TRAIN")
    use_case = container.get_train_use_case()
    use_case.execute()
    logger.info("Training pipeline completed successfully.")


def _cmd_evaluate(container: Container, splits: list[str]) -> None:
    """Execute the evaluation use case.

    Args:
        container: Wired DI container.
        splits: List of split names to evaluate.
    """
    logger.info(f"Command: EVALUATE — splits={splits}")
    use_case = container.get_evaluate_use_case()
    results = use_case.execute(splits=splits)
    for split, result in results.items():
        print(f"\n{'=' * 60}")
        print(f"SPLIT: {split.upper()}")
        print(f"  F1           : {result.f1:.4f}")
        print(f"  Precision    : {result.precision:.4f}")
        print(f"  Recall       : {result.recall:.4f}")
        print(f"  Accuracy     : {result.accuracy:.4f}")
        print(f"  PrincipleAcc : {result.principle_accuracy:.4f}")
        print(f"{'=' * 60}\n")
    logger.info("Evaluation pipeline completed successfully.")


def _cmd_infer(
    container: Container,
    prompt_file: Path | None,
    prompt: str | None,
) -> None:
    """Execute single-sample inference.

    Args:
        container: Wired DI container.
        prompt_file: Optional path to a file containing the prompt.
        prompt: Optional inline prompt string.
    """
    logger.info("Command: INFER")

    if prompt_file is not None:
        if not prompt_file.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            sys.exit(1)
        input_prompt = prompt_file.read_text(encoding="utf-8").strip()
    elif prompt is not None:
        input_prompt = prompt.strip()
    else:
        logger.error("No prompt provided.")
        sys.exit(1)

    use_case = container.get_inference_use_case()
    result = use_case.execute(input_prompt)

    print("\n=== INFERENCE RESULT ===")
    print(f"  principle_id            : {result.principle_id}")
    print(f"  principle_name          : {result.principle_name}")
    print(f"  is_ableist (derived)    : {result.is_ableist}")
    print(f"  evidence_quote          : {result.evidence_quote!r}")
    print(f"  justification_reasoning : {result.justification_reasoning!r}")
    print("========================\n")


def main() -> None:
    """Main CLI entry point.

    Parses arguments, validates CUDA, wires the container, and dispatches
    to the appropriate command handler.

    Raises:
        SystemExit: On validation failure or unrecoverable error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # CUDA must be available before any model operations
    try:
        _assert_cuda()
    except CUDANotAvailableError as exc:
        logger.critical(str(exc))
        sys.exit(1)

    # Wire the DI container
    try:
        container = Container(config_path=args.config)
        container.wire()
    except ConfigurationError as exc:
        logger.critical(f"Configuration error: {exc}")
        sys.exit(1)
    except JDCException as exc:
        logger.critical(f"Startup error: {exc}")
        sys.exit(1)

    # Dispatch command
    try:
        if args.command == "train":
            _cmd_train(container)
        elif args.command == "evaluate":
            _cmd_evaluate(container, splits=args.splits)
        elif args.command == "infer":
            _cmd_infer(
                container,
                prompt_file=getattr(args, "prompt_file", None),
                prompt=getattr(args, "prompt", None),
            )
        else:
            parser.print_help()
            sys.exit(1)
    except JDCException as exc:
        logger.critical(f"Fatal JDC error: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
