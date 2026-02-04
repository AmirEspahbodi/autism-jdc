"""
Main entry point for the JDC system.

This module demonstrates Clean Architecture principles through dependency
injection, wiring together domain entities, application use cases, and
infrastructure adapters.
"""

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import login

from src.application import EvaluateModelUseCase, FineTuneModelUseCase
from src.config import KnowledgeBaseConfig, SystemConfig

# Updated imports to include real data loaders
from src.infrastructure import (
    ConsoleReportGenerator,
    DetailedReportGenerator,
    HuggingFaceInferenceAdapter,
    LenientJSONParser,
    LoRAAdapter,
    StandardMetricsRepository,
)

# Importing directly from module to ensure access to PreformattedDataLoader
from src.infrastructure.data_loader import PreformattedDataLoader


def check_gpu() -> None:
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(
            f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("⚠ Warning: No GPU detected. Training will be slow.")
        print("  Consider using Google Colab or a GPU-enabled environment.")


def get_training_data_loader(config: SystemConfig):
    """Factory to select the appropriate training data loader."""
    # Strict Policy: Only SFT Pre-formatted dataset (dataset.json) is allowed.
    # Unsafe fallbacks to legacy loaders have been removed to prevent runtime instability.
    sft_path = config.data_dir / "dataset.json"

    if not sft_path.exists():
        print(f"✗ Critical: dataset.json not found at {sft_path}")
        print("  The system strictly requires the SFT dataset. Aborting.")
        sys.exit(1)

    print(f"✓ Found SFT dataset at: {sft_path}")
    return PreformattedDataLoader(train_path=sft_path)


def get_evaluation_data_loader(config: SystemConfig):
    """Factory to select the appropriate evaluation data loader."""
    # Strict Policy: Only SFT Pre-formatted test dataset (test_dataset.json) is allowed.
    sft_test_path = config.data_dir / "test_dataset.json"

    if not sft_test_path.exists():
        print(f"✗ Critical: test_dataset.json not found at {sft_test_path}")
        print("  The system strictly requires the SFT test dataset. Aborting.")
        sys.exit(1)

    print(f"✓ Found SFT test dataset at: {sft_test_path}")
    # We inject the found path as 'test_path'.
    return PreformattedDataLoader(test_path=sft_test_path)


def run_fine_tuning(config: SystemConfig) -> None:
    """Execute the fine-tuning pipeline.

    Args:
        config: System configuration.
    """
    import gc  # Required for memory cleanup

    print("\n" + "=" * 70)
    print("INITIALIZING FINE-TUNING PIPELINE")
    print("=" * 70)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Dependency Injection: Wire up the adapters
    print("\n[Dependency Injection] Wiring adapters...")

    # Infrastructure Layer: Concrete implementations
    trainer = LoRAAdapter(config)

    data_loader = get_training_data_loader(config)

    print(f"✓ Adapters initialized (Loader: {data_loader.__class__.__name__})")

    # Application Layer: Use case
    fine_tune_use_case = FineTuneModelUseCase(
        trainer=trainer,
        data_loader=data_loader,
        config=config,
    )

    result = False

    # Execute
    try:
        # Note: Validation is now safely handled by the loader internally
        fine_tune_use_case.execute(use_validation=True)
        print("\n✓ Fine-tuning completed successfully!")
        result = True
    except Exception as e:
        print(f"\n✗ Fine-tuning failed: {str(e)}")
        # import traceback
        # traceback.print_exc()
        result = False

    # --- MANDATORY GPU CLEANUP BLOCK ---
    print("\n[Cleanup] Releasing GPU resources to prevent OOM in next stage...")

    # 1. Break references to the model
    del fine_tune_use_case
    del trainer

    # 2. Force Python garbage collection
    gc.collect()

    # 3. Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU memory cache cleared.")

    return result


def run_evaluation(config: SystemConfig) -> None:
    """Execute the evaluation pipeline.

    Args:
        config: System configuration.
    """
    print("\n" + "=" * 70)
    print("INITIALIZING EVALUATION PIPELINE")
    print("=" * 70)

    # Check if model exists
    model_path = str(config.output_dir / "fine_tuned_model")
    if not Path(model_path).exists():
        print(f"\n✗ Error: Fine-tuned model not found at {model_path}")
        print("  Please run fine-tuning first: python main.py --mode train")
        return False

    # Dependency Injection: Wire up the adapters
    print("\n[Dependency Injection] Wiring adapters...")

    # Infrastructure Layer: Concrete implementations
    inference_engine = HuggingFaceInferenceAdapter(config, model_path)
    parser = LenientJSONParser()  # Use lenient parser for robustness
    metrics_repo = StandardMetricsRepository()
    report_generator = DetailedReportGenerator()
    console_reporter = ConsoleReportGenerator()
    kb_config = KnowledgeBaseConfig()

    data_loader = get_evaluation_data_loader(config)

    print(f"✓ Adapters initialized (Loader: {data_loader.__class__.__name__})")

    # Application Layer: Use case
    evaluate_use_case = EvaluateModelUseCase(
        inference_engine=inference_engine,
        parser=parser,
        metrics_repo=metrics_repo,
        report_generator=report_generator,
        data_loader=data_loader,
        kb_config=kb_config,
    )

    # Execute
    try:
        output_base = str(config.output_dir / "evaluation")
        metrics = evaluate_use_case.execute(output_base)

        # Also print to console
        console_reporter.generate_report([], metrics)

        print("\n✓ Evaluation completed successfully!")
        print(f"\n📊 Results saved to:")
        print(f"   - Metrics: {output_base}_metrics.json")
        print(f"   - Report:  {output_base}_report.json")
        return True

    except Exception as e:
        print(f"\n✗ Evaluation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def run_full_pipeline(config: SystemConfig) -> None:
    """Run both fine-tuning and evaluation.

    Args:
        config: System configuration.
    """
    check_gpu()

    # Stage 3: Fine-tuning
    success = run_fine_tuning(config)
    if not success:
        print("\n✗ Pipeline aborted due to fine-tuning failure.")
        return

    # Stage 4: Evaluation
    success = run_evaluation(config)
    if not success:
        print("\n✗ Pipeline aborted due to evaluation failure.")
        return

    print("\n" + "=" * 70)
    print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Justification-Driven Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (training + evaluation)
  python main.py --mode full

  # Only fine-tuning
  python main.py --mode train

  # Only evaluation (requires existing model)
  python main.py --mode eval

  # Use Mistral instead of Llama
  python main.py --model mistral

  # Custom output directory
  python main.py --output-dir ./my_results

  # Custom data directory (where dataset.json or train.json is located)
  python main.py --data-dir ./my_data
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "eval"],
        default="full",
        help="Pipeline mode: 'full' (train+eval), 'train' (only training), 'eval' (only evaluation)",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["llama3", "mistral"],
        default="llama3",
        help="Base model to use (default: llama3)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for models and results (default: ./outputs)",
    )

    # Added explicit data-dir argument support
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dataset",
        help="Directory containing dataset.json, train.json, or test.json (default: ./data)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (can also be set via HF_TOKEN env var)",
    )
    args = parser.parse_args()

    if hf_token:
        print("✓ Authenticating with Hugging Face...")
        login(token=hf_token)
    else:
        print(
            "⚠ Warning: No Hugging Face token provided. Gated models (Llama 3) may fail."
        )

    from src.config import ModelType

    config = SystemConfig(
        model_type=ModelType.LLAMA3_8B
        if args.model == "llama3"
        else ModelType.MISTRAL_7B,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        hf_token=args.hf_token,
    )

    # Override hyperparameters from CLI
    config.training_hyperparameters.num_epochs = args.epochs
    config.training_hyperparameters.batch_size = args.batch_size
    config.training_hyperparameters.learning_rate = args.learning_rate

    # Execute based on mode
    if args.mode == "full":
        run_full_pipeline(config)
    elif args.mode == "train":
        check_gpu()
        run_fine_tuning(config)
    elif args.mode == "eval":
        check_gpu()
        run_evaluation(config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
