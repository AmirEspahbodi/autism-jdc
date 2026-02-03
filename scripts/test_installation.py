"""
Test script to validate JDC framework installation and components.

This script tests all major components without requiring GPU or model downloads.

Usage:
    python scripts/test_installation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        import transformers
        import peft
        import trl
        import bitsandbytes
        import datasets
        import pandas
        import sklearn
        from json_repair import repair_json
        
        logger.info("✓ All required packages imported successfully")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - Transformers version: {transformers.__version__}")
        logger.info(f"  - PEFT version: {peft.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("  - CUDA not available (CPU-only mode)")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    logger.info("\nTesting configuration loading...")
    
    try:
        from utils.config import TrainingConfig, EvaluationConfig, load_config_from_yaml
        
        # Test creating configs programmatically
        train_config = TrainingConfig(
            train_data_path="data/train.jsonl",
            model_name="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        
        eval_config = EvaluationConfig(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            adapter_path="./checkpoints/test",
            test_data_path="data/test.jsonl"
        )
        
        logger.info("✓ Configuration schemas working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_json_parser():
    """Test the resilient JSON parser."""
    logger.info("\nTesting JSON parser...")
    
    try:
        from inference.parser import ResilientJSONParser, PrincipleMapper
        
        parser = ResilientJSONParser()
        
        # Test valid JSON
        valid = '{"principle_id": "P1", "justification": "Test justification"}'
        result = parser.parse(valid)
        assert result["principle_id"] == "P1"
        logger.info("✓ Valid JSON parsing works")
        
        # Test JSON in markdown
        markdown = '```json\n{"principle_id": "P2", "justification": "Test"}\n```'
        result = parser.parse(markdown)
        assert result["principle_id"] == "P2"
        logger.info("✓ Markdown extraction works")
        
        # Test malformed JSON (should use fallback)
        malformed = '{"principle_id": "P3", "justification": "Test"'  # Missing closing brace
        result = parser.parse(malformed)
        assert "principle_id" in result  # Should still return something
        logger.info("✓ Fallback mechanism works")
        
        # Test principle mapper
        mapper = PrincipleMapper()
        assert mapper.map_to_label("P0") == 0
        assert mapper.map_to_label("P1") == 1
        assert mapper.map_to_label("P_ERR") == -1
        logger.info("✓ Principle mapping works")
        
        return True
    except Exception as e:
        logger.error(f"✗ JSON parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading and formatting."""
    logger.info("\nTesting dataset loading...")
    
    try:
        from data.dataset import JDCDataset, validate_data_format
        
        # Create sample data
        sample_data = [
            {
                "input_prompt": "Test sentence",
                "target_output": '{"principle_id": "P1", "justification": "Test"}',
                "label": 1
            }
        ]
        
        # Validate format
        validate_data_format(sample_data)
        logger.info("✓ Data validation works")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    logger.info("\nTesting metrics calculation...")
    
    try:
        from utils.metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        
        # Test perfect predictions
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        
        metrics = calculator.compute_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        logger.info("✓ Perfect prediction metrics correct")
        
        # Test with errors
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]  # 2 correct, 2 wrong
        
        metrics = calculator.compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5
        logger.info("✓ Metrics calculation correct")
        
        # Test report formatting
        report = calculator.format_metrics_report(metrics)
        assert "EVALUATION METRICS REPORT" in report
        logger.info("✓ Report formatting works")
        
        return True
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_data():
    """Test that sample data file is valid."""
    logger.info("\nTesting sample data file...")
    
    try:
        sample_path = Path(__file__).parent.parent / "data" / "sample_data.jsonl"
        
        if not sample_path.exists():
            logger.warning("! Sample data file not found (optional)")
            return True
        
        # Load and validate
        with open(sample_path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                
                # Check required fields
                assert "input_prompt" in data, f"Line {i}: missing input_prompt"
                assert "target_output" in data, f"Line {i}: missing target_output"
                
                # Validate target_output is valid JSON
                target = json.loads(data["target_output"])
                assert "principle_id" in target, f"Line {i}: target missing principle_id"
                assert "justification" in target, f"Line {i}: target missing justification"
        
        logger.info("✓ Sample data file is valid")
        return True
    except Exception as e:
        logger.error(f"✗ Sample data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("JDC FRAMEWORK INSTALLATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_config_loading),
        ("JSON Parser", test_json_parser),
        ("Dataset Loading", test_dataset_loading),
        ("Metrics Calculation", test_metrics),
        ("Sample Data", test_sample_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("-" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("=" * 60)
        logger.info("✓ ALL TESTS PASSED - Installation successful!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Prepare your training data (see USAGE_GUIDE.md)")
        logger.info("2. Configure training (configs/train_config.yaml)")
        logger.info("3. Run training: python scripts/train.py --config configs/train_config.yaml")
        return 0
    else:
        logger.error("=" * 60)
        logger.error(f"✗ {total - passed} TESTS FAILED")
        logger.error("=" * 60)
        logger.error("\nPlease fix the failures before proceeding.")
        logger.error("Check that all dependencies are installed: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
