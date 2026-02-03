"""
Evaluation metrics for binary classification.

Computes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
"""

import logging
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate and format evaluation metrics.
    
    Handles:
    - Binary classification metrics
    - Error case handling (label = -1)
    - Detailed reporting
    """
    
    def __init__(self, exclude_errors: bool = True):
        """
        Initialize metrics calculator.
        
        Args:
            exclude_errors: If True, exclude examples with label=-1 from metrics
        """
        self.exclude_errors = exclude_errors
    
    def compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[int]
    ) -> Dict[str, Any]:
        """
        Compute all classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with all metrics
            
        Note:
            If exclude_errors=True, examples where either y_true or y_pred
            is -1 (error) are excluded from metric calculation but counted
            separately.
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Count errors
        error_mask = (y_true == -1) | (y_pred == -1)
        num_errors = error_mask.sum()
        
        metrics = {
            "total_examples": len(y_true),
            "num_errors": int(num_errors),
            "error_rate": float(num_errors / len(y_true)) if len(y_true) > 0 else 0.0
        }
        
        # Filter out errors if requested
        if self.exclude_errors and num_errors > 0:
            valid_mask = ~error_mask
            y_true_filtered = y_true[valid_mask]
            y_pred_filtered = y_pred[valid_mask]
            
            logger.info(
                f"Excluded {num_errors} errors from metrics. "
                f"Computing on {len(y_true_filtered)} valid examples."
            )
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred
        
        # Check if we have valid data
        if len(y_true_filtered) == 0:
            logger.warning("No valid examples to compute metrics")
            return metrics
        
        # Accuracy
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        metrics["accuracy"] = float(accuracy)
        
        # Precision, Recall, F1
        # Use 'binary' average for binary classification
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_filtered,
            y_pred_filtered,
            average='binary',
            zero_division=0
        )
        
        metrics["precision"] = float(precision)
        metrics["recall"] = float(recall)
        metrics["f1_score"] = float(f1)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true_filtered,
                y_pred_filtered,
                average=None,
                zero_division=0
            )
        
        # Class 0: Non-Ableist, Class 1: Ableist
        class_names = ["non_ableist", "ableist"]
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f"{class_name}_precision"] = float(precision_per_class[i])
                metrics[f"{class_name}_recall"] = float(recall_per_class[i])
                metrics[f"{class_name}_f1"] = float(f1_per_class[i])
                metrics[f"{class_name}_support"] = int(support_per_class[i])
        
        # Confusion matrix
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)
        
        if cm.shape == (2, 2):
            metrics["confusion_matrix"] = {
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1])
            }
        
        logger.info(
            f"Metrics computed: Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )
        
        return metrics
    
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics into human-readable report.
        
        Args:
            metrics: Metrics dictionary from compute_metrics
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("EVALUATION METRICS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overview
        report.append("OVERVIEW")
        report.append("-" * 60)
        report.append(f"Total Examples:    {metrics['total_examples']}")
        report.append(f"Parsing Errors:    {metrics['num_errors']} "
                     f"({metrics['error_rate']*100:.1f}%)")
        report.append("")
        
        # Main metrics
        report.append("CLASSIFICATION METRICS")
        report.append("-" * 60)
        report.append(f"Accuracy:          {metrics.get('accuracy', 0):.4f}")
        report.append(f"Precision:         {metrics.get('precision', 0):.4f}")
        report.append(f"Recall:            {metrics.get('recall', 0):.4f}")
        report.append(f"F1 Score:          {metrics.get('f1_score', 0):.4f}")
        report.append("")
        
        # Per-class metrics
        if 'non_ableist_precision' in metrics:
            report.append("PER-CLASS METRICS")
            report.append("-" * 60)
            report.append("Non-Ableist (Class 0):")
            report.append(f"  Precision:       {metrics.get('non_ableist_precision', 0):.4f}")
            report.append(f"  Recall:          {metrics.get('non_ableist_recall', 0):.4f}")
            report.append(f"  F1 Score:        {metrics.get('non_ableist_f1', 0):.4f}")
            report.append(f"  Support:         {metrics.get('non_ableist_support', 0)}")
            report.append("")
            report.append("Ableist (Class 1):")
            report.append(f"  Precision:       {metrics.get('ableist_precision', 0):.4f}")
            report.append(f"  Recall:          {metrics.get('ableist_recall', 0):.4f}")
            report.append(f"  F1 Score:        {metrics.get('ableist_f1', 0):.4f}")
            report.append(f"  Support:         {metrics.get('ableist_support', 0)}")
            report.append("")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            report.append("CONFUSION MATRIX")
            report.append("-" * 60)
            report.append("                  Predicted")
            report.append("                  Non-Ab  Ableist")
            report.append(f"Actual  Non-Ab    {cm['true_negatives']:6d}  {cm['false_positives']:6d}")
            report.append(f"        Ableist   {cm['false_negatives']:6d}  {cm['true_positives']:6d}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def compute_metrics_from_results(
    processed_results: Dict[str, Any],
    exclude_errors: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to compute metrics from processed results.
    
    Args:
        processed_results: Output from PredictionPostProcessor
        exclude_errors: Exclude error cases from metrics
        
    Returns:
        Metrics dictionary
    """
    calculator = MetricsCalculator(exclude_errors=exclude_errors)
    
    y_true = processed_results["ground_truth_labels"]
    y_pred = processed_results["predicted_labels"]
    
    return calculator.compute_metrics(y_true, y_pred)
