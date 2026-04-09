#!/usr/bin/env python3
"""
Classifier Validation Script - IMPROVEMENT 4

Validates zero-shot and baseline classifiers against a manually 
annotated gold set using accuracy, Cohen's kappa, precision, recall, F1, 
and confusion matrices.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import jsonlines

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from annotate import classify_safety

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
GOLD_SET_FILE = DATA_DIR / "gold_set.json"
OUTPUT_FILE = RESULTS_DIR / "classifier_validation.json"

# Label mapping for consistency
LABEL_MAPPING = {
    "safe": "safe",
    "unsafe": "unsafe",
    "partial": "partial",
    "warning": "partial",  # Map "warning" to "partial" for consistency
}

# Classes for multi-class metrics
CLASSES = ["safe", "unsafe", "partial"]


def load_gold_set(file_path: Path) -> List[Dict]:
    """Load manually annotated gold set.
    
    Args:
        file_path (Path): Path to gold_set.json.
    
    Returns:
        List[Dict]: Gold set entries with output_text and human_label.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Gold set not found: {file_path}")
    
    with open(file_path, "r") as f:
        gold_set = json.load(f)
    
    logger.info(f"Loaded {len(gold_set)} gold set examples")
    return gold_set


def classify_with_both_methods(
    output_texts: List[str]
) -> Tuple[List[str], List[str]]:
    """Classify outputs using both zero-shot and baseline methods.
    
    Args:
        output_texts (List[str]): List of output texts to classify.
    
    Returns:
        Tuple[List[str], List[str]]: (zeroshot_labels, baseline_labels)
    """
    zeroshot_labels = []
    baseline_labels = []
    
    for text in output_texts:
        # Zero-shot classification
        try:
            zs_label, zs_score = classify_safety(
                text,
                classifier_type="zeroshot"
            )
            zeroshot_labels.append(LABEL_MAPPING.get(zs_label, "safe"))
        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")
            zeroshot_labels.append("safe")  # Default to safe
        
        # Baseline classification
        try:
            bl_label, bl_score = classify_safety(
                text,
                classifier_type="baseline"
            )
            baseline_labels.append(LABEL_MAPPING.get(bl_label, "safe"))
        except Exception as e:
            logger.warning(f"Baseline classification failed: {e}")
            baseline_labels.append("safe")  # Default to safe
    
    return zeroshot_labels, baseline_labels


def compute_metrics(
    ground_truth: List[str],
    predictions: List[str],
    classifier_name: str
) -> Dict:
    """Compute comprehensive metrics for a classifier.
    
    Args:
        ground_truth (List[str]): Human-annotated labels.
        predictions (List[str]): Classifier predictions.
        classifier_name (str): Name of classifier (for logging).
    
    Returns:
        Dict: Metrics including accuracy, kappa, per-class metrics.
    
    Example:
        >>> metrics = compute_metrics(['safe', 'unsafe'], ['safe', 'safe'], 'zershot')
        >>> metrics['accuracy']
        0.5
    """
    # Ensure labels match
    assert len(ground_truth) == len(predictions), \
        f"Length mismatch: {len(ground_truth)} != {len(predictions)}"
    
    # Convert to numeric for sklearn
    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    gt_numeric = np.array([label_to_idx[label] for label in ground_truth])
    pred_numeric = np.array([label_to_idx[label] for label in predictions])
    
    # Overall metrics
    accuracy = accuracy_score(ground_truth, predictions)
    kappa = cohen_kappa_score(ground_truth, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=CLASSES,
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(
        ground_truth,
        predictions,
        labels=CLASSES
    )
    
    # Build results dict
    metrics = {
        "classifier": classifier_name,
        "accuracy": float(accuracy),
        "cohen_kappa": float(kappa),
        "per_class_metrics": {},
        "confusion_matrix": conf_matrix.tolist(),
        "total_samples": len(ground_truth),
    }
    
    # Add per-class metrics
    for label, prec, rec, f1_score, supp in zip(
        CLASSES, precision, recall, f1, support
    ):
        metrics["per_class_metrics"][label] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_score),
            "support": int(supp),
        }
    
    # Log detailed report
    logger.debug(f"\n{classifier_name} Classification Report:")
    logger.debug(classification_report(
        ground_truth,
        predictions,
        labels=CLASSES,
        zero_division=0
    ))
    
    return metrics


def compute_agreement_metrics(
    ground_truth: List[str],
    zeroshot_preds: List[str],
    baseline_preds: List[str]
) -> Dict:
    """Compute inter-rater agreement between classifiers.
    
    Args:
        ground_truth (List[str]): Human-annotated labels.
        zeroshot_preds (List[str]): Zero-shot predictions.
        baseline_preds (List[str]): Baseline predictions.
    
    Returns:
        Dict: Agreement metrics between classifiers and humans.
    """
    # Agreement with humans
    zeroshot_vs_human = cohen_kappa_score(ground_truth, zeroshot_preds)
    baseline_vs_human = cohen_kappa_score(ground_truth, baseline_preds)
    
    # Agreement between classifiers
    classifier_agreement = cohen_kappa_score(zeroshot_preds, baseline_preds)
    
    return {
        "zeroshot_vs_human": float(zeroshot_vs_human),
        "baseline_vs_human": float(baseline_vs_human),
        "zeroshot_vs_baseline": float(classifier_agreement),
    }


def print_comparison_table(
    metrics_zeroshot: Dict,
    metrics_baseline: Dict,
    agreement: Dict
) -> None:
    """Print formatted comparison table.
    
    Args:
        metrics_zeroshot (Dict): Zero-shot classifier metrics.
        metrics_baseline (Dict): Baseline classifier metrics.
        agreement (Dict): Agreement metrics.
    """
    print("\n" + "="*70)
    print("CLASSIFIER VALIDATION RESULTS")
    print("="*70)
    
    print("\n📊 OVERALL PERFORMANCE")
    print("-"*70)
    print(f"{'Metric':<25} {'Zero-Shot':<20} {'Baseline':<20}")
    print("-"*70)
    
    print(f"{'Accuracy':<25} {metrics_zeroshot['accuracy']:<20.4f} {metrics_baseline['accuracy']:<20.4f}")
    print(f"{'Cohen\'s Kappa':<25} {metrics_zeroshot['cohen_kappa']:<20.4f} {metrics_baseline['cohen_kappa']:<20.4f}")
    
    print("\n📈 PER-CLASS METRICS (ZERO-SHOT)")
    print("-"*70)
    print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
    print("-"*70)
    for label in CLASSES:
        metrics = metrics_zeroshot["per_class_metrics"][label]
        print(f"{label:<15} {metrics['precision']:<15.4f} {metrics['recall']:<15.4f} {metrics['f1']:<15.4f}")
    
    print("\n📈 PER-CLASS METRICS (BASELINE)")
    print("-"*70)
    print(f"{'Class':<15} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
    print("-"*70)
    for label in CLASSES:
        metrics = metrics_baseline["per_class_metrics"][label]
        print(f"{label:<15} {metrics['precision']:<15.4f} {metrics['recall']:<15.4f} {metrics['f1']:<15.4f}")
    
    print("\n🤝 AGREEMENT METRICS")
    print("-"*70)
    print(f"Zero-Shot vs Human:        {agreement['zeroshot_vs_human']:<20.4f}")
    print(f"Baseline vs Human:         {agreement['baseline_vs_human']:<20.4f}")
    print(f"Zero-Shot vs Baseline:     {agreement['zeroshot_vs_baseline']:<20.4f}")
    
    print("\n🔲 CONFUSION MATRICES")
    print("-"*70)
    
    print("\nZero-Shot Confusion Matrix (rows=actual, cols=predicted):")
    print(f"          {str(CLASSES)}")
    for i, label in enumerate(CLASSES):
        print(f"{label:<10} {metrics_zeroshot['confusion_matrix'][i]}")
    
    print("\nBaseline Confusion Matrix (rows=actual, cols=predicted):")
    print(f"          {str(CLASSES)}")
    for i, label in enumerate(CLASSES):
        print(f"{label:<10} {metrics_baseline['confusion_matrix'][i]}")
    
    print("\n" + "="*70)
    
    # Interpretation guide
    print("\n📖 INTERPRETATION GUIDE")
    print("-"*70)
    print("Accuracy: Proportion of correct predictions (higher is better)")
    print("Cohen's Kappa: Agreement corrected for chance (0-1, 0.41-0.60=moderate)")
    print("Precision: % of predicted class that are actually correct")
    print("Recall: % of actual class that were correctly identified")
    print("F1: Harmonic mean of precision and recall (balanced metric)")
    print("="*70 + "\n")


def main():
    """Main validation pipeline."""
    logger.info("\n" + "="*70)
    logger.info("IMPROVEMENT 4: Classifier Validation")
    logger.info("="*70 + "\n")
    
    # Load gold set
    try:
        gold_set = load_gold_set(GOLD_SET_FILE)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Please create {GOLD_SET_FILE} with gold standard annotations")
        return
    
    # Extract components
    output_texts = [item["output_text"] for item in gold_set]
    human_labels = [item["human_label"] for item in gold_set]
    
    # Normalize human labels
    human_labels = [LABEL_MAPPING.get(label, "safe") for label in human_labels]
    
    logger.info(f"Processing {len(gold_set)} gold set examples...")
    
    # Run both classifiers
    logger.info("Running zero-shot classifier...")
    logger.info("Running baseline classifier...")
    zeroshot_labels, baseline_labels = classify_with_both_methods(output_texts)
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics_zeroshot = compute_metrics(human_labels, zeroshot_labels, "Zero-Shot")
    metrics_baseline = compute_metrics(human_labels, baseline_labels, "Baseline")
    agreement = compute_agreement_metrics(human_labels, zeroshot_labels, baseline_labels)
    
    # Print comparison table
    print_comparison_table(metrics_zeroshot, metrics_baseline, agreement)
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "validation_date": str(Path(__file__).stat().st_mtime),
        "gold_set_size": len(gold_set),
        "zero_shot": metrics_zeroshot,
        "baseline": metrics_baseline,
        "agreement_metrics": agreement,
        "class_labels": CLASSES,
        "gold_set_file": str(GOLD_SET_FILE),
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Validation results saved to: {OUTPUT_FILE}")
    logger.info("="*70 + "\n")
    
    # Summary
    better_classifier = "Zero-Shot" if metrics_zeroshot['accuracy'] > metrics_baseline['accuracy'] else "Baseline"
    improvement = abs(metrics_zeroshot['accuracy'] - metrics_baseline['accuracy']) * 100
    
    logger.info(f"Summary:")
    logger.info(f"  ✓ {better_classifier} classifier performs better (+{improvement:.1f}%)")
    logger.info(f"  ✓ Zero-Shot accuracy: {metrics_zeroshot['accuracy']:.1%}")
    logger.info(f"  ✓ Baseline accuracy: {metrics_baseline['accuracy']:.1%}")
    logger.info(f"  ✓ Inter-classifier agreement: {agreement['zeroshot_vs_baseline']:.4f}")


if __name__ == "__main__":
    main()
