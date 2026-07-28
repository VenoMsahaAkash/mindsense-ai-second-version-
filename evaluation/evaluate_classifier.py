"""
MindSense AI - Classifier Evaluation
=======================================
Evaluates the mental health classifier on a benchmark dataset.

Computes:
  - Accuracy, Precision, Recall, F1 (macro/weighted)
  - Per-class metrics
  - Confusion matrix
  - Classification report

Usage::

    python evaluation/evaluate_classifier.py
    python evaluation/evaluate_classifier.py --dataset datasets/test.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.classifier.loader import classifier_loader
from evaluation.metrics import evaluation_metrics
from utils.logger import get_logger
from utils.helpers import timer, get_utc_timestamp, safe_json_save

logger = get_logger(__name__)


def load_test_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Load a CSV or JSON test dataset for classifier evaluation.

    Expected CSV format: ``text,label``
    Expected JSON format: list of ``{"text": ..., "label": ...}`` dicts

    Args:
        path: Path to the dataset file.

    Returns:
        Tuple of (texts, labels) lists.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts: List[str] = []
    labels: List[str] = []

    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        import csv
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text") or row.get("statement") or ""
                label = row.get("label") or row.get("status") or ""
                if text and label:
                    texts.append(text.strip())
                    labels.append(label.strip())

    elif suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            text = item.get("text", "")
            label = item.get("label", "")
            if text and label:
                texts.append(text.strip())
                labels.append(label.strip())

    logger.info(f"Loaded {len(texts)} samples from {path}")
    return texts, labels


def compute_confusion_matrix(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Compute a confusion matrix as a nested dict.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: List of unique class labels.

    Returns:
        Nested dict: ``{true_label: {pred_label: count}}``.
    """
    matrix: Dict[str, Dict[str, int]] = {
        l: {l2: 0 for l2 in labels} for l in labels
    }
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix:
            matrix[true][pred] += 1
    return matrix


def compute_f1_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 per class and macro/weighted averages.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: All class labels.

    Returns:
        Dict with per-class and aggregate metrics.
    """
    results = {}
    total_support = len(y_true)

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-10, precision + recall)

        results[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        weight = support / max(1, total_support)
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_p += precision * weight
        weighted_r += recall * weight
        weighted_f1 += f1 * weight

    n = max(1, len(labels))
    results["macro_avg"] = {
        "precision": round(macro_p / n, 4),
        "recall": round(macro_r / n, 4),
        "f1": round(macro_f1 / n, 4),
    }
    results["weighted_avg"] = {
        "precision": round(weighted_p, 4),
        "recall": round(weighted_r, 4),
        "f1": round(weighted_f1, 4),
    }

    return results


@timer
def evaluate_classifier(dataset_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the classifier on the test dataset and compute all metrics.

    Args:
        dataset_path: Path to test CSV/JSON file.
        output_path: Optional path to save results JSON.

    Returns:
        Evaluation results dict.
    """
    logger.info(f"Starting classifier evaluation | dataset={dataset_path}")

    texts, true_labels = load_test_dataset(dataset_path)

    if not texts:
        logger.error("No data loaded from dataset.")
        return {}

    # Run predictions
    logger.info(f"Running predictions on {len(texts)} samples...")
    pred_labels = []
    pred_confidences = []

    for i, text in enumerate(texts):
        try:
            result = classifier_loader.predict(text)
            pred_labels.append(result["label"])
            pred_confidences.append(result["confidence"])
        except Exception as e:
            logger.error(f"Prediction failed for sample {i}: {e}")
            pred_labels.append("Normal")
            pred_confidences.append(0.0)

    # Compute metrics
    all_labels = list(set(true_labels + pred_labels))
    accuracy_metrics = evaluation_metrics.classifier_accuracy(true_labels, pred_labels)
    f1_metrics = compute_f1_metrics(true_labels, pred_labels, all_labels)
    confusion = compute_confusion_matrix(true_labels, pred_labels, all_labels)

    results = {
        "timestamp": get_utc_timestamp(),
        "dataset": dataset_path,
        "n_samples": len(texts),
        "accuracy": accuracy_metrics["accuracy"],
        "per_class_accuracy": accuracy_metrics["per_class"],
        "classification_report": f1_metrics,
        "confusion_matrix": confusion,
        "avg_confidence": round(sum(pred_confidences) / max(1, len(pred_confidences)), 4),
    }

    logger.info(f"Classifier evaluation complete | accuracy={results['accuracy']:.4f}")

    if output_path:
        safe_json_save(results, output_path)
        logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("CLASSIFIER EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(texts)}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {f1_metrics['macro_avg']['f1']:.4f}")
    print(f"Weighted F1: {f1_metrics['weighted_avg']['f1']:.4f}")
    print("\nPer-Class F1 Scores:")
    for cls, m in f1_metrics.items():
        if cls not in ("macro_avg", "weighted_avg"):
            print(f"  {cls:25s}: F1={m['f1']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the MindSense AI mental health classifier")
    parser.add_argument("--dataset", default="datasets/test.csv", help="Path to test dataset")
    parser.add_argument("--output", default="evaluation/results/classifier_results.json")
    args = parser.parse_args()
    evaluate_classifier(args.dataset, args.output)
