"""
utils.py
--------
Shared helper functions used across the project: plotting, metrics
printing, and simple file-path helpers. Keeping these in one place
avoids duplicated code between train.py, predict.py and app.py.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sns.set_theme(style="whitegrid")


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def print_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute and pretty-print classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Name to display in the printed report.

    Returns:
        Dictionary of the computed metric values.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"Metrics for: {model_name}")
    print(f"{'=' * 50}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Title label for the plot.
        save_path: File path to save the resulting PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results: dict, save_path: str) -> None:
    """Plot a bar chart comparing accuracy across multiple models.

    Args:
        results: Dict mapping model_name -> metrics dict (must contain
            an 'accuracy' key).
        save_path: File path to save the resulting PNG.
    """
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, accuracies, color=sns.color_palette("viridis", len(names)))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                  f"{acc:.2%}", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path: str,
                             model_name: str = "Model") -> None:
    """Plot feature importance for tree-based models, or coefficient
    magnitude for linear models like Logistic Regression.

    Args:
        model: A fitted sklearn estimator.
        feature_names: List of feature column names.
        save_path: File path to save the resulting PNG.
        model_name: Title label for the plot.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        print(f"[utils] {model_name} does not support feature importance.")
        return

    order = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features,
                hue=sorted_features, palette="crest", legend=False)
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
