"""
train.py
--------
End-to-end training script for the Student Placement Prediction project.

Workflow:
    1. Load & preprocess the dataset (src/preprocess.py)
    2. Train multiple candidate models
    3. Evaluate & compare them
    4. Select the best model (highest F1 score)
    5. Save the best model + scaler + feature list with Joblib

Run this file from the project root OR from inside src/:
    python src/train.py
"""

from __future__ import annotations

import os
import sys

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Allow running this script both as `python train.py` (from src/) and
# `python src/train.py` (from project root) by fixing up sys.path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_pipeline           # noqa: E402
from utils import (                                  # noqa: E402
    ensure_dir,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    print_metrics,
)

# ----------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ----------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "placement.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")
MODEL_PATH = os.path.join(MODELS_DIR, "placement_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def get_candidate_models() -> dict:
    """Return a dictionary of model_name -> untrained sklearn estimator.

    Returns:
        Dict of candidate classifiers to train and compare.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_and_evaluate() -> None:
    """Train all candidate models, evaluate them, and persist the best one."""
    ensure_dir(MODELS_DIR)
    ensure_dir(SCREENSHOTS_DIR)

    print("Loading and preprocessing data...")
    data = preprocess_pipeline(DATA_PATH)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_columns = data["feature_columns"]

    models = get_candidate_models()
    results = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = print_metrics(y_test, y_pred, model_name=name)
        results[name] = metrics
        fitted_models[name] = model

        # Save a confusion matrix plot per model
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test, y_pred, model_name=name,
            save_path=os.path.join(SCREENSHOTS_DIR, f"confusion_matrix_{safe_name}.png"),
        )

    # Select best model based on F1 score (good balance for imbalanced data)
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]

    print(f"\n{'#' * 60}")
    print(f"Best model: {best_name} (F1 = {results[best_name]['f1']:.4f})")
    print(f"{'#' * 60}")

    # Save comparison chart & feature importance for the best model
    plot_model_comparison(results, os.path.join(SCREENSHOTS_DIR, "model_comparison.png"))
    plot_feature_importance(
        best_model, feature_columns,
        os.path.join(SCREENSHOTS_DIR, "feature_importance.png"),
        model_name=best_name,
    )

    # Persist best model, scaler, and feature column order
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(data["scaler"], SCALER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    print(f"\nSaved best model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved feature columns to: {FEATURES_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
