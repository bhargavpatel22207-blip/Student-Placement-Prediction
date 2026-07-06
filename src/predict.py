"""
predict.py
----------
Loads the trained model, scaler and feature list, and exposes a simple
`predict_placement` function used both by the Streamlit app and for
programmatic / CLI predictions.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "placement_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def load_artifacts():
    """Load the trained model, scaler, and feature column order from disk.

    Raises:
        FileNotFoundError: If train.py has not been run yet.

    Returns:
        Tuple (model, scaler, feature_columns).
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError(
            "Model artifacts not found. Please run `python src/train.py` "
            "first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns


def predict_placement(input_dict: dict) -> dict:
    """Predict placement outcome for a single student.

    Args:
        input_dict: Dictionary with keys matching the model's expected
            feature columns, e.g.:
                {
                    "CGPA": 8.2,
                    "IQ": 105,
                    "Previous_Semester_Percentage": 78.5,
                    "Communication_Skills": 7.0,
                    "Aptitude_Score": 70.0,
                    "Technical_Skills_Score": 75.0,
                    "Projects_Completed": 3,
                    "Internships": 1,
                    "Certifications": 2,
                    "Attendance_Percentage": 88.0,
                    "Backlogs": 0,
                }

    Returns:
        Dict containing:
            - prediction: "Placed" or "Not Placed"
            - confidence: float, model confidence (0-100)
            - recommendation: str, human-readable suggestion
    """
    model, scaler, feature_columns = load_artifacts()

    row = pd.DataFrame([input_dict], columns=feature_columns)
    row_scaled = scaler.transform(row)

    pred = model.predict(row_scaled)[0]

    # Not every model exposes predict_proba by default, but all four
    # candidates in train.py do. Guard just in case.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        confidence = float(np.max(proba) * 100)
    else:
        confidence = 100.0

    label = "Placed" if pred == 1 else "Not Placed"
    recommendation = generate_recommendation(input_dict, pred)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "recommendation": recommendation,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions for a batch of students (e.g. uploaded CSV).

    Args:
        df: DataFrame containing the required feature columns.

    Returns:
        Copy of the input DataFrame with 'Prediction' and 'Confidence(%)'
        columns appended.
    """
    model, scaler, feature_columns = load_artifacts()

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled).max(axis=1) * 100
    else:
        probas = np.full(len(preds), 100.0)

    result_df = df.copy()
    result_df["Prediction"] = np.where(preds == 1, "Placed", "Not Placed")
    result_df["Confidence(%)"] = probas.round(2)
    return result_df


def generate_recommendation(input_dict: dict, prediction: int) -> str:
    """Generate a simple, rule-based, human-readable recommendation.

    Args:
        input_dict: The student's raw input feature values.
        prediction: 1 if predicted placed, 0 otherwise.

    Returns:
        A short recommendation string.
    """
    tips = []

    if input_dict.get("CGPA", 0) < 7:
        tips.append("work on improving your CGPA")
    if input_dict.get("Communication_Skills", 0) < 6:
        tips.append("practice communication and interview skills")
    if input_dict.get("Internships", 0) == 0:
        tips.append("try to complete at least one internship")
    if input_dict.get("Projects_Completed", 0) < 2:
        tips.append("build more hands-on projects")
    if input_dict.get("Certifications", 0) == 0:
        tips.append("pursue relevant certifications")
    if input_dict.get("Backlogs", 0) > 0:
        tips.append("clear pending academic backlogs")

    if prediction == 1:
        if not tips:
            return "Excellent academic profile. Keep up the strong momentum!"
        return "Strong profile overall. To stay ahead, " + ", ".join(tips) + "."
    else:
        if not tips:
            return ("Your profile is close to the placement threshold. "
                    "Consider more mock interviews and aptitude practice.")
        return "To improve your placement chances, " + ", ".join(tips) + "."


if __name__ == "__main__":
    # Simple manual smoke test
    sample = {
        "CGPA": 8.2,
        "IQ": 108,
        "Previous_Semester_Percentage": 78.5,
        "Communication_Skills": 7.5,
        "Aptitude_Score": 72.0,
        "Technical_Skills_Score": 80.0,
        "Projects_Completed": 4,
        "Internships": 1,
        "Certifications": 2,
        "Attendance_Percentage": 90.0,
        "Backlogs": 0,
    }
    print(predict_placement(sample))
