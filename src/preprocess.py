"""
preprocess.py
-------------
Data cleaning and preprocessing utilities for the Student Placement
Prediction project.

This module is responsible for:
    * Loading the raw dataset
    * Handling missing values
    * Removing duplicate rows
    * Encoding the categorical target column
    * Scaling numerical features
    * Splitting data into train/test sets

All functions are pure (no side effects other than reading/returning data)
so they can be reused by both train.py and any exploratory notebooks.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns used as model features (order matters for the saved scaler/model)
FEATURE_COLUMNS = [
    "CGPA",
    "IQ",
    "Previous_Semester_Percentage",
    "Communication_Skills",
    "Aptitude_Score",
    "Technical_Skills_Score",
    "Projects_Completed",
    "Internships",
    "Certifications",
    "Attendance_Percentage",
    "Backlogs",
]

TARGET_COLUMN = "Placement"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the placement dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file containing raw data.

    Returns:
        A pandas DataFrame with the raw data.
    """
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    Steps performed:
        1. Drop exact duplicate rows.
        2. Fill missing numeric values with the column median.
        3. Ensure the target column has no missing values (rows dropped
           if the target itself is missing, since it cannot be imputed).

    Args:
        df: Raw input DataFrame.

    Returns:
        A cleaned copy of the DataFrame.
    """
    df = df.copy()

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"[preprocess] Removed {before - after} duplicate rows.")

    # Drop rows where the target itself is missing (can't impute a label)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Fill missing numeric feature values with median (robust to outliers)
    for col in FEATURE_COLUMNS:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[preprocess] Filled {n_missing} missing values in "
                  f"'{col}' with median={median_val:.2f}.")

    df.reset_index(drop=True, inplace=True)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Placement target column (Yes/No) into binary (1/0).

    Args:
        df: Cleaned DataFrame containing the TARGET_COLUMN as strings.

    Returns:
        DataFrame with an additional/overwritten numeric target column.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


def split_features_target(df: pd.DataFrame):
    """Split a cleaned, encoded DataFrame into features (X) and target (y).

    Args:
        df: Cleaned & encoded DataFrame.

    Returns:
        Tuple (X, y) where X is a DataFrame of features and y is a Series.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale numerical features using StandardScaler.

    The scaler is fit ONLY on the training data to avoid data leakage,
    then applied to both train and test sets.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        Tuple (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """Run the full preprocessing pipeline end-to-end.

    Args:
        csv_path: Path to the raw CSV dataset.
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing X_train, X_test, y_train, y_test (scaled
        numpy arrays / series) plus the fitted scaler and the cleaned
        DataFrame (useful for EDA).
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df_encoded = encode_target(df)

    X, y = split_features_target(df_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "scaler": scaler,
        "clean_df": df,  # human-readable (Yes/No) cleaned data, for EDA
        "feature_columns": FEATURE_COLUMNS,
    }


if __name__ == "__main__":
    # Quick manual test when running this file directly
    result = preprocess_pipeline("../data/placement.csv")
    print("Train shape:", result["X_train"].shape)
    print("Test shape:", result["X_test"].shape)
