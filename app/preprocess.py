"""Preprocessing utilities for Elastic Net Regression."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data to use for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of X_train, X_test, y_train, y_test.
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler.

    Elastic Net is sensitive to feature scale because the L1 and L2 penalties
    depend on coefficient magnitudes. Scaling ensures each feature contributes
    fairly to the penalty terms.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        Scaled X_train, scaled X_test, fitted scaler instance.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


__all__ = ["split_data", "scale_features"]
