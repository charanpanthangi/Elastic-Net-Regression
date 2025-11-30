"""Data loading utilities for Elastic Net Regression tutorial."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes


def load_diabetes_data(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn diabetes dataset.

    Args:
        as_frame: When True returns pandas DataFrame and Series.

    Returns:
        Tuple containing feature matrix X and target vector y.
    """

    dataset = load_diabetes(as_frame=as_frame)
    X = dataset.data
    y = dataset.target
    return X, y


__all__ = ["load_diabetes_data"]
