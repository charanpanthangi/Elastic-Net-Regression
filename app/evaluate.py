"""Evaluation metrics for Elastic Net Regression."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics.

    Args:
        y_true: Ground truth targets.
        y_pred: Model predictions.

    Returns:
        Dictionary containing MSE, MAE, RMSE, and R2.
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}


__all__ = ["evaluate_regression"]
