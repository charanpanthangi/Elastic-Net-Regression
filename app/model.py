"""Model utilities for training Elastic Net Regression."""

from typing import Tuple

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_elastic_net(alpha: float = 0.1, l1_ratio: float = 0.5, max_iter: int = 5000) -> Pipeline:
    """Create a pipeline with scaling and Elastic Net regression.

    The `alpha` parameter controls the overall strength of regularization.
    Larger values shrink coefficients more. The `l1_ratio` controls the mix
    between L1 (feature selection) and L2 (shrinkage) penalties.

    Args:
        alpha: Regularization strength (0 means no regularization).
        l1_ratio: Balance between L1 and L2. 0 -> Ridge, 1 -> Lasso.
        max_iter: Maximum iterations for convergence.

    Returns:
        A scikit-learn Pipeline ready for training.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "elastic",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42),
            ),
        ]
    )


def train_model(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Fit the Elastic Net pipeline."""

    pipeline.fit(X_train, y_train)
    return pipeline


def predict(pipeline: Pipeline, X_test: np.ndarray) -> np.ndarray:
    """Generate predictions using the trained pipeline."""

    return pipeline.predict(X_test)


__all__ = ["build_elastic_net", "train_model", "predict"]
