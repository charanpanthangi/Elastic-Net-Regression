import numpy as np

from app.evaluate import evaluate_regression


def test_evaluate_returns_numbers():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    metrics = evaluate_regression(y_true, y_pred)

    expected_keys = {"MSE", "MAE", "RMSE", "R2"}
    assert expected_keys == set(metrics.keys())
    assert all(isinstance(value, float) for value in metrics.values())
