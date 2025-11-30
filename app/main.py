"""Run the full Elastic Net Regression workflow."""

from pathlib import Path
from typing import Dict

import numpy as np

from app.data import load_diabetes_data
from app.evaluate import evaluate_regression
from app.model import build_elastic_net, predict, train_model
from app.preprocess import scale_features, split_data
from app.visualize import plot_coefficients, plot_predictions


OUTPUT_DIR = Path("outputs")


def run_pipeline() -> Dict[str, float]:
    """Execute the end-to-end pipeline and return metrics."""

    # 1. Load data
    X, y = load_diabetes_data()

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Scale features (critical for fair L1/L2 penalties)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 4. Build and train Elastic Net model
    pipeline = build_elastic_net()
    trained_model = train_model(pipeline, X_train_scaled, y_train)

    # 5. Predict on the test set
    y_pred = predict(trained_model, X_test_scaled)

    # 6. Evaluate performance
    metrics = evaluate_regression(y_test, y_pred)

    # 7. Visualize results
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_predictions(y_test.to_numpy(), y_pred, OUTPUT_DIR / "predictions.svg")
    elastic_step = trained_model.named_steps["elastic"]
    plot_coefficients(elastic_step.coef_, X.columns, OUTPUT_DIR / "coefficients.svg")

    # 8. Print metrics and sparsity info
    non_zero = (elastic_step.coef_ != 0).sum()
    total = elastic_step.coef_.shape[0]
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    print(f"Non-zero coefficients: {non_zero}/{total}")

    return metrics


if __name__ == "__main__":
    run_pipeline()
