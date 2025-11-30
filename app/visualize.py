"""Visualization helpers for Elastic Net Regression results."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | None = None) -> None:
    """Plot true vs predicted values and optionally save as SVG."""

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", label="Predicted vs True")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", label="Ideal fit")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Elastic Net Regression: True vs Predicted")
    plt.legend()
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="svg")
    plt.close()


def plot_coefficients(coefs: Iterable[float], feature_names: Iterable[str], save_path: Path | None = None) -> None:
    """Plot coefficient magnitudes to visualize feature importance."""

    coefs = np.array(list(coefs))
    feature_names = list(feature_names)
    indices = np.argsort(np.abs(coefs))[::-1]

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(coefs)), coefs[indices], color="steelblue")
    plt.xticks(range(len(coefs)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient Value")
    plt.title("Elastic Net Coefficient Magnitudes")
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="svg")
    plt.close()


__all__ = ["plot_predictions", "plot_coefficients"]
