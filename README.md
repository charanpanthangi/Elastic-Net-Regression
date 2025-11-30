# Elastic Net Regression Tutorial & Template

This repository is a beginner-friendly, end-to-end template for training and evaluating an **Elastic Net Regression** model on the scikit-learn **diabetes** dataset. It shows how to combine **L1 (Lasso)** and **L2 (Ridge)** regularization to handle multicollinearity while performing light feature selection.

## Why Elastic Net?
- **Blends Lasso + Ridge:** `l1_ratio` sets the mix of L1 (pushes some coefficients exactly to zero) and L2 (shrinks coefficients smoothly). `alpha` scales the overall strength of both penalties.
- **Handles multicollinearity:** Groups of correlated features are handled more gracefully than Lasso alone.
- **Feature selection:** With a higher `l1_ratio`, Elastic Net can zero-out less useful coefficients.
- **When it shines:** When you have many correlated predictors and want a balance between Ridge stability and Lasso sparsity.

## How it works (step-by-step)
1. **Load data:** Use scikit-learn's diabetes dataset.
2. **Split:** Create train/test sets to estimate generalization.
3. **Scale:** Apply `StandardScaler` because Elastic Net is sensitive to feature scales.
4. **Model:** Fit `ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)` inside a pipeline.
5. **Evaluate:** Report MSE, MAE, RMSE, and R².
6. **Visualize:** Optional SVG plots for true vs. predicted values and coefficient magnitudes.

## Project structure
```
app/
  data.py          # Load the diabetes dataset
  preprocess.py    # Train/test split and scaling
  model.py         # Build, train, and predict with Elastic Net
  evaluate.py      # Regression metrics
  visualize.py     # Plot predictions and coefficients
  main.py          # Runs the full pipeline
notebooks/
  demo_elastic_net_regression.ipynb  # Walkthrough notebook
examples/
  README_examples.md  # Quick usage notes
tests/
  test_data.py, test_model.py, test_evaluate.py  # Pytest-based checks
```

## Running the project
### 1) Local Python
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app/main.py
```

### 2) Jupyter Notebook
```bash
jupyter notebook notebooks/demo_elastic_net_regression.ipynb
```

### 3) Docker
```bash
docker build -t elastic-net-demo .
docker run --rm elastic-net-demo
```

## Notes on hyperparameters
- `alpha` (strength): higher values mean stronger combined regularization.
- `l1_ratio` (mix): `0` -> Ridge-like, `1` -> Lasso-like. Values in-between give Elastic Net.
- `max_iter`: increased to 5000 to ensure convergence on small datasets.

## Future improvements
- Add **ElasticNetCV** for automated hyperparameter tuning.
- Compare **Ridge vs. Lasso vs. Elastic Net** side by side.
- Add a simple **hyperparameter search** (GridSearchCV or RandomizedSearchCV).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
