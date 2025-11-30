import numpy as np

from app.data import load_diabetes_data
from app.model import build_elastic_net, predict, train_model
from app.preprocess import scale_features, split_data


def test_model_fit_and_predict():
    X, y = load_diabetes_data()
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.2, random_state=0)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    model = build_elastic_net()
    trained = train_model(model, X_train_scaled, y_train)
    preds = predict(trained, X_test_scaled)

    assert preds.shape[0] == X_test.shape[0]
    assert np.isfinite(preds).all()
