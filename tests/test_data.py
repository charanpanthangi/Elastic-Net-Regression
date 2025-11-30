from app.data import load_diabetes_data


def test_load_diabetes_data_shapes():
    X, y = load_diabetes_data()
    assert not X.empty
    assert len(X) == len(y)
