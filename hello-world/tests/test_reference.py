"""Verifies output matches known-correct reference on toy input."""
import json
import numpy as np
from sklearn.linear_model import LinearRegression


def test_linear_model_on_toy_input():
    with open("/tests/reference/toy_input.json") as f:
        toy = json.load(f)
    with open("/tests/reference/expected_output.json") as f:
        expected = json.load(f)

    X = np.array(toy["x"]).reshape(-1, 1)
    y = np.array(toy["y"])
    model = LinearRegression()
    model.fit(X, y)

    test_x = np.array([4, 5, 6]).reshape(-1, 1)
    preds = model.predict(test_x)
    expected_preds = np.array(expected["predictions"])

    np.testing.assert_allclose(preds, expected_preds, atol=0.5)
