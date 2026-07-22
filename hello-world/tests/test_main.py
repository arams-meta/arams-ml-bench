"""Main scoring tests — range-bounded and directional assertions."""
import pandas as pd
import numpy as np


def test_predictions_in_range():
    preds = pd.read_csv("/output/predictions.csv")
    for _, row in preds.iterrows():
        assert 30 <= row["prediction"] <= 60, (
            f"Prediction {row['prediction']} for id {row['id']} outside plausible range [30, 60]"
        )


def test_predictions_monotonic():
    preds = pd.read_csv("/output/predictions.csv")
    values = preds.sort_values("id")["prediction"].values
    for i in range(1, len(values)):
        assert values[i] > values[i - 1], (
            f"Predictions not monotonically increasing: {values[i-1]} >= {values[i]}"
        )


def test_predictions_close_to_expected():
    preds = pd.read_csv("/output/predictions.csv")
    expected = [43.0, 45.0, 47.0, 49.0, 51.0]
    for i, exp in enumerate(expected):
        pred = preds.loc[preds["id"] == i, "prediction"].values[0]
        assert abs(pred - exp) < 2.0, (
            f"Prediction for id {i}: {pred:.2f}, expected ~{exp:.1f} (tolerance 2.0)"
        )
