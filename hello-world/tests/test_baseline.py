"""Verifies solution beats a trivial baseline (mean prediction)."""
import pandas as pd
import numpy as np


def test_beats_mean_baseline():
    train = pd.read_csv("/app/data/train.csv")
    preds = pd.read_csv("/output/predictions.csv")
    expected = [43.0, 45.0, 47.0, 49.0, 51.0]

    mean_pred = train["y"].mean()
    baseline_mae = np.mean(np.abs(np.array(expected) - mean_pred))

    agent_mae = np.mean(np.abs(preds.sort_values("id")["prediction"].values - np.array(expected)))

    with open("/logs/verifier/baseline_score.txt", "w") as f:
        f.write(f"agent={agent_mae:.3f} baseline={baseline_mae:.3f}\n")

    assert agent_mae < baseline_mae, (
        f"Agent MAE {agent_mae:.3f} did not beat baseline MAE {baseline_mae:.3f}"
    )
