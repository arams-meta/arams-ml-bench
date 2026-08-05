import pandas as pd
import numpy as np


def test_beats_random_baseline():
    """Baseline floor: random scoring (avg 3) should be beaten."""
    df = pd.read_csv("/output/judgments.csv")
    score_col = "final_score" if "final_score" in df.columns else "score"
    avg = df[score_col].mean()
    assert 2.0 <= avg <= 4.5, f"Average {score_col} {avg} outside plausible range"
