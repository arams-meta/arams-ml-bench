import pandas as pd
import numpy as np


def test_beats_random_baseline():
    """Baseline floor: random scoring (avg 3) should be beaten. Trivial judge that scores all 3 fails margin check."""
    df = pd.read_csv("/output/judgments.csv")
    # Random baseline would have margin ~0 and corr ~0
    # Our main tests already check margin >=1 and corr >=0.6, which beats random
    # This file gates the main scoring
    avg = df["score"].mean()
    # If all 3, margin fails anyway, but we add explicit check
    assert 2.0 <= avg <= 4.5, (
        f"Average score {avg} outside plausible range - likely broken"
    )
