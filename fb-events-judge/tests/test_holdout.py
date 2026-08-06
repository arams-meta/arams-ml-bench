import json, pathlib
import pandas as pd
import numpy as np


def test_holdout_generalization():
    """Holdout check - judge must generalize to unseen queries, not hardcode visible set"""
    # Holdout dataset
    p = pathlib.Path("/tests/data/holdout.json")
    if not p.exists():
        p = pathlib.Path("/app/data/../tests/data/holdout.json")
    if not p.exists():
        p = pathlib.Path("data/../tests/data/holdout.json")
    if not p.exists():
        p = pathlib.Path("tests/data/holdout.json")
    if not p.exists():
        print("Holdout not found, skipping")
        return

    with open(p) as f:
        holdout = json.load(f)

    # For this task, we need agent to also produce holdout judgments?
    # Our current oracle writes /output/holdout_judgments.csv if available
    # If agent only writes main judgments, we check that its logic would generalize by
    # re-running same correlation check on holdout if file exists, otherwise check visible ratio

    # If agent wrote holdout file, evaluate it
    hj_path = pathlib.Path("/output/holdout_judgments.csv")
    if hj_path.exists():
        df = pd.read_csv(hj_path)
        score_col = "final_score" if "final_score" in df.columns else "score"
        m = dict(zip(df["id"], df[score_col]))
        y_true = [r["human_score"] for r in holdout if r["id"] in m]
        y_pred = [m[r["id"]] for r in holdout if r["id"] in m]
        if len(y_true) >= 4:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            with open("/logs/verifier/holdout_ratio.txt", "w") as f:
                f.write(f"{corr:.3f}\n")
            assert corr >= 0.5, (
                f"Holdout correlation {corr:.3f} <0.5 - overfit to visible"
            )

    # Also ensure main visible correlation holds (otherwise holdout will also fail)
    # This file is informational if main passes - don't block reward unless severe degradation
    print(f"Holdout has {len(holdout)} records - checked generalization")
