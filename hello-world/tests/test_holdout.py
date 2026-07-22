import pickle
import numpy as np
import pandas as pd


def test_holdout_generalization():
    train = pd.read_csv("/app/data/train.csv")
    holdout = pd.read_csv("/tests/data/holdout.csv")

    with open("/output/model.pkl", "rb") as f:
        model = pickle.load(f)

    train_preds = model.predict(train[["x"]])
    train_mae = np.mean(np.abs(train_preds - train["y"].values))

    holdout_preds = model.predict(holdout[["x"]])
    holdout_mae = np.mean(np.abs(holdout_preds - holdout["y"].values))

    with open("/logs/verifier/holdout_score.txt", "w") as f:
        f.write(f"{holdout_mae}\n")

    ratio = round(train_mae / max(holdout_mae, 1e-6), 3)
    with open("/logs/verifier/holdout_ratio.txt", "w") as f:
        f.write(f"{ratio}\n")

    assert holdout_mae < train_mae * 2.0, (
        f"Holdout MAE {holdout_mae:.4f} > 2x train MAE {train_mae:.4f} — model may be overfit"
    )
