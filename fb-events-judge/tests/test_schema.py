import pandas as pd


def test_schema_columns():
    df = pd.read_csv("/output/judgments.csv")
    # Support both single-judge (score) and multi-judge (final_score)
    assert "id" in df.columns
    assert "reasoning" in df.columns
    # Must have either score or final_score
    assert ("score" in df.columns) or ("final_score" in df.columns), (
        f"Need score or final_score, got {list(df.columns)}"
    )


def test_score_range():
    df = pd.read_csv("/output/judgments.csv")
    # Check main score columns 1-5
    score_col = "final_score" if "final_score" in df.columns else "score"
    assert df[score_col].between(1, 5).all(), (
        f"Main score {score_col} must be 1-5, got {df[score_col].unique()}"
    )
    # If multi-judge, check sub-scores
    for col in ["accuracy_score", "filter_score", "spam_score", "helpfulness_score"]:
        if col in df.columns:
            assert df[col].between(1, 5).all(), f"{col} must be 1-5"


def test_reasoning_non_empty():
    df = pd.read_csv("/output/judgments.csv")
    assert df["reasoning"].astype(str).str.len().min() > 5, (
        "Reasoning must be non-empty"
    )


def test_id_matches_dataset():
    import json, pathlib

    ds_path = pathlib.Path("/app/data/judge_dataset.json")
    if not ds_path.exists():
        ds_path = pathlib.Path("data/judge_dataset.json")
    if ds_path.exists():
        with open(ds_path) as f:
            data = json.load(f)
        expected_ids = set(r["id"] for r in data)
        df = pd.read_csv("/output/judgments.csv")
        actual_ids = set(df["id"].astype(str))
        missing = expected_ids - actual_ids
        assert len(missing) == 0, f"Missing judgments for ids {missing}"


def test_multi_judge_columns_if_present():
    df = pd.read_csv("/output/judgments.csv")
    # If multi-judge, should have at least 2 sub-scores
    multi_cols = [
        c
        for c in ["accuracy_score", "filter_score", "spam_score", "helpfulness_score"]
        if c in df.columns
    ]
    if len(multi_cols) > 0:
        assert len(multi_cols) >= 2, (
            f"Multi-judge should have >=2 sub-scores, got {multi_cols}"
        )
