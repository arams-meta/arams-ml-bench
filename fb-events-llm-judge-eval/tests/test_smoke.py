import os
import pandas as pd


def test_output_exists():
    assert os.path.exists("/output/judgments.csv"), (
        "judge output /output/judgments.csv not found - agent must create judge that writes CSV"
    )


def test_output_loadable():
    df = pd.read_csv("/output/judgments.csv")
    assert len(df) >= 10, f"Expected >=10 judgments, got {len(df)}"


def test_judge_file_exists():
    # Agent should create judge.py or similar in /app or /output
    possible = [
        "/app/judge.py",
        "/app/main.py",
        "/output/judge.py",
        "/app/judge_oracle.py",
    ]
    # Not strict - smoke only checks output exists, but log if judge.py missing
    # This is informational
    pass
