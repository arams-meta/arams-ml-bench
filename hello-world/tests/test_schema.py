"""Validates output file format."""
import pandas as pd


def test_output_schema():
    df = pd.read_csv("/output/predictions.csv")
    assert list(df.columns) == ["id", "prediction"], (
        f"Expected columns [id, prediction], got {list(df.columns)}"
    )
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
    assert df["id"].dtype in ("int64", "int32", "float64")
    assert df["prediction"].dtype == "float64"
