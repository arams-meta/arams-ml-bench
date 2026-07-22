import os
import pandas as pd


def test_output_file_exists():
    assert os.path.exists("/output/predictions.csv"), "predictions.csv not found"


def test_output_loadable():
    df = pd.read_csv("/output/predictions.csv")
    assert len(df) > 0, "predictions file is empty"
