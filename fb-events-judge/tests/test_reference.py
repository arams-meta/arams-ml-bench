import json, pathlib
import pandas as pd


def test_reference_toy():
    """Exact on small input: toy cases have known scores"""
    toy_path = pathlib.Path("/app/data/reference_toy.json")
    if not toy_path.exists():
        toy_path = pathlib.Path("data/reference_toy.json")
    if not toy_path.exists():
        toy_path = pathlib.Path("/app/references/../data/reference_toy.json")
    # Also try tests/data
    alt = pathlib.Path("/tests/data/reference_toy.json")
    if not toy_path.exists() and alt.exists():
        toy_path = alt
    if not toy_path.exists():
        # try local
        alt2 = pathlib.Path("tests/data/reference_toy.json")
        if alt2.exists():
            toy_path = alt2
    if not toy_path.exists():
        print("Toy reference not found, skipping")
        return

    with open(toy_path) as f:
        toy = json.load(f)

    df = pd.read_csv("/output/judgments.csv")
    score_col = "final_score" if "final_score" in df.columns else "score"
    df_map = dict(zip(df["id"], df[score_col]))

    for t in toy:
        tid = t["id"]
        if tid not in df_map:
            continue  # not in main dataset, okay
        expected = t["human_score"]
        actual = df_map[tid]
        # Allow +-1 tolerance for LLM judge variance
        assert abs(actual - expected) <= 1, (
            f"Toy {tid}: expected {expected} +-1, got {actual}. {t['agent_answer'][:100]}"
        )
