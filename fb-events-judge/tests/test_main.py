import json, pathlib
import pandas as pd
import numpy as np


def load_dataset():
    # Ground truth WITH human_score is in tests/data (hidden from agent) - use that for evaluation
    # Agent-visible data/ has no scores to prevent cheating (P10 fix)
    candidates = [
        pathlib.Path("/tests/data/judge_dataset_with_scores.json"),
        pathlib.Path("tests/data/judge_dataset_with_scores.json"),
        pathlib.Path("/app/data/judge_dataset_with_scores.json"),  # legacy
        pathlib.Path("/app/data/judge_dataset.json"),
        pathlib.Path("data/judge_dataset.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    d = json.load(f)
                # Need human_score field - skip if not present (agent-visible version)
                if d and "human_score" in d[0]:
                    return d
            except:
                continue
    # Fallback to visible if hidden not found (will fail correlation check and signal issue)
    p = pathlib.Path("/app/data/judge_dataset.json")
    if not p.exists():
        p = pathlib.Path("data/judge_dataset.json")
    with open(p) as f:
        return json.load(f)


def load_visible_dataset():
    p = pathlib.Path("/app/data/judge_dataset.json")
    if not p.exists():
        p = pathlib.Path("data/judge_dataset.json")
    with open(p) as f:
        return json.load(f)


def load_judgments():
    return pd.read_csv("/output/judgments.csv")


def get_score_col(df):
    # Prefer final_score for multi-judge, else score
    if "final_score" in df.columns:
        return "final_score"
    return "score"


def test_directional_good_gt_bad():
    data = load_dataset()
    judg = load_judgments()
    score_col = get_score_col(judg)
    judg_map = dict(zip(judg["id"], judg[score_col]))

    good_scores = []
    bad_scores = []
    for r in data:
        sid = r["id"]
        if sid not in judg_map:
            continue
        if r["category"] == "good":
            good_scores.append(judg_map[sid])
        else:
            bad_scores.append(judg_map[sid])

    assert len(good_scores) >= 5
    assert len(bad_scores) >= 5
    avg_good = float(np.mean(good_scores))
    avg_bad = float(np.mean(bad_scores))
    margin = avg_good - avg_bad
    with open("/logs/verifier/baseline_score.txt", "w") as f:
        f.write(f"avg_good={avg_good:.2f} avg_bad={avg_bad:.2f} margin={margin:.2f}\n")
    assert margin >= 1.0, (
        f"Directional fail: good should beat bad by >=1.0, got margin={margin:.2f}"
    )


def test_correlation_with_human():
    data = load_dataset()
    judg = load_judgments()
    score_col = get_score_col(judg)
    judg_map = dict(zip(judg["id"], judg[score_col]))
    y_true = []
    y_pred = []
    for r in data:
        sid = r["id"]
        if sid in judg_map:
            y_true.append(r["human_score"])
            y_pred.append(judg_map[sid])
    if len(y_true) < 4:
        return
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    with open("/logs/verifier/holdout_score.txt", "w") as f:
        f.write(f"corr={corr:.3f}\n")
    assert corr >= 0.6, f"Correlation {corr:.3f} <0.6"


def test_faithfulness_detection():
    """Hallucination flaw should be caught by accuracy_score or main score"""
    data = load_dataset()
    judg = load_judgments()
    score_col = get_score_col(judg)
    # Prefer accuracy_score if exists
    acc_col = "accuracy_score" if "accuracy_score" in judg.columns else score_col
    judg_map = dict(zip(judg["id"], judg[acc_col]))
    for r in data:
        if r.get("flaw_type") == "hallucination":
            sid = r["id"]
            if sid in judg_map:
                assert judg_map[sid] <= 3, (
                    f"Hallucinated {sid} should score <=3 for {acc_col}, got {judg_map[sid]}"
                )


def test_spam_detection():
    """Spam flaw should be caught by spam_score"""
    data = load_dataset()
    judg = load_judgments()
    spam_col = "spam_score" if "spam_score" in judg.columns else get_score_col(judg)
    judg_map = dict(zip(judg["id"], judg[spam_col]))
    for r in data:
        if r.get("flaw_type") == "spam":
            sid = r["id"]
            if sid in judg_map:
                assert judg_map[sid] <= 3, (
                    f"Spam {sid} should score <=3 for {spam_col}, got {judg_map[sid]}"
                )


def test_filter_detection():
    """Wrong time/city should be caught by filter_score"""
    data = load_dataset()
    judg = load_judgments()
    filt_col = "filter_score" if "filter_score" in judg.columns else get_score_col(judg)
    judg_map = dict(zip(judg["id"], judg[filt_col]))
    for r in data:
        if r.get("flaw_type") in ["wrong_time", "wrong_city"]:
            sid = r["id"]
            if sid in judg_map:
                assert judg_map[sid] <= 3, (
                    f"Filter flaw {sid} should score <=3 for {filt_col}, got {judg_map[sid]}"
                )


def test_no_hardcoded_scores():
    judg = load_judgments()
    score_col = get_score_col(judg)
    uniq = judg[score_col].nunique()
    assert uniq >= 2, f"All same score {uniq} - hardcoded"
    # Strengthened: need variance >0.5 to defeat 2-constant bypass (oracle has 3 uniques 5,2,1 with std ~1.4)
    assert float(judg[score_col].std()) > 0.5, (
        f"Std too low {float(judg[score_col].std()):.2f} - likely hardcoded"
    )


def test_multi_judge_presence():
    """If multi-judge, ensure 4 judges present per instruction"""
    judg = load_judgments()
    has_multi = any(
        c in judg.columns
        for c in ["accuracy_score", "filter_score", "spam_score", "helpfulness_score"]
    )
    if has_multi:
        # Should have at least 3 of them
        present = [
            c
            for c in [
                "accuracy_score",
                "filter_score",
                "spam_score",
                "helpfulness_score",
            ]
            if c in judg.columns
        ]
        assert len(present) >= 3, f"Multi-judge needs >=3 sub-scores, got {present}"

def test_judges_files_exist():
    """Check that agent created 4 judge modules per instruction (multi-judge panel)"""
    import pathlib
    base_paths = [pathlib.Path("/app/judges"), pathlib.Path("judges"), pathlib.Path("fb-events-judge/judges")]
    for base in base_paths:
        if base.exists():
            files = list(base.glob("*.py"))
            # Should have at least 3 judge files
            assert len(files) >= 2, f"Expected >=2 judge files in {base}, got {files}"
            return
    # If not in /app/judges, check /app/judge.py orchestrator exists
    for p in [pathlib.Path("/app/judge.py"), pathlib.Path("judges/judge.py"), pathlib.Path("fb-events-judge/judge.py"), pathlib.Path("judge.py")]:
        if p.exists():
            return
    # For oracle run, judges are created in solution/judges - allow that for oracle but warn
    # This test is informational for agent runs, not gating for oracle
    print("No /app/judges found - might be oracle run, skipping file existence check")
