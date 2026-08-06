import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import pandas as pd


def get_score_col(df):
    if "final_score" in df.columns:
        return "final_score"
    return "score"


def test_holdout_generalization():
    """Holdout gate: re-run /app/judge.py on stripped holdout and check margin/corr"""
    holdout_path = pathlib.Path("/tests/data/holdout.json")
    if not holdout_path.exists():
        holdout_path = pathlib.Path("tests/data/holdout.json")
    if not holdout_path.exists():
        print("Holdout not found, skipping")
        return

    with open(holdout_path) as f:
        holdout = json.load(f)

    if not holdout:
        print("Holdout empty, skipping")
        return

    # 2. Strip labels — keep only agent-visible fields
    allowed = {
        "id",
        "query_id",
        "user_question",
        "query_filters",
        "agent_answer",
        "ground_truth_ids",
    }
    stripped = []
    for r in holdout:
        nr = {k: r[k] for k in allowed if k in r}
        # Ensure id exists
        if "id" not in nr:
            continue
        stripped.append(nr)

    tmp_input = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    )
    try:
        json.dump(stripped, tmp_input)
        tmp_input.close()
        tmp_input_path = tmp_input.name

        tmp_output = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir="/tmp"
        )
        tmp_output.close()
        tmp_output_path = tmp_output.name

        # 3. Assert orchestrator exists
        judge_path = pathlib.Path("/app/judge.py")
        assert judge_path.exists(), (
            f"/app/judge.py not found - required for holdout re-run. Checked {judge_path}"
        )

        # 4. Run with env vars
        env = os.environ.copy()
        env["JUDGE_DATASET"] = tmp_input_path
        env["JUDGE_OUTPUT"] = tmp_output_path

        try:
            result = subprocess.run(
                ["python3", "/app/judge.py"],
                env=env,
                timeout=600,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            assert False, (
                f"Holdout judge.py timed out after 600s: {e}. stdout={getattr(e, 'stdout', '')[:2000]} stderr={getattr(e, 'stderr', '')[:2000]}"
            )

        if result.returncode != 0:
            tail = (result.stderr or "")[-3000:] + "\n" + (result.stdout or "")[-2000:]
            assert False, (
                f"/app/judge.py failed on holdout (rc={result.returncode}). stderr+stdout tail:\n{tail}"
            )

        # 5. Read CSV
        assert pathlib.Path(tmp_output_path).exists(), (
            f"Judge did not create JUDGE_OUTPUT={tmp_output_path}"
        )
        df = pd.read_csv(tmp_output_path)
        assert not df.empty, "Holdout judgments CSV is empty"
        score_col = get_score_col(df)
        assert score_col in df.columns, (
            f"Score column {score_col} missing in holdout output, columns={list(df.columns)}"
        )
        pred_map = dict(zip(df["id"], df[score_col]))

        # 6. Assert every holdout id present
        missing = [r["id"] for r in holdout if r["id"] not in pred_map]
        assert not missing, (
            f"Missing {len(missing)} holdout ids in output: {missing[:10]}"
        )

        # 7. Compute margin and corr
        good_scores = []
        bad_scores = []
        y_true = []
        y_pred = []
        for r in holdout:
            sid = r["id"]
            if sid not in pred_map:
                continue
            pred = pred_map[sid]
            if r.get("category") == "good":
                good_scores.append(pred)
            else:
                bad_scores.append(pred)
            y_true.append(r["human_score"])
            y_pred.append(pred)

        assert len(good_scores) >= 2, (
            f"Need >=2 good in holdout, got {len(good_scores)}"
        )
        assert len(bad_scores) >= 2, f"Need >=2 bad in holdout, got {len(bad_scores)}"

        margin = (
            float(np.mean(good_scores) - np.mean(bad_scores))
            if good_scores and bad_scores
            else 0.0
        )
        if len(y_true) >= 2:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # 8. Write both to holdout_ratio.txt
        out_path = pathlib.Path("/logs/verifier/holdout_ratio.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(f"margin={margin:.3f} corr={corr:.3f}\n")
            f.write(
                f"avg_good={float(np.mean(good_scores)):.2f} avg_bad={float(np.mean(bad_scores)):.2f}\n"
            )

        # 9. Gate
        assert margin >= 0.6, (
            f"Holdout directional fail: good should beat bad by >=0.6, got margin={margin:.3f} (good={np.mean(good_scores):.2f} bad={np.mean(bad_scores):.2f})"
        )
        assert corr >= 0.5, f"Holdout correlation {corr:.3f} <0.5 - overfit to visible"

    finally:
        # Cleanup temp files
        for p in [locals().get("tmp_input_path"), locals().get("tmp_output_path")]:
            if p:
                try:
                    pathlib.Path(p).unlink()
                except:
                    pass
