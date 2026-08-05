"""
Reference multi-judge LLM-as-Judge implementation for FB Events.

Not used by the oracle solve.sh - this is the worked example of what the agent
is asked to build: four dimension-specific judges, each its own LLM call, plus
the weighted aggregation and the full judgments.csv schema.
"""

import os, json, csv, pathlib, re
from openai import OpenAI

DIMENSIONS = ["accuracy", "filter", "spam", "helpfulness"]

WEIGHTS = {"accuracy": 0.35, "filter": 0.30, "spam": 0.20, "helpfulness": 0.15}

# Column in judgments.csv carrying each judge's pass/fail verdict.
VERDICT_COL = {
    "accuracy": "faithfulness",
    "filter": "filter_adherence",
    "spam": "spam_dedup",
    "helpfulness": "helpfulness",
}

FIELDNAMES = [
    "id",
    "final_score",
    "accuracy_score",
    "filter_score",
    "spam_score",
    "helpfulness_score",
    "reasoning",
    "faithfulness",
    "filter_adherence",
    "spam_dedup",
    "helpfulness",
    "score",
]

FOCUS = {
    "accuracy": (
        "Does every event the answer lists actually exist in the inventory? "
        "An event exists only if some inventory row has both that exact title "
        "and that exact venue. Invented events are the failure you are looking "
        "for. Ignore city, date, spam and style."
    ),
    "filter": (
        "Does every listed event respect the query filters? Each listed event "
        "must resolve to an inventory row whose city equals the requested city, "
        "and its date must fall inside the requested time window. Ignore whether "
        "events exist, spam, duplicates and style."
    ),
    "spam": (
        "Does the answer avoid spam and duplicates? Fail it if two listed events "
        "share a cluster_id, if the same event is listed twice, or if any listed "
        "event has is_spam true. Ignore existence, city, date and style."
    ),
    "helpfulness": (
        "Does the answer actually help - does it name events with venue and "
        "date, and respond to what was asked? A correct 'no events found' for a "
        "genuinely empty result is helpful. Ignore correctness of the underlying "
        "facts; other judges cover that."
    ),
}


def load_rubric():
    for p in [
        pathlib.Path("/app/references/rubric_multi_judge.md"),
        pathlib.Path("/app/references/rubric.md"),
        pathlib.Path("references/rubric_multi_judge.md"),
        pathlib.Path("references/rubric.md"),
    ]:
        if p.exists():
            return p.read_text()
    return "Score 1-5 on the stated dimension."


def load_events():
    for p in [pathlib.Path("/app/data/events.jsonl"), pathlib.Path("data/events.jsonl")]:
        if p.exists():
            with open(p) as f:
                return [json.loads(line) for line in f if line.strip()]
    return []


def inventory_context(item, events):
    """The slice of inventory a judge needs: everything in the queried city."""
    filters = item.get("query_filters", {})
    city = filters.get("city")
    local = [e for e in events if e.get("city") == city]
    lines = []
    for e in local[:120]:
        lines.append(
            f"{e['id']} | {e['title']} | venue={e['venue']} | city={e['city']} "
            f"| start={e['start_time'][:10]} | is_spam={e.get('is_spam')} "
            f"| cluster={e.get('cluster_id')}"
        )
    # Titles listed in the answer may belong to other cities; include those rows
    # so the filter judge can resolve them instead of guessing.
    for title in re.findall(r'"([^"]+)"', item.get("agent_answer", "")):
        for e in events:
            if e["title"] == title and e.get("city") != city:
                lines.append(
                    f"{e['id']} | {e['title']} | venue={e['venue']} | city={e['city']} "
                    f"| start={e['start_time'][:10]} | is_spam={e.get('is_spam')} "
                    f"| cluster={e.get('cluster_id')}"
                )
    return "\n".join(lines)


def judge_one(client, rubric, dimension, item, events):
    prompt = f"""{rubric}

You are the {dimension} judge. Score ONLY this dimension.

{FOCUS[dimension]}

User question: {item["user_question"]}
Query filters: {json.dumps(item.get("query_filters", {}))}

Inventory rows you may rely on (an event not listed here does not exist):
{inventory_context(item, events)}

Agent answer to judge:
{item["agent_answer"]}

Score 5 when this dimension is fully satisfied, 1-2 when it is violated.
Output JSON only: {{"score": int 1-5, "reasoning": "one sentence", "verdict": "pass" or "fail"}}
"""
    try:
        resp = client.chat.completions.create(
            model=os.getenv("JUDGE_MODEL", "Llama-3.3-70B-Instruct"),
            messages=[
                {
                    "role": "system",
                    "content": f"You are a strict FB Events {dimension} judge. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return 3, txt[:200] or f"{dimension}: unparseable response", "fail"
        data = json.loads(m.group(0))
        score = max(1, min(5, int(data.get("score", 3))))
        reasoning = str(data.get("reasoning") or f"{dimension}: no reasoning given")
        verdict = data.get("verdict") or ("pass" if score >= 4 else "fail")
        return score, reasoning, verdict
    except Exception as e:
        # Never raise: a judge that dies takes the whole run down with it.
        print(f"{dimension} judge failed for {item['id']}: {e}")
        return 3, f"{dimension} judge error: {e}", "fail"


def aggregate(scores):
    """Weighted blend of the four sub-scores, clamped to the 1-5 range."""
    total = sum(WEIGHTS[d] * scores[d] for d in DIMENSIONS)
    return max(1, min(5, round(total)))


def main():
    dataset_path = pathlib.Path("/app/data/judge_dataset.json")
    if not dataset_path.exists():
        dataset_path = pathlib.Path("data/judge_dataset.json")
    with open(dataset_path) as f:
        dataset = json.load(f)

    events = load_events()
    rubric = load_rubric()

    client = OpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "https://api.llama.com/compat/v1/"),
        api_key=os.getenv("OPENAI_API_KEY", "test"),
    )

    out_path = pathlib.Path("/output/judgments.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        for item in dataset:
            scores, reasons, verdicts = {}, [], {}
            for dim in DIMENSIONS:
                score, reasoning, verdict = judge_one(
                    client, rubric, dim, item, events
                )
                scores[dim] = score
                verdicts[dim] = verdict
                reasons.append(f"{dim}: {reasoning}")

            final = aggregate(scores)
            row = {
                "id": item["id"],
                "final_score": final,
                "score": final,
                "reasoning": " | ".join(reasons)[:900],
            }
            for dim in DIMENSIONS:
                row[f"{dim}_score"] = scores[dim]
                row[VERDICT_COL[dim]] = verdicts[dim]
            writer.writerow(row)
            print(f"Judged {item['id']}: final={final} {scores}")


if __name__ == "__main__":
    main()
