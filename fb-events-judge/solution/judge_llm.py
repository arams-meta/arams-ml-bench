"""
Real LLM-as-Judge implementation for FB Events - for reference, not used by oracle solve.sh
This would be what agent should build in /app/judge.py
"""

import os, json, csv, pathlib
from openai import OpenAI


def load_rubric():
    p = pathlib.Path("/app/references/rubric.md")
    if not p.exists():
        p = pathlib.Path("references/rubric.md")
    if p.exists():
        return p.read_text()
    return "Score 1-5 for faithfulness, filter adherence, spam/dedup"


def judge_one(client, rubric, item, events_map):
    q = item["user_question"]
    filters = item["query_filters"]
    answer = item["agent_answer"]
    gt_ids = item.get("ground_truth_ids", [])

    # Build context of real events for faithfulness check
    relevant_events = [events_map.get(eid, {}) for eid in gt_ids[:3]]
    inventory_snippet = "\n".join(
        [
            f"{e.get('id')}: {e.get('title')} in {e.get('city')}"
            for e in relevant_events
            if e
        ]
    )

    prompt = f"""{rubric}

---

User Question: {q}
Filters: {filters}
Ground truth relevant ids (for your reference, not to output): {gt_ids}
Inventory snippet (real events that exist):
{inventory_snippet}

Agent Answer to Judge:
{answer}

Task: Score this agent answer 1-5 per rubric. Check:
1. Does it hallucinate events not in inventory?
2. Does it respect city/category/time filters?
3. Does it avoid spam and dedup cluster_id?
4. Is it helpful?

Output JSON only: {{"score": int, "reasoning": "brief", "checks": {{"faithfulness":"pass/fail","filter_adherence":"pass/fail","spam_dedup":"pass/fail","helpfulness":"pass/fail"}}}}
"""

    try:
        resp = client.chat.completions.create(
            model=os.getenv("JUDGE_MODEL", "Llama-3.3-70B-Instruct"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict FB Events eval judge. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        txt = resp.choices[0].message.content.strip()
        # Try parse JSON
        import re

        # Extract JSON block
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            score = int(data.get("score", 3))
            reasoning = data.get("reasoning", txt[:200])
            checks = data.get("checks", {})
        else:
            score = 3
            reasoning = txt[:200]
            checks = {}
        return score, reasoning, checks
    except Exception as e:
        print(f"Judge call failed for {item['id']}: {e}")
        return 3, f"Error: {e}", {}


def main():
    dataset_path = pathlib.Path("/app/data/judge_dataset.json")
    if not dataset_path.exists():
        dataset_path = pathlib.Path("data/judge_dataset.json")
    events_path = pathlib.Path("/app/data/events.jsonl")
    if not events_path.exists():
        events_path = pathlib.Path("data/events.jsonl")

    with open(dataset_path) as f:
        dataset = json.load(f)

    events_map = {}
    if events_path.exists():
        with open(events_path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    events_map[e["id"]] = e
                except:
                    pass

    rubric = load_rubric()

    client = OpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "https://api.llama.com/compat/v1/"),
        api_key=os.getenv("OPENAI_API_KEY", "test"),
    )

    out_path = pathlib.Path("/output/judgments.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "id",
                "score",
                "reasoning",
                "faithfulness",
                "filter_adherence",
                "spam_dedup",
                "helpfulness",
            ],
        )
        writer.writeheader()
        for item in dataset:
            score, reasoning, checks = judge_one(client, rubric, item, events_map)
            writer.writerow(
                {
                    "id": item["id"],
                    "score": max(1, min(5, score)),
                    "reasoning": reasoning[:500],
                    "faithfulness": checks.get("faithfulness", "pass"),
                    "filter_adherence": checks.get("filter_adherence", "pass"),
                    "spam_dedup": checks.get("spam_dedup", "pass"),
                    "helpfulness": checks.get("helpfulness", "pass"),
                }
            )
            print(f"Judged {item['id']}: {score}")


if __name__ == "__main__":
    main()
