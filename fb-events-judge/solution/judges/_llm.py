"""OpenAI-compatible judge call, with a hard requirement that it never raises.

Each judge asks the model first. If the endpoint is unconfigured, unreachable,
or answers with something unparseable, ask() returns None and the caller falls
back to its deterministic inventory check. That keeps oracle validation
reproducible on a machine with no API access while still exercising the LLM
path the task asks for.
"""

import json
import os
import re

_client = None
_client_tried = False


def _get_client():
    global _client, _client_tried
    if _client_tried:
        return _client
    _client_tried = True
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        _client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "https://api.llama.com/compat/v1/"),
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            max_retries=1,
        )
    except Exception:
        _client = None
    return _client


def inventory_slice(item, events, limit=120):
    """The inventory rows a judge needs: the queried city, plus any row whose
    title appears in the answer (so out-of-city claims can be resolved)."""
    city = (item.get("query_filters") or {}).get("city")
    answer = item.get("agent_answer", "")
    rows, seen = [], set()
    for e in events:
        if e.get("city") == city and len(rows) < limit:
            rows.append(e)
            seen.add(e["id"])
    for e in events:
        if e["id"] not in seen and e["title"] and e["title"] in answer:
            rows.append(e)
    return "\n".join(
        f"{e['id']} | {e['title']} | venue={e['venue']} | city={e['city']} "
        f"| start={e['start_time'][:10]} | is_spam={e.get('is_spam')} "
        f"| cluster={e.get('cluster_id')}"
        for e in rows
    )


def ask(dimension, focus, item, events):
    """Score one dimension via the LLM, or None if that isn't possible."""
    client = _get_client()
    if client is None:
        return None

    prompt = f"""You are the {dimension} judge for an FB Events answer.

Score ONLY this dimension: {focus}

User question: {item.get("user_question")}
Query filters: {json.dumps(item.get("query_filters") or {})}

Inventory rows you may rely on. An event that does not appear here does not exist:
{inventory_slice(item, events)}

Answer to judge:
{item.get("agent_answer")}

Give 5 when this dimension is fully satisfied and 1-2 when it is violated.
Reply with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>", "verdict": "pass" or "fail"}}"""

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
        text = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        score = int(data["score"])
        if not 1 <= score <= 5:
            return None
        reasoning = str(data.get("reasoning") or "").strip()
        if not reasoning:
            return None
        verdict = data.get("verdict") or ("pass" if score >= 4 else "fail")
        return {"score": score, "reasoning": reasoning, "verdict": verdict}
    except Exception:
        return None
