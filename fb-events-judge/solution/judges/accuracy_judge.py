"""Accuracy judge: does every claimed event actually exist in the inventory?

Oracle implementation - deterministic and grounded in events.jsonl. A real
solution is expected to reach the same verdicts via the LLM API; see
solution/judge_llm.py for the prompt-based version.
"""

try:
    from ._common import (
    decide,
    parse_listings,
    resolve,
    says_no_results,
    )
except ImportError:  # loaded as a loose module rather than a package
    from _common import (
    decide,
    parse_listings,
    resolve,
    says_no_results,
    )

def _deterministic(item, events):
    answer = item.get("agent_answer", "")
    if says_no_results(answer):
        return {
            "score": 5,
            "reasoning": "Answer claims no events; nothing to hallucinate.",
            "verdict": "pass",
        }

    listings = parse_listings(answer)
    if not listings:
        return {
            "score": 3,
            "reasoning": "No parseable event listings to verify against inventory.",
            "verdict": "fail",
        }

    invented = [l for l in listings if resolve(l, events) is None]
    if invented:
        names = ", ".join(sorted({l["head"] for l in invented}))
        return {
            "score": 1,
            "reasoning": f"Not present in events.jsonl: {names}.",
            "verdict": "fail",
        }
    return {
        "score": 5,
        "reasoning": f"All {len(listings)} listed events resolve to inventory rows.",
        "verdict": "pass",
    }


FOCUS = (
    "does every event the answer lists actually exist in the inventory (same title and venue)? Ignore city, date, spam and style."
)


def judge(item, events):
    """Ask the LLM first; fall back to the deterministic inventory check."""
    return decide("accuracy", FOCUS, item, events, _deterministic)
