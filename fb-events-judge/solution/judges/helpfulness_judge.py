"""Helpfulness judge: does the answer actually respond, with venue and date?

Oracle implementation. Judges presentation only - whether the facts are correct
is the other three judges' business.
"""

try:
    from ._common import (
    decide,
    parse_listings,
    says_no_results,
    )
except ImportError:  # loaded as a loose module rather than a package
    from _common import (
    decide,
    parse_listings,
    says_no_results,
    )

def _deterministic(item, events):
    answer = (item.get("agent_answer") or "").strip()

    if says_no_results(answer):
        return {
            "score": 5,
            "reasoning": "Clear, direct 'no matching events' response.",
            "verdict": "pass",
        }

    if len(answer) < 20:
        return {
            "score": 1,
            "reasoning": "Answer is too short to be useful.",
            "verdict": "fail",
        }

    listings = parse_listings(answer)
    if not listings:
        return {
            "score": 2,
            "reasoning": "Names no events with a venue and date the user could act on.",
            "verdict": "fail",
        }

    dated = [l for l in listings if l["date"]]
    if len(dated) < len(listings):
        return {
            "score": 3,
            "reasoning": "Some listed events are missing a date.",
            "verdict": "fail",
        }

    return {
        "score": 5,
        "reasoning": f"Names {len(listings)} events, each with venue and date.",
        "verdict": "pass",
    }


FOCUS = (
    "does the answer actually help - naming events with a date, or correctly reporting none? Ignore whether the underlying facts are right."
)


def judge(item, events):
    """Ask the LLM first; fall back to the deterministic inventory check."""
    return decide("helpfulness", FOCUS, item, events, _deterministic)
