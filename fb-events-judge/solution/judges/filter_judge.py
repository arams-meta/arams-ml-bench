"""Filter judge: does every claimed event respect the query's city and window?

Oracle implementation - deterministic and grounded in events.jsonl.
"""

try:
    from ._common import (
    decide,
    parse_listings,
    resolve,
    says_no_results,
    in_window,
    stated_city,
    answer_cities,
    )
except ImportError:  # loaded as a loose module rather than a package
    from _common import (
    decide,
    parse_listings,
    resolve,
    says_no_results,
    in_window,
    stated_city,
    answer_cities,
    )

def _deterministic(item, events):
    answer = item.get("agent_answer", "")
    filters = item.get("query_filters", {}) or {}
    want_city = filters.get("city")
    cities = {e["city"] for e in events}

    if says_no_results(answer):
        return {
            "score": 5,
            "reasoning": "Answer claims no events; no filter to violate.",
            "verdict": "pass",
        }

    problems = []
    # Some answers name the wrong city in the preamble rather than per-listing.
    for other in sorted(answer_cities(answer, cities) - {want_city}):
        problems.append(f"answer places results in {other}, not {want_city}")

    for listing in parse_listings(answer):
        head = listing["head"]
        event = resolve(listing, events)
        # A city named in the answer text is a claim in its own right, even if
        # the underlying event resolves to somewhere else.
        claimed = stated_city(listing, cities)
        if claimed and want_city and claimed != want_city:
            problems.append(f"{head} placed in {claimed}, not {want_city}")
        elif event and want_city and event["city"] != want_city:
            problems.append(f"{head} is in {event['city']}, not {want_city}")
        if not in_window(listing["date"], filters):
            problems.append(
                f"{head} on {listing['date']} falls outside "
                f"{(filters.get('time_window_start') or '')[:10]}.."
                f"{(filters.get('time_window_end') or '')[:10]}"
            )

    if problems:
        return {
            "score": 2,
            "reasoning": "Filter violations: " + "; ".join(problems[:3]) + ".",
            "verdict": "fail",
        }
    return {
        "score": 5,
        "reasoning": f"All listed events match city {want_city} and the requested window.",
        "verdict": "pass",
    }


FOCUS = (
    "does every listed event resolve to a row whose city equals the requested city, with its date inside the requested window? Ignore existence, spam and style."
)


def judge(item, events):
    """Ask the LLM first; fall back to the deterministic inventory check."""
    return decide("filter", FOCUS, item, events, _deterministic)
