"""Spam judge: duplicates, undeduped clusters, and flagged-spam events.

Oracle implementation - deterministic and grounded in events.jsonl.
"""

try:
    from ._common import parse_listings, resolve, says_no_results
except ImportError:
    from _common import parse_listings, resolve, says_no_results


def judge(item, events):
    answer = item.get("agent_answer", "")
    if says_no_results(answer):
        return {
            "score": 5,
            "reasoning": "Answer claims no events; nothing duplicated.",
            "verdict": "pass",
        }

    problems = []
    seen_listings, seen_clusters = set(), {}
    for listing in parse_listings(answer):
        key = (listing["head"], listing["date"])
        if key in seen_listings:
            problems.append(f"{listing['head']} listed more than once")
        seen_listings.add(key)

        event = resolve(listing, events)
        if not event:
            continue
        if event.get("is_spam"):
            problems.append(f"{event['title']} is flagged is_spam")
        cluster = event.get("cluster_id")
        if cluster:
            if cluster in seen_clusters and seen_clusters[cluster] != event["id"]:
                problems.append(
                    f"{event['title']} repeats cluster {cluster} already covered "
                    f"by {seen_clusters[cluster]}"
                )
            seen_clusters.setdefault(cluster, event["id"])

    if problems:
        return {
            "score": 1,
            "reasoning": "Spam/dedup failures: " + "; ".join(problems[:3]) + ".",
            "verdict": "fail",
        }
    return {
        "score": 5,
        "reasoning": "No duplicate clusters, repeats, or flagged-spam events.",
        "verdict": "pass",
    }
