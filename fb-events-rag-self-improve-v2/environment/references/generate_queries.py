#!/usr/bin/env python3
"""
Deterministic query + relevance generator for synthetic events.
Reads events.jsonl, writes queries.jsonl and relevance.jsonl.

Relevance rule is rule-based (no LLM) to avoid labeling ambiguity:
 - event city == query city
 - haversine distance <= radius_km
 - start_time within [time_window_start, time_window_end]
 - category match if query specifies category
 - facet/topic match: ev.get("topic") == query.facet
 - not is_spam, not in past relative to base date (2026-02-01)
 - dedup: keep only 1 per cluster_id (highest popularity)

Anchored sampling: 80% queries anchored to real events to guarantee >=1 relevant.
Query text injects facet for semantic relevance.
"""

import argparse
import json
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict

CATEGORIES = [
    "Music",
    "Sports",
    "Food",
    "Tech",
    "Arts",
    "Nightlife",
    "Community",
    "Business",
]

# Templates that inject facet
QUERY_TEMPLATES_FACET = [
    "best {facet} {category} events near {city} {time_phrase}",
    "{facet} {category} {activity} {time_phrase} in {city}",
    "{category} events about {facet} in {city} {time_phrase}",
    "looking for {facet} {category} events in {city} {time_phrase}",
    "{facet} events {time_phrase} {city}",
    "things to do {time_phrase} in {city} - {facet} {category}",
    "{city} {facet} {category} {activity} {time_phrase}",
    "weekend {facet} {category} in {city}",
    "{facet} {category} meetup {time_phrase} {city}",
    "{activity} about {facet} in {city} {time_phrase}",
    "where to find {facet} {category} in {city} {time_phrase}",
]

TIME_PHRASES = [
    ("this weekend", 2, 3),
    ("next week", 7, 14),
    ("this month", 0, 30),
    ("next month", 30, 60),
    ("tomorrow", 1, 1),
    ("this Friday", 4, 4),
]

ACTIVITIES = [
    "meetup",
    "festival",
    "concert",
    "workshop",
    "market",
    "gathering",
    "showcase",
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def parse_time(s):
    return datetime.fromisoformat(s)


def dedup_by_cluster(rel_ids, ev_map):
    """
    Keep only 1 per cluster_id (highest popularity per cluster) before capping 30.
    ev_map: id -> event
    """
    cluster_best = {}  # cluster_id -> (popularity, ev_id)
    singletons = []
    for eid in rel_ids:
        ev = ev_map.get(eid)
        if not ev:
            continue
        cid = ev.get("cluster_id")
        if cid:
            pop = ev.get("popularity", 0)
            if cid not in cluster_best or pop > cluster_best[cid][0]:
                cluster_best[cid] = (pop, eid)
        else:
            singletons.append(eid)
    # gather best per cluster
    deduped = [v[1] for v in cluster_best.values()] + singletons
    # sort by popularity desc before capping
    deduped_sorted = sorted(
        deduped, key=lambda eid: ev_map.get(eid, {}).get("popularity", 0), reverse=True
    )
    return deduped_sorted[:30]


def compute_relevance_for_query(q, valid_events, ev_map):
    """Apply relevance rule including facet check."""
    q_lat = q["lat"]
    q_lng = q["lng"]
    q_start = parse_time(q["time_window_start"])
    q_end = parse_time(q["time_window_end"])
    facet = q.get("facet")
    city = q["city"]
    category = q.get("category")
    radius = q["radius_km"]
    rel_ids = []
    for ev in valid_events:
        if ev["city"] != city:
            continue
        if category and ev["category"] != category:
            continue
        # facet / topic match
        if facet:
            if ev.get("topic") != facet:
                continue
        ev_start = parse_time(ev["start_time"])
        if not (q_start <= ev_start <= q_end):
            continue
        dist = haversine(q_lat, q_lng, ev["lat"], ev["lng"])
        if dist > radius:
            continue
        rel_ids.append(ev["id"])
    # dedup by cluster keeping highest popularity
    deduped = dedup_by_cluster(rel_ids, ev_map)
    return deduped


def make_single_query(
    anchor_ev, by_city, valid_events, base_anchor, anchored_flag, query_id
):
    """Create one query dict given anchored flag and optional anchor event.
    Spec: When anchored (80%): anchor = random valid event, set city=anchor.city, category=anchor.category, facet=anchor.topic
    """
    if anchored_flag and anchor_ev is not None:
        city = anchor_ev["city"]
        category = anchor_ev["category"]
        facet = anchor_ev.get("topic") or anchor_ev.get("category") or "general"
        ev_start = parse_time(anchor_ev["start_time"])
        delta = (ev_start - base_anchor).days
        if delta <= 2:
            time_phrase = "tomorrow"
            phrase_offset = 1
        elif delta <= 7:
            time_phrase = "this weekend"
            phrase_offset = 3
        elif delta <= 14:
            time_phrase = "next week"
            phrase_offset = 10
        elif delta <= 31:
            time_phrase = "this month"
            phrase_offset = 15
        else:
            time_phrase = "next month"
            phrase_offset = 45

        query_date = base_anchor + timedelta(days=random.randint(-3, 10))
        # Temporal semantics deterministic from query_date: window = query_date + phrase_offset ±20 fixed (no random jitter)
        # Per feedback: make hidden time windows follow standard deterministic rule from query_date so reasonable parsers can pass R01/R08
        window_start = (
            query_date + timedelta(days=phrase_offset) - timedelta(days=20)
        ).replace(hour=0, minute=0, second=0, microsecond=0)
        window_end = (
            query_date + timedelta(days=phrase_offset) + timedelta(days=20)
        ).replace(hour=23, minute=59, second=0, microsecond=0)
        q_lat = anchor_ev["lat"]
        q_lng = anchor_ev["lng"]
        radius_km = 50
        anchor_id = anchor_ev["id"]
    else:
        # random query: deterministic time windows from query_date per practice guide
        city = random.choice(list(by_city.keys()))
        category = random.choice(CATEGORIES + [None, None])
        candidates = by_city.get(city, [])
        if category:
            cat_cands = [e for e in candidates if e["category"] == category]
            if cat_cands:
                candidates = cat_cands
        if candidates:
            anchor_ev = random.choice(candidates)
            facet = anchor_ev.get("topic") or anchor_ev.get("category") or "general"
            q_lat = anchor_ev["lat"]
            q_lng = anchor_ev["lng"]
            anchor_id = anchor_ev["id"]
        else:
            anchor_ev = random.choice(valid_events) if valid_events else None
            facet = anchor_ev.get("topic") if anchor_ev else "general"
            q_lat = random.choice(by_city[city])["lat"] if by_city[city] else 0
            q_lng = random.choice(by_city[city])["lng"] if by_city[city] else 0
            anchor_id = anchor_ev["id"] if anchor_ev else None

        time_phrase, offset_start, offset_end = random.choice(TIME_PHRASES)
        query_date = base_anchor + timedelta(days=random.randint(-5, 25))
        # Deterministic: window = query_date + offset (no extra random jitter beyond phrase definition)
        window_start = (query_date + timedelta(days=offset_start)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        window_end = (query_date + timedelta(days=offset_end)).replace(
            hour=23, minute=59, second=0, microsecond=0
        )
        radius_km = 50

    activity = random.choice(ACTIVITIES)
    tmpl = random.choice(QUERY_TEMPLATES_FACET)
    cat_text = category if category else ""
    text = tmpl.format(
        city=city,
        category=cat_text,
        time_phrase=time_phrase,
        activity=activity,
        facet=facet,
    )
    text = " ".join(text.split())
    # clean double spaces
    text = " ".join(text.split())
    # If category empty, remove extra spaces around but keep facet
    # Ensure facet appears
    if facet.lower() not in text.lower():
        text = f"{facet} {text}"

    q = {
        "id": query_id,
        "text": text,
        "city": city,
        "category": category,
        "facet": facet,
        "time_window_start": window_start.isoformat(),
        "time_window_end": window_end.isoformat(),
        "radius_km": radius_km,
        "lat": round(q_lat, 6),
        "lng": round(q_lng, 6),
        "query_date": query_date.isoformat(),
        "anchored": anchored_flag,
        "_anchor_id": anchor_id,  # internal, stripped before output
    }
    return q


def generate_queries(events, n, seed):
    random.seed(seed)
    base_anchor = datetime(2026, 2, 15, 12, 0, 0)
    cutoff = datetime(2026, 2, 1)
    valid_events = [
        e for e in events if not e["is_spam"] and parse_time(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid_events}

    by_city = defaultdict(list)
    for e in valid_events:
        by_city[e["city"]].append(e)

    queries = []
    relevance = []

    for i in range(n):
        query_id = f"q_{i:05d}"
        # Try loop to guarantee >=1 relevant
        attempts = 0
        max_attempts = 30
        final_q = None
        final_rel = []
        # Decide anchored for this query (80%)
        is_anchored_intent = random.random() < 0.80

        # Precompute low-pop events to make popularity baseline fail (relevant low-pop)
        low_pop_events = [e for e in valid_events if e.get("popularity", 50) < 40]
        if not low_pop_events:
            low_pop_events = valid_events

        while attempts < max_attempts:
            anchored = is_anchored_intent if attempts == 0 else (random.random() < 0.80)
            if anchored and valid_events:
                # Prefer low popularity anchor 80% of time to make baseline (pop sort) fail
                if random.random() < 0.80:
                    anchor_ev = random.choice(low_pop_events)
                else:
                    anchor_ev = random.choice(valid_events)
            else:
                # for random case, still pick anchor for location/facet if possible, prefer low pop
                if random.random() < 0.60:
                    anchor_ev = (
                        random.choice(low_pop_events) if low_pop_events else None
                    )
                else:
                    anchor_ev = random.choice(valid_events) if valid_events else None
            q = make_single_query(
                anchor_ev, by_city, valid_events, base_anchor, anchored, query_id
            )
            rel_ids = compute_relevance_for_query(q, valid_events, ev_map)
            # Guarantee logic
            if len(rel_ids) == 0:
                # anchored -> force-include anchor ID
                if anchored and q.get("_anchor_id"):
                    # Check anchor still matches filters except maybe facet already matches
                    # Force include
                    rel_ids = [q["_anchor_id"]]
                else:
                    # random: resample unless attempts near end, then expand radius and force
                    if attempts >= max_attempts - 5:
                        # expand radius to 50km and recompute
                        q["radius_km"] = 50
                        rel_ids = compute_relevance_for_query(q, valid_events, ev_map)
                        if len(rel_ids) == 0 and q.get("_anchor_id"):
                            rel_ids = [q["_anchor_id"]]
                        if len(rel_ids) > 0:
                            final_q = q
                            final_rel = rel_ids
                            break
                    attempts += 1
                    continue
            final_q = q
            final_rel = rel_ids
            break

        # If still empty after attempts, force anchor from valid_events (prefer low pop)
        if not final_q:
            # fallback: create anchored query
            anchor_ev = random.choice(
                low_pop_events if low_pop_events else valid_events
            )
            final_q = make_single_query(
                anchor_ev, by_city, valid_events, base_anchor, True, query_id
            )
            final_q["radius_km"] = 50
            final_rel = compute_relevance_for_query(final_q, valid_events, ev_map)
            if len(final_rel) == 0:
                final_rel = [anchor_ev["id"]]

        # Strip internal _anchor_id before output, but keep facet and anchored
        out_q = {k: v for k, v in final_q.items() if not k.startswith("_")}
        # Ensure anchored bool stored
        out_q["anchored"] = final_q["anchored"]
        out_q["facet"] = final_q["facet"]
        queries.append(out_q)
        relevance.append({"query_id": query_id, "relevant_event_ids": final_rel})

    # Post-process: ensure no empty relevance left (should already be guaranteed)
    empty = [r for r in relevance if len(r["relevant_event_ids"]) == 0]
    if empty:
        print(
            f"Warning: {len(empty)} queries still empty after generation, forcing anchors"
        )
        for idx, rel in enumerate(relevance):
            if len(rel["relevant_event_ids"]) == 0:
                q = queries[idx]
                # find matching event by city/category/facet
                cands = [
                    e
                    for e in valid_events
                    if e["city"] == q["city"] and e.get("topic") == q.get("facet")
                ]
                if q.get("category"):
                    cands = [e for e in cands if e["category"] == q["category"]]
                if cands:
                    rel["relevant_event_ids"] = [cands[0]["id"]]
                else:
                    # any in city
                    cands2 = [e for e in valid_events if e["city"] == q["city"]]
                    if cands2:
                        rel["relevant_event_ids"] = [cands2[0]["id"]]

    return queries, relevance


def recompute_holdout_relevance(queries, valid_events, ev_map):
    """Recompute relevance for holdout queries with facet rule + dedup + guarantee."""
    new_rels = []
    for q in queries:
        rel_ids = compute_relevance_for_query(q, valid_events, ev_map)
        if len(rel_ids) == 0:
            # expand radius to 50km
            q_expanded = dict(q)
            q_expanded["radius_km"] = 50
            rel_ids = compute_relevance_for_query(q_expanded, valid_events, ev_map)
            if len(rel_ids) == 0:
                # force include an event matching city/facet if exists
                cands = [
                    e
                    for e in valid_events
                    if e["city"] == q["city"] and e.get("topic") == q.get("facet")
                ]
                if q.get("category"):
                    cands_f = [e for e in cands if e["category"] == q["category"]]
                    if cands_f:
                        cands = cands_f
                if cands:
                    rel_ids = [cands[0]["id"]]
        new_rels.append({"query_id": q["id"], "relevant_event_ids": rel_ids})
    return new_rels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-queries", type=str, default="queries.jsonl")
    parser.add_argument("--out-rel", type=str, default="relevance.jsonl")
    parser.add_argument("--out-holdout-queries", type=str, default=None)
    parser.add_argument("--out-holdout-rel", type=str, default=None)
    parser.add_argument("--holdout-n", type=int, default=100)
    args = parser.parse_args()

    with open(args.events) as f:
        events = [json.loads(line) for line in f]

    queries, relevance = generate_queries(events, args.n, args.seed)

    with open(args.out_queries, "w") as fq, open(args.out_rel, "w") as fr:
        for q, r in zip(queries, relevance):
            fq.write(json.dumps(q) + "\n")
            fr.write(json.dumps(r) + "\n")

    print(f"Wrote {len(queries)} queries to {args.out_queries}")
    print(f"Wrote {len(relevance)} relevance to {args.out_rel}")
    non_empty = sum(1 for r in relevance if len(r["relevant_event_ids"]) > 0)
    print(
        f"  queries with >=1 relevant: {non_empty}/{len(relevance)} ({non_empty / len(relevance):.1%})"
    )

    if args.out_holdout_queries and args.out_holdout_rel:
        h_queries, _ = generate_queries(events, args.holdout_n, seed=args.seed + 999)
        # after +35d shift, recompute with facet rule
        for q in h_queries:
            ws = parse_time(q["time_window_start"]) + timedelta(days=35)
            we = parse_time(q["time_window_end"]) + timedelta(days=35)
            q["time_window_start"] = ws.isoformat()
            q["time_window_end"] = we.isoformat()
        cutoff = datetime(2026, 2, 1)
        valid_events = [
            e
            for e in events
            if not e["is_spam"] and parse_time(e["start_time"]) >= cutoff
        ]
        ev_map = {e["id"]: e for e in valid_events}
        h_rel = recompute_holdout_relevance(h_queries, valid_events, ev_map)

        # Ensure holdout still has >=1 where possible; log
        h_non_empty = sum(1 for r in h_rel if len(r["relevant_event_ids"]) > 0)
        print(
            f"  holdout queries with >=1 relevant after shift: {h_non_empty}/{len(h_rel)} ({h_non_empty / len(h_rel):.1%})"
        )

        with (
            open(args.out_holdout_queries, "w") as fq,
            open(args.out_holdout_rel, "w") as fr,
        ):
            for q, r in zip(h_queries, h_rel):
                fq.write(json.dumps(q) + "\n")
                fr.write(json.dumps(r) + "\n")
        print(f"Wrote holdout {len(h_queries)} to {args.out_holdout_queries}")


if __name__ == "__main__":
    main()
