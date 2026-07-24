"""
Baseline floor: beats trivial popularity heuristic that ignores facet.
Per TBR: recompute improved recall from retrieved.jsonl vs hidden/recomputed ground truth, not trusting report.
Ground truth from /opt/eval first (hidden) or host hidden full, fallback to free-form extraction.
"""

import json
import math
import os
import random
from datetime import datetime, timedelta
from collections import defaultdict

random.seed(42)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


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


def recall_at_k(retrieved, relevant, k=10):
    if not relevant:
        return None
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def dedup_by_cluster_ids(rel_ids, ev_map):
    cluster_best = {}
    single = []
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
            single.append(eid)
    deduped = [v[1] for v in cluster_best.values()] + single
    deduped_sorted = sorted(
        deduped, key=lambda eid: ev_map.get(eid, {}).get("popularity", 0), reverse=True
    )
    return deduped_sorted[:30]


def compute_relevance(q, ev_map, by_city):
    city = q.get("city")
    if not city:
        return []
    facet = q.get("facet")
    cat = q.get("category")
    try:
        q_lat = q["lat"]
        q_lng = q["lng"]
        ws = datetime.fromisoformat(q["time_window_start"])
        we = datetime.fromisoformat(q["time_window_end"])
        radius = q["radius_km"]
        has = True
    except:
        has = False
        ws = we = None
        q_lat = q_lng = None
        radius = 500
    rel = []
    for ev in by_city.get(city, []):
        if cat and ev["category"] != cat:
            continue
        if facet and ev.get("topic") != facet:
            continue
        if has:
            ev_start = datetime.fromisoformat(ev["start_time"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q_lat, q_lng, ev["lat"], ev["lng"]) > radius:
                continue
        rel.append(ev["id"])
    return dedup_by_cluster_ids(rel, ev_map)


def load_ground_truth():
    events_paths = [
        "/opt/eval/events.jsonl",
        "/app/data/events.jsonl",
        "data/events.jsonl",
    ]
    queries_paths = [
        "/opt/eval/queries.jsonl",
        "data/queries_full_hidden.jsonl",
        "/app/data/queries.jsonl",
        "data/queries.jsonl",
    ]
    rel_paths = [
        "/opt/eval/relevance.jsonl",
        "data/relevance_hidden.jsonl",
        "/app/data/relevance.jsonl",
    ]

    events_file = first_existing(events_paths) or "/app/data/events.jsonl"
    queries_file = first_existing(queries_paths) or "/app/data/queries.jsonl"

    events = [json.loads(l) for l in open(events_file)]
    queries = [json.loads(l) for l in open(queries_file)]

    relevance = {}
    for rp in rel_paths:
        if os.path.exists(rp):
            relevance = {r["query_id"]: r["relevant_event_ids"] for r in load_jsonl(rp)}
            break

    cutoff = datetime(2026, 2, 1)
    valid_events = [
        e
        for e in events
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid_events}
    by_city = defaultdict(list)
    for e in valid_events:
        by_city[e["city"]].append(e)

    if not relevance:
        all_cities = list(by_city.keys())
        all_categories = [
            "Music",
            "Sports",
            "Food",
            "Tech",
            "Arts",
            "Nightlife",
            "Community",
            "Business",
        ]
        all_topics = sorted(
            set(e.get("topic", "") for e in valid_events if e.get("topic"))
        )
        city_centers = {
            city: (
                sum(ev["lat"] for ev in evs) / len(evs),
                sum(ev["lng"] for ev in evs) / len(evs),
            )
            for city, evs in by_city.items()
        }
        city_facet_events = defaultdict(list)
        for ev in valid_events:
            city_facet_events[(ev["city"], ev.get("topic"))].append(ev)

        def extract(text, qdate_str=None):
            low = text.lower()
            city = None
            for c in all_cities:
                if c.lower() in low:
                    city = c
                    break
            cat = None
            for cc in all_categories:
                if cc.lower() in low:
                    cat = cc
                    break
            facet = None
            for topic in sorted(all_topics, key=len, reverse=True):
                if topic and topic.lower() in low:
                    facet = topic
                    break
            qdate = (
                datetime.fromisoformat(qdate_str)
                if qdate_str
                else datetime(2026, 2, 15)
            )
            offset = 15
            if "tomorrow" in low:
                offset = 1
            elif "weekend" in low:
                offset = 3
            elif "next week" in low:
                offset = 10
            elif "next month" in low:
                offset = 45
            est = qdate + timedelta(days=offset)
            anchor = None
            if city and facet:
                cands = city_facet_events.get((city, facet), [])
                if not cands:
                    cands = [
                        e for e in by_city.get(city, []) if e.get("topic") == facet
                    ]
                if cands:
                    cands_sorted = sorted(
                        cands,
                        key=lambda ev: abs(
                            (
                                datetime.fromisoformat(ev["start_time"]) - est
                            ).total_seconds()
                        ),
                    )
                    anchor = cands_sorted[0]
            if anchor:
                ev_start = datetime.fromisoformat(anchor["start_time"])
                ws = (ev_start - timedelta(days=30)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                we = (ev_start + timedelta(days=30)).replace(
                    hour=23, minute=59, second=0, microsecond=0
                )
                lat = anchor["lat"]
                lng = anchor["lng"]
                radius = 50
            else:
                ws = datetime(2026, 1, 1, 0, 0, 0)
                we = datetime(2026, 7, 1, 23, 59, 0)
                if city and city in city_centers:
                    lat, lng = city_centers[city]
                    radius = 50
                else:
                    lat, lng = 0, 0
                    radius = 500
            return {
                "city": city,
                "category": cat,
                "facet": facet,
                "time_window_start": ws.isoformat(),
                "time_window_end": we.isoformat(),
                "lat": lat,
                "lng": lng,
                "radius_km": radius,
            }

        for q in queries:
            if "city" in q and "facet" in q:
                relevance[q["id"]] = compute_relevance(q, ev_map, by_city)
            else:
                ext = extract(q["text"], q.get("query_date"))
                if ext["city"] and ext["facet"]:
                    relevance[q["id"]] = compute_relevance(ext, ev_map, by_city)

    return events, queries, relevance


def trivial_baseline():
    events, queries, relevance = load_ground_truth()
    cutoff = datetime(2026, 2, 1)
    valid_events = [
        e
        for e in events
        if not e.get("is_spam") and datetime.fromisoformat(e["start_time"]) >= cutoff
    ]
    ev_map = {e["id"]: e for e in valid_events}
    by_city = defaultdict(list)
    for e in valid_events:
        by_city[e["city"]].append(e)

    recalls = []
    for q in queries:
        qid = q.get("id")
        rel = relevance.get(qid, [])
        if not rel or "city" not in q or "time_window_start" not in q:
            continue
        ws = datetime.fromisoformat(q["time_window_start"])
        we = datetime.fromisoformat(q["time_window_end"])
        q_lat, q_lng = q["lat"], q["lng"]
        radius = q["radius_km"]
        city = q["city"]
        cat = q.get("category")
        cand = by_city.get(city, [])
        filtered = []
        for e in cand:
            if cat and e["category"] != cat:
                continue
            ev_start = datetime.fromisoformat(e["start_time"])
            if not (ws <= ev_start <= we):
                continue
            if haversine(q_lat, q_lng, e["lat"], e["lng"]) > radius:
                continue
            filtered.append(e)
        cluster_best = {}
        singletons = []
        for e in filtered:
            cid = e.get("cluster_id")
            if cid:
                if (
                    cid not in cluster_best
                    or e["popularity"] > cluster_best[cid]["popularity"]
                ):
                    cluster_best[cid] = e
            else:
                singletons.append(e)
        deduped = list(cluster_best.values()) + singletons
        sorted_by_pop = sorted(deduped, key=lambda x: x["popularity"], reverse=True)
        retrieved = [e["id"] for e in sorted_by_pop[:10]]
        r = recall_at_k(retrieved, rel, k=10)
        if r is not None:
            recalls.append(r)
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_improved_recall_from_output():
    _, _, relevance = load_ground_truth()
    rel_sets = {k: set(v) for k, v in relevance.items()}
    retrieved = {}
    if os.path.exists("/output/retrieved.jsonl"):
        with open("/output/retrieved.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                retrieved[obj["query_id"]] = obj.get("retrieved_ids", [])
    recalls = []
    for qid, rel in rel_sets.items():
        if not rel:
            continue
        ret = retrieved.get(qid, [])
        if not ret:
            recalls.append(0.0)
            continue
        recalls.append(len(set(ret[:10]) & rel) / len(rel))
    return sum(recalls) / len(recalls) if recalls else 0.0


def test_beats_trivial():
    improved = compute_improved_recall_from_output()
    base = trivial_baseline()
    os.makedirs("/logs/verifier", exist_ok=True)
    with open("/logs/verifier/baseline_score.txt", "w") as bf:
        bf.write(f"agent={improved:.3f} baseline_popularity={base:.3f}\n")
    # If trivial baseline couldn't compute (0.0), use 0.27 as reference from earlier analysis
    if base == 0.0:
        base = 0.27
    assert improved > base + 0.05, (
        f"Agent {improved:.3f} should beat trivial popularity baseline {base:.3f} by 0.05"
    )
