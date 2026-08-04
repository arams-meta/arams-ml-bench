#!/usr/bin/env python3
"""
Deterministic synthetic Facebook Events inventory generator.
No LLM, no external API - pure Python with fixed seed.
Bake output into Docker image via Dockerfile.

Usage:
  python generate_events.py --n 20000 --seed 42 --out /app/data/events.jsonl
"""

import argparse
import json
import random
import math
from datetime import datetime, timedelta

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

CITIES = {
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "New York": (40.7128, -74.0060),
    "Austin": (30.2672, -97.7431),
    "Seattle": (47.6062, -122.3321),
    "Chicago": (41.8781, -87.6298),
    "Miami": (25.7617, -80.1918),
    "Boston": (42.3601, -71.0589),
    "Denver": (39.7392, -104.9903),
    "Atlanta": (33.7490, -84.3880),
    "Portland": (45.5155, -122.6789),
    "Philadelphia": (39.9526, -75.1652),
}

VENUES = [
    "Downtown Plaza",
    "City Hall",
    "Central Park",
    "Riverside Hall",
    "Sunset Lounge",
    "Grand Theater",
    "Community Center",
    "Tech Hub",
    "Harbor View",
    "Market Square",
    "The Loft",
    "Skyline Rooftop",
    "Warehouse 22",
    "Union Hall",
    "Civic Auditorium",
]

ARTISTS = [
    "Luna Echo",
    "The Midnight Signals",
    "Cedar & Stone",
    "Neon Harbor",
    "Ava Rhodes",
    "Silver Lake Collective",
    "Juniper Jones",
    "Orbit 9",
    "The Riverside",
    "Mila Hart",
    "Crestline",
    "Blue Violet",
]

CUISINES = [
    "Mexican",
    "Thai",
    "Italian",
    "Vegan",
    "BBQ",
    "Japanese",
    "Indian",
    "Mediterranean",
]
TECH_TOPICS = [
    "AI",
    "Web3",
    "Cloud Native",
    "Data Engineering",
    "Robotics",
    "Cybersecurity",
]
TECH_SUBTOPICS = [
    "Hands-on Workshop",
    "Fireside Chat",
    "Lightning Talks",
    "Demo Night",
    "Hiring Mixer",
]
SPORTS = ["Soccer", "Basketball", "Running", "Yoga", "Cycling", "Tennis"]
ARTS = ["Painting", "Photography", "Sculpture", "Film Screening", "Poetry"]

TITLE_TEMPLATES = {
    "Music": [
        "{artist} Live at {venue}",
        "{artist} Concert - {city} Nights",
        "Acoustic Evening with {artist}",
    ],
    "Food": [
        "{cuisine} Festival 2026",
        "{cuisine} Street Food Market at {venue}",
        "{cuisine} Tasting Night",
    ],
    "Tech": [
        "{tech_topic} Meetup: {subtopic} at {venue}",
        "{tech_topic} Summit - {city}",
        "{tech_topic} {subtopic}",
    ],
    "Sports": [
        "{sport} Community {activity} at {venue}",
        "Weekend {sport} League",
        "{sport} & Brunch",
    ],
    "Arts": [
        "{art} {activity}: {venue} Showcase",
        "{art} Night Market",
        "{city} {art} Walk",
    ],
    "Nightlife": [
        "{city} Nightlife Crawl @ {venue}",
        "DJ Night at {venue}",
        "Rooftop Party - {venue}",
    ],
    "Community": [
        "{city} Community Cleanup",
        "Volunteer Day at {venue}",
        "Neighborhood Potluck - {venue}",
    ],
    "Business": [
        "{tech_topic} Startup Pitch @ {venue}",
        "Networking Mixer - {city}",
        "Founder Breakfast at {venue}",
    ],
}

DESC_TEMPLATES = {
    "Music": "Join {artist} for an unforgettable live performance at {venue} in {city}. Doors open at {hour}:00. All ages welcome. Expect a {vibe} atmosphere.",
    "Food": "Taste authentic {cuisine} dishes from local chefs at {venue}. Family-friendly, outdoor seating available. Part of {city}'s food week.",
    "Tech": "Connect with {tech_topic} builders in {city}. This {subtopic} covers practical tips, demos, and networking. Bring your laptop.",
    "Sports": "Weekly {sport} gathering for all skill levels. Meet at {venue}, {city}. Equipment provided. Free for members.",
    "Arts": "Explore {art} from local creators at {venue}. Curated by {city} Arts Council. Opening reception included.",
    "Nightlife": "Dance the night away at {venue}. {city}'s top DJs, craft cocktails, rooftop views. 21+.",
    "Community": "Give back to {city}. Meet at {venue}, supplies provided. Great for families and groups.",
    "Business": "Pitch, network, and learn. {tech_topic} founders sharing lessons at {venue}. Coffee and light breakfast included.",
}

VIBES = ["chill", "energetic", "intimate", "lively", "cozy", "festive"]


def jitter_lat_lng(lat, lng):
    return lat + random.uniform(-0.06, 0.06), lng + random.uniform(-0.06, 0.06)


def random_time(base, is_past=False):
    # base = 2026-02-01 for determinism
    if is_past:
        # 1 to 30 days in past
        delta_days = random.uniform(1, 30)
        dt = base - timedelta(days=delta_days, hours=random.uniform(0, 23))
    else:
        delta_days = random.uniform(1, 180)
        dt = base + timedelta(
            days=delta_days,
            hours=random.uniform(0, 23),
            minutes=random.choice([0, 15, 30, 45]),
        )
    # snap minutes
    return dt.replace(second=0, microsecond=0)


def create_base_tokens(cat, city, venue):
    artist = random.choice(ARTISTS)
    cuisine = random.choice(CUISINES)
    tech_topic = random.choice(TECH_TOPICS)
    subtopic = random.choice(TECH_SUBTOPICS)
    sport = random.choice(SPORTS)
    art = random.choice(ARTS)
    activity = random.choice(["Meetup", "Workshop", "Gathering", "Social", "Expo"])
    vibe = random.choice(VIBES)

    # Semantic facet per category — shared across cluster members via base_tokens
    if cat == "Tech":
        facet = tech_topic
    elif cat == "Food":
        facet = cuisine
    elif cat == "Music":
        facet = artist
    elif cat == "Sports":
        facet = sport
    elif cat == "Arts":
        facet = art
    elif cat in ("Nightlife", "Business", "Community"):
        # activity or venue for these categories
        facet = activity
    else:
        facet = activity

    tokens = {
        "artist": artist,
        "cuisine": cuisine,
        "tech_topic": tech_topic,
        "subtopic": subtopic,
        "sport": sport,
        "art": art,
        "activity": activity,
        "vibe": vibe,
        "city": city,
        "venue": venue,
        "facet": facet,
    }
    # popularity_base will be set by caller for clusters; set a default here
    # so singletons also have a stable base if needed later
    tokens["popularity_base"] = random.randint(20, 95)
    return tokens


def gen_single_event(
    event_id,
    city,
    cat,
    base_date,
    cluster_id=None,
    fixed_time=None,
    fixed_venue=None,
    fixed_lat=None,
    fixed_lng=None,
    base_tokens=None,
    paraphrase_idx=0,
):
    city_lat, city_lng = CITIES[city]
    lat, lng = (
        (fixed_lat, fixed_lng)
        if fixed_lat is not None
        else jitter_lat_lng(city_lat, city_lng)
    )
    venue = fixed_venue if fixed_venue else random.choice(VENUES)
    start = (
        fixed_time
        if fixed_time
        else random_time(base_date, is_past=(random.random() < 0.10))
    )
    duration_hours = random.choice([1, 1.5, 2, 3, 4, 5, 6, 8])
    end = start + timedelta(hours=duration_hours)

    tokens = base_tokens if base_tokens else create_base_tokens(cat, city, venue)
    # ensure city/venue override for consistency
    tokens = dict(tokens)
    tokens["city"] = city
    tokens["venue"] = venue

    # Title: for cluster members, keep same template but vary slightly via suffix/prefix
    # Pick template deterministically per cluster if possible
    if cluster_id is not None and base_tokens is not None:
        # reuse same title template for cluster members
        tmpl = tokens.get("title_template") or random.choice(
            TITLE_TEMPLATES.get(cat, TITLE_TEMPLATES["Community"])
        )
    else:
        tmpl = random.choice(TITLE_TEMPLATES.get(cat, TITLE_TEMPLATES["Community"]))
        tokens["title_template"] = tmpl

    title = tmpl.format(
        artist=tokens["artist"],
        venue=venue,
        city=city,
        cuisine=tokens["cuisine"],
        tech_topic=tokens["tech_topic"],
        subtopic=tokens["subtopic"],
        sport=tokens["sport"],
        art=tokens["art"],
        activity=tokens["activity"],
    )

    # For cluster members, add tiny variation to title to simulate near-duplicate listings
    # but keep core entity same (e.g., Yoga meetup remains Yoga meetup)
    if cluster_id is not None and paraphrase_idx > 0:
        # light paraphrase: add city prefix or change case, not change sport/cuisine
        variations = [
            lambda t: f"{t} - {city}",
            lambda t: f"{t} (Official)",
            lambda t: (
                t.replace("2026", "").strip() + f" 2026" if "2026" not in t else t
            ),
            lambda t: t,
        ]
        # deterministic variation based on idx
        title = variations[paraphrase_idx % len(variations)](title)

    desc_tmpl = DESC_TEMPLATES.get(cat, DESC_TEMPLATES["Community"])
    description = desc_tmpl.format(
        artist=tokens["artist"],
        venue=venue,
        city=city,
        cuisine=tokens["cuisine"],
        tech_topic=tokens["tech_topic"],
        subtopic=tokens["subtopic"],
        sport=tokens["sport"],
        art=tokens["art"],
        activity=tokens["activity"],
        hour=start.hour,
        vibe=tokens["vibe"],
    )

    # small paraphrase variant for dedup clusters
    if cluster_id is not None:
        suffixes = [
            " Don't miss it!",
            " Free entry. Limited spots.",
            " Presented by local organizers. Outdoor seating available.",
            " Family-friendly. All are welcome.",
        ]
        description = description + " " + suffixes[paraphrase_idx % len(suffixes)]
        # Slightly vary popularity to simulate different listings of same event
        # but keep close
        pop_base = tokens.get("popularity_base", random.randint(20, 95))
        popularity = max(1, min(100, pop_base + random.randint(-5, 5)))
    else:
        popularity = random.randint(1, 100)

    # Ensure facet exists (defensive for any base_tokens missing it)
    facet = tokens.get("facet")
    if not facet:
        if cat == "Tech":
            facet = tokens.get("tech_topic", "")
        elif cat == "Food":
            facet = tokens.get("cuisine", "")
        elif cat == "Music":
            facet = tokens.get("artist", "")
        elif cat == "Sports":
            facet = tokens.get("sport", "")
        elif cat == "Arts":
            facet = tokens.get("art", "")
        else:
            facet = tokens.get("activity", "") or tokens.get("venue", "")

    return {
        "id": event_id,
        "title": title.strip(),
        "description": description,
        "category": cat,
        "city": city,
        "venue": venue,
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "popularity": popularity,
        "is_spam": random.random() < 0.05,
        "cluster_id": cluster_id,
        "topic": facet,
        "_base_tokens": tokens,  # internal, stripped before output
    }


def generate(n, seed):
    random.seed(seed)
    base_date = datetime(2026, 2, 1, 10, 0, 0)
    events = []
    cluster_counter = 0
    event_counter = 0

    while len(events) < n:
        city = random.choice(list(CITIES.keys()))
        cat = random.choice(CATEGORIES)

        # 12% chance to start a near-duplicate cluster of size 2-3 (lowered from 15% to target ~20-25% clustered events)
        if random.random() < 0.12 and len(events) + 3 <= n:
            cluster_id = f"cluster_{cluster_counter}"
            cluster_counter += 1
            cluster_size = random.choice([2, 3])
            # shared attributes
            shared_time = random_time(
                base_date, is_past=(random.random() < 0.02)
            )  # even less likely past for clusters
            shared_venue = random.choice(VENUES)
            shared_lat, shared_lng = jitter_lat_lng(*CITIES[city])
            base_tokens = create_base_tokens(cat, city, shared_venue)
            base_tokens["popularity_base"] = random.randint(20, 95)
            # pre-pick title template for consistent cluster
            base_tokens["title_template"] = random.choice(
                TITLE_TEMPLATES.get(cat, TITLE_TEMPLATES["Community"])
            )
            for k in range(cluster_size):
                if len(events) >= n:
                    break
                eid = f"evt_{event_counter:06d}"
                event_counter += 1
                ev = gen_single_event(
                    eid,
                    city,
                    cat,
                    base_date,
                    cluster_id=cluster_id,
                    fixed_time=shared_time,
                    fixed_venue=shared_venue,
                    fixed_lat=shared_lat + random.uniform(-0.0003, 0.0003),
                    fixed_lng=shared_lng + random.uniform(-0.0003, 0.0003),
                    base_tokens=base_tokens,
                    paraphrase_idx=k,
                )
                ev.pop("_base_tokens", None)
                events.append(ev)
        else:
            eid = f"evt_{event_counter:06d}"
            event_counter += 1
            ev = gen_single_event(eid, city, cat, base_date)
            ev.pop("_base_tokens", None)
            events.append(ev)

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000, help="number of events")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="events.jsonl")
    args = parser.parse_args()

    events = generate(args.n, args.seed)
    with open(args.out, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"Wrote {len(events)} events to {args.out} (seed={args.seed})")
    # quick stats
    past = sum(
        1
        for e in events
        if datetime.fromisoformat(e["start_time"]) < datetime(2026, 2, 1)
    )
    spam = sum(1 for e in events if e["is_spam"])
    clustered = sum(1 for e in events if e["cluster_id"] is not None)
    print(
        f"  past: {past} ({past / len(events):.1%}), spam: {spam} ({spam / len(events):.1%}), clustered: {clustered} ({clustered / len(events):.1%})"
    )


if __name__ == "__main__":
    main()
