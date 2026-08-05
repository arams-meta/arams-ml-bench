"""Shared answer-parsing and inventory-resolution helpers for the oracle judges.

The judges work only from the agent answer text plus events.jsonl. They never
look at flaw_type, human_score or category - those live in /tests/data and are
not available to a real solution.

Parsing note: inventory titles frequently embed their own venue, e.g.
"Founder Breakfast at Skyline Rooftop" held at venue "Skyline Rooftop". So the
clause "Founder Breakfast at Skyline Rooftop on 2026-04-27" is ambiguous - it
could be title+venue+date, or title+date where the title contains " at ".
resolve() therefore tries every reading and accepts whichever one the inventory
actually confirms, rather than committing to a split up front.
"""

import re

DATE = re.compile(r"\bon (?P<date>\d{4}-\d{2}-\d{2})")


def split_clauses(answer):
    """Break an answer into one clause per claimed event."""
    body = answer.split(":", 1)[1] if ":" in answer else answer
    parts = re.split(r";|\band\b", body)
    out = []
    for p in parts:
        p = p.strip().rstrip(".").strip()
        # Drop trailing commentary like "Both in Austin".
        if p and not re.match(r"^(both|all|these|they)\b", p, re.I):
            out.append(p)
    return out


NO_RESULTS = re.compile(r"could\s*n'?t find|no .* events? (found|in)", re.I)


def parse_listings(answer):
    """Each claimed event as {head, date}; head is the un-split descriptor."""
    out = []
    for clause in split_clauses(answer):
        m = DATE.search(clause)
        date = m.group("date") if m else None
        head = (clause[: m.start()] if m else clause).strip().rstrip(",").strip()
        if head:
            out.append({"head": head, "date": date})
    return out


def says_no_results(answer):
    return bool(NO_RESULTS.search(answer))


def _candidates(head):
    """Every plausible (title, venue) reading of a listing descriptor."""
    yield head, None
    if " at " in head:
        title, venue = head.rsplit(" at ", 1)
        yield title, venue
        yield title, None
    if " in " in head:
        yield head.rsplit(" in ", 1)[0], None


def resolve(listing, events):
    """The inventory row a listing refers to, or None if it invents one."""
    head = listing["head"]
    for title, venue in _candidates(head):
        matches = [e for e in events if e["title"] == title]
        if venue is not None:
            matches = [e for e in matches if e["venue"] == venue]
        if not matches:
            continue
        if listing.get("date"):
            dated = [e for e in matches if e["start_time"][:10] == listing["date"]]
            if dated:
                return dated[0]
        return matches[0]
    return None


def answer_cities(answer, cities):
    """Cities the answer names outright, including in the preamble before the colon."""
    found = set()
    for c in cities:
        if re.search(r"\bin " + re.escape(c) + r"\b", answer):
            found.add(c)
    return found


def stated_city(listing, cities):
    """A city the listing itself names, e.g. "... in Berlin"."""
    head = listing["head"]
    if " in " not in head:
        return None
    tail = head.rsplit(" in ", 1)[1].strip()
    # Only treat it as a city claim if it isn't part of a real event title.
    return tail if tail in cities or " " not in tail else None


def decide(dimension, focus, item, events, deterministic):
    """Ask the LLM for this dimension; fall back to the inventory check.

    The fallback is what makes oracle validation reproducible without API
    access. It is not a shortcut: both paths answer the same question against
    the same inventory.
    """
    try:
        from ._llm import ask
    except ImportError:
        from _llm import ask
    verdict = ask(dimension, focus, item, events)
    if verdict is not None:
        return verdict
    return deterministic(item, events)


def in_window(date, filters):
    if not date:
        return True
    start = (filters.get("time_window_start") or "")[:10]
    end = (filters.get("time_window_end") or "")[:10]
    if not start or not end:
        return True
    return start <= date <= end
