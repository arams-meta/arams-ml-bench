#!/usr/bin/env python3
"""Strip structured fields from queries to leave free-form only id+text+query_date for agent"""

import json
import sys

if len(sys.argv) < 3:
    print("usage: strip_queries.py in.jsonl out.jsonl")
    sys.exit(1)

in_path, out_path = sys.argv[1], sys.argv[2]

with open(in_path) as fin, open(out_path, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        minimal = {
            "id": obj["id"],
            "text": obj["text"],
            "query_date": obj["query_date"],
        }
        fout.write(json.dumps(minimal) + "\n")

print(f"Stripped {in_path} -> {out_path} to free-form only")
