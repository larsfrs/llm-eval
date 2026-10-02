#!/usr/bin/env python3
"""Fetch the linguistic benchmark and write it in the canonical question format.

Canonical format (what eval consumes from data/prepared/):
  - Multiple choice: index, name (semantic label), category, source,
    question, human_answer, correct_answer (full choice text),
    multiple_choice (list), answer_index (1-based, resolved here)
  - Numeric: same, but multiple_choice is null and correct_answer is a bare
    JSON number (no answer_index)

answer_index is resolved with exact string matching against the choices, so
eval only ever has to compare numbers, never long text.
"""

import json
import os
import sys

import requests

URL = os.environ.get(
    "URL",
    "https://raw.githubusercontent.com/autogenai/easy-problems-that-llms-get-wrong"
    "/main/linguistic_benchmark_multi_choice.json",
)
OUT_PATH = os.environ.get("OUT_PATH", "data/prepared/linguistic_benchmark.json")
SOURCE = "linguistic_benchmark"


def convert(entry):
    choices = entry["multiple_choice"]
    hits = [i for i, c in enumerate(choices, 1) if c == entry["correct_answer"]]
    if len(hits) != 1:
        sys.exit(
            f"Q{entry['index']}: correct_answer {entry['correct_answer']!r} "
            f"matched {len(hits)} of {len(choices)} choices"
        )
    return {
        "index": entry["index"],
        "name": f"{entry['category'].lower().replace(' ', '_')}_{entry['index']}",
        "category": entry["category"],
        "source": SOURCE,
        "question": entry["question"],
        "human_answer": entry["human_answer"],
        "correct_answer": entry["correct_answer"],
        "multiple_choice": choices,
        "answer_index": hits[0],
    }


def main():
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    raw = r.json()
    if not isinstance(raw, list) or not raw:
        sys.exit(f"Unexpected payload from {URL}")

    questions = [convert(e) for e in raw]
    indices = [q["index"] for q in questions]
    if len(set(indices)) != len(indices):
        sys.exit(f"Duplicate indices in payload: {sorted(indices)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
