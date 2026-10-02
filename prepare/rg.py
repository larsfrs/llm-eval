#!/usr/bin/env python3
"""Sample reasoning-gym puzzles into prepared questions for eval/rg.py.

Items keep the library's own fields verbatim (question / answer / metadata)
plus our envelope fields (index / name / category / source).
Each dataset is sampled in difficulty tiers, easy first; every question
gets a human-readable name encoding dataset, tier and number, e.g.
"knights_knaves_easy_1". The name has semantic meaning for us only —
nothing in the eval pipeline reads it.
"""

import json
import os

from reasoning_gym import create_dataset

OUT_PATH = os.environ.get("OUT_PATH", "data/prepared/rg.json")
SOURCE = "rg"
SEED = int(os.environ.get("SEED", "42"))

# (dataset, [(tier name, config, count), ...]) — tiers go easy -> hard;
# the count is a per-cell knob (one question per cell by default)
TIERED_DATASETS = [
    ("knights_knaves", [
        ("easy", dict(n_people=2, depth_constraint=2, width_constraint=2), 1),
        ("medium", dict(n_people=3, depth_constraint=2, width_constraint=2), 1),
        ("hard", dict(n_people=3, depth_constraint=3, width_constraint=3), 1),
    ]),
    ("cryptarithm", [
        ("easy", dict(min_words=2, max_words=2), 1),
        ("hard", dict(min_words=3, max_words=3), 1),
    ]),
    ("zebra_puzzles", [
        ("easy", dict(num_people=3, num_characteristics=3), 1),
        ("medium", dict(num_people=4, num_characteristics=4), 1),
        ("hard", dict(num_people=5, num_characteristics=5), 1),
    ]),
]


def main():
    questions = []
    for dataset, tiers in TIERED_DATASETS:
        for tier, config, count in tiers:
            ds = create_dataset(dataset, seed=SEED, **config)
            for idx in range(count):
                item = ds[idx]
                questions.append({
                    "index": 0,  # assigned globally after all datasets
                    "name": f"{dataset}_{tier}_{idx + 1}",
                    "category": dataset,
                    "source": SOURCE,
                    "question": item["question"],
                    "answer": item["answer"],
                    "metadata": item["metadata"],
                })
        print(f"{dataset}: {sum(c for _, _, c in tiers)} items over "
              f"{len(tiers)} tiers (easy -> hard)")

    for i, q in enumerate(questions, 1):
        q["index"] = i

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
