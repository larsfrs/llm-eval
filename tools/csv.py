#!/usr/bin/env python3
"""Reading and writing result rows in the single results.csv.

Rows are upserted: a combo writes its row after every suite (suites not
run yet stay empty), replacing its previous partial line in place. The
score is written only when the combo finishes — an empty score marks an
interrupted combo, so reruns pick it up and replace the partial line.
"""

import csv
import os

# Fixed columns; each suite appends its "{suite}_Q{n}" columns after these.
FIXED_FIELDS = [
    "model", "version", "run", "repeat",
    "temperature", "top_p", "top_k", "presence_penalty", "max_tokens",
    "score",
]
# Columns identifying a combo, used to find and skip already-completed rows.
KEY_FIELDS = FIXED_FIELDS[:-1]


def read_completed(csv_path):
    """Return the combo keys of COMPLETED rows.

    A row counts as complete only once its score is written. A row with an
    empty score belongs to an interrupted combo: it does not count as
    done, and rerunning replaces its partial line in place.
    """
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = tuple(row[k] for k in KEY_FIELDS)
            except KeyError:
                break  # foreign schema, treat as fresh
            if row.get("score", ""):
                done.add(key)
    return done


def upsert_row(csv_path, row, fieldnames):
    """Write the row for its combo key, replacing its line in place.

    The file's columns are never dropped: the header only ever grows (new
    suites append their columns at the end). A suite removed from
    eval.json keeps its columns and old data untouched; newer rows simply
    leave them empty. The whole file is rewritten (one row per combo, so
    it stays small); each combo stays on its original line.
    """
    # NOTE: the row holds native types (int/float), the file holds strings
    key = tuple(str(row[k]) for k in KEY_FIELDS)
    rows, header, replaced = [], [], False
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            for existing in reader:
                try:
                    existing_key = tuple(existing[k] for k in KEY_FIELDS)
                except KeyError:
                    rows.append(existing)  # foreign line, keep untouched
                    continue
                if existing_key == key and not replaced:
                    rows.append(row)  # this combo's line: replace it
                    replaced = True
                else:
                    rows.append(existing)
    if not replaced:
        rows.append(row)

    # the header only ever grows: existing columns first, new ones appended
    fieldnames = header + [name for name in fieldnames if name not in header]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
