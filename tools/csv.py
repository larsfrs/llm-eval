#!/usr/bin/env python3
"""Reading and writing result rows in the single results.csv.

Rows are upserted: a combo rewrites its line after every suite it touched,
keeping the cells it already answered. Empty question cells are the unit
of work (cell-level resume): the score is written only once every cell is
filled, so an empty score marks a combo that still has questions to run.
sync_header grows the header to the configured columns, giving every
existing row its empty cells for new questions.
"""

import csv
import os

# Fixed columns; each suite appends its "{suite}_Q{n}" columns after these.
FIXED_FIELDS = [
    "model", "version", "run", "repeat",
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "repeat_penalty", "max_tokens",
    "score",
]
# Columns identifying a combo: its row is looked up, resumed and replaced
# by these — all params are identity, so changed params re-measure. (When
# min_p/repeat_penalty joined mid-v2, existing rows were backfilled with
# the llama.cpp defaults so their keys kept matching.)
KEY_FIELDS = FIXED_FIELDS[:-1]


def read_rows(csv_path):
    """All rows of the file keyed by combo key: {key: row}.

    Rows without this schema's key columns cannot be keyed and are
    skipped; they stay in the file untouched.
    """
    rows = {}
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = tuple(row[k] for k in KEY_FIELDS)
            except KeyError:
                continue  # foreign schema, cannot key it
            rows[key] = row
    return rows


def sync_header(csv_path, fieldnames):
    """Grow the file's header to the configured columns.

    Existing columns keep their position and data; configured columns
    missing from the file are appended, and every existing row gains an
    empty cell for them (new questions). Returns the added column names,
    empty when the file is missing or already in sync.
    """
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    added = [name for name in fieldnames if name not in header]
    if not added:
        return []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=header + added, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return added


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
