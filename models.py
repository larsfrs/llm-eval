#!/usr/bin/env python3
"""Render the README's model table from models.json.

models.json is the hand-maintained model db — the source of truth for
model metadata (params, architecture, thinking, quantization, release,
licence, status). The README section between the two markers is generated
from it: never hand-edit the table, edit the db and rerun this.

    uv run models.py

The markers delimiting the section in README.md:
    <!-- models: start --> ... <!-- models: end -->
"""

import json
import sys

DB_PATH = "models.json"
README_PATH = "README.md"
START, END = "<!-- models: start -->", "<!-- models: end -->"

COLUMNS = (
    "Model", "Params (active/effective)", "Params (total)",
    "Architecture", "Thinking", "Quantization", "Released", "Licence",
)
FIELDS = (
    "params_active", "params_total", "architecture",
    "thinking", "quantization", "released", "licence",
)


def render_table(models):
    """The markdown table: one row per model, its status as the leading tag."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(" --- " for _ in COLUMNS) + " |",
    ]
    for model in models:
        if not model.get("name"):
            sys.exit(f"{DB_PATH}: a model entry has no 'name'")
        tag = f"`[{model.get('status', 'testing').title()}]` "
        cells = [tag + model["name"]] + [
            str(model.get(field, "")) for field in FIELDS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    try:
        with open(DB_PATH) as f:
            models = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{DB_PATH}: not found")
    with open(README_PATH) as f:
        readme = f.read()
    if START not in readme or END not in readme:
        sys.exit(f"{README_PATH}: missing the {START!r} / {END!r} markers")

    head, _, rest = readme.partition(START)
    _, _, tail = rest.partition(END)
    table = render_table(models)
    with open(README_PATH, "w") as f:
        f.write(f"{head}{START}\n{table}\n{END}{tail}")
    print(f"{README_PATH}: model table rendered from {DB_PATH} ({len(models)} models)")


if __name__ == "__main__":
    main()
