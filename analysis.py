#!/usr/bin/env python3
"""Overview graphs from results.csv.

One group of slim stacked bars per model, ranked best-first by total
questions passed in its anchor run — the deterministic run, or the
"ukn" run for models whose provider refuses sampling control (the
claude entries): within a group the anchor bar stands on the left
(marked d, or u for ukn) and every sampled run to its right, each bar
marked just below the axis. Colored segments = the datasets (one per
suite, e.g. static, linguistic_benchmark, rg), the bare pass count
printed inside each segment and each section's question total in the
legend. A dotted reference line marks the total number of configured
questions; inside each group a dotted black line crosses the bars at
the model's average total passed across all of its runs, with its
share of the total questions riding the line's right end.
Reference models (REFERENCE_MODELS) sit in their own block on the left,
cut off from the ranked rest by an orange dotted line. Below the model
names a thin strip carries one fact box per model, built from the model
db: total params, active params, release ("Mon YYYY").
A small note box in the top-right corner explains the groups in plain
words (the title carries no version).
Writes plots/results_overview.svg (the OUT env var overrides).
Run: uv run analysis.py
"""

import csv
import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # headless renderer: draw into a buffer, save to png
import matplotlib.pyplot as plt

from tools import settings

OUT_PATH = os.environ.get("OUT", os.path.join("plots", "results_overview.svg"))

# reference models: the frontier points the consumer-GPU models are
# measured against — their own block on the left, cut off from the ranked
# rest by an orange dotted line. Unmeasured references simply don't plot:
# the chart's model list is the ranked rows intersected with this set.
REFERENCE_MODELS = {
    "zai-glm-5-3",
    "claude-opus-4-8-no-think",
    "claude-opus-5-think",
    "claude-fable-5-1",
}

# the model db: models.json is the hand-maintained source of truth for
# model metadata — the README's model table (rendered by models.py) and
# the reference models' param labels come from it, not from hardcodes
MODELS_DB = "models.json"


def load_model_facts(models):
    """Per-model fact boxes from the model db: total, active, Mon YYYY.

    Every plotted model must carry a models.json entry (it feeds the
    README table too); a missing one exits loudly."""
    with open(MODELS_DB) as f:
        db = {model.get("name"): model for model in json.load(f)}
    facts = {}
    for name in models:
        entry = db.get(name)
        if entry is None:
            sys.exit(f"{MODELS_DB}: no entry for plotted model {name!r}")
        lines = [entry["params_total"], entry["params_active"]]
        released = entry.get("released", "")
        if released:
            try:
                lines.append(datetime.strptime(released, "%Y-%m").strftime("%Y %b"))
            except ValueError:
                sys.exit(f"{MODELS_DB}: {name} 'released' must be YYYY-MM, got {released!r}")
        facts[name] = "\n".join(lines)
    return facts


def load_dataset_labels():
    """Build the join map: suite name -> {question index: dataset label}.

    Each suite is one dataset in the overview chart; the rg questions
    still carry their puzzle categories and tier names for later, deeper
    analysis.
    """
    with open(settings.CONFIG_PATH) as f:
        config = json.load(f)

    labels_by_suite = {}
    for suite in config["suites"]:
        with open(suite["data"]) as f:
            questions = json.load(f)

        labels_by_suite[suite["name"]] = {
            question["index"]: suite["name"] for question in questions
        }
    return labels_by_suite


def load_current_version():
    """The active version label from eval.json (see VERSIONS.md)."""
    with open(settings.CONFIG_PATH) as f:
        config = json.load(f)
    return config.get("version", "")


def count_questions_per_dataset(labels_by_suite):
    """dataset label -> how many questions belong to it (a segment's max)."""
    question_totals = {}
    for question_labels in labels_by_suite.values():
        for dataset in question_labels.values():
            question_totals[dataset] = question_totals.get(dataset, 0) + 1
    return question_totals


def collect_results_per_model(labels_by_suite, version):
    """Three aggregates from results.csv, finished rows of the given version.

    Returns (det_counts_by_model, ukn_counts_by_model, sampled_counts_by_model):
    - det: model -> dataset -> passed count (the d bar segments)
    - ukn: same shape, for runs recorded as "ukn" — unknown sampling, the
      u bars (models whose provider refuses sampling control)
    - smp: model -> list of dataset -> passed count, one per sampled run

    Columns are resolved by NAME from the header, so files from before a
    schema change (or with columns in a different order) never misread.
    """
    if not os.path.isfile(settings.CSV_PATH):
        sys.exit(f"{settings.CSV_PATH}: not found — run eval.py first")

    with open(settings.CSV_PATH, newline="") as f:
        result_rows = csv.reader(f)
        header = next(result_rows, None)
        if header is None:
            sys.exit(f"{settings.CSV_PATH}: empty")

        # fixed columns by name: model, version, run, score
        column_index = {name: position for position, name in enumerate(header)}
        model_col = column_index["model"]
        run_col = column_index["run"]
        score_col = column_index["score"]
        version_col = column_index.get("version")

        # every question column is named "{suite}_Q{n}"; resolve it to its
        # dataset label once, straight from the header
        question_columns = []
        for position, column_name in enumerate(header):
            suite_name, separator, question_number = column_name.rpartition("_Q")
            if suite_name and separator and question_number.isdigit():
                label = labels_by_suite.get(suite_name, {}).get(int(question_number))
                question_columns.append((position, label))

        # det rows become the d bars, ukn rows the u bars (unknown
        # sampling), sampled runs the s bars; rows of other (or
        # pre-versioning) question sets do not count
        det_counts_by_model = {}
        ukn_counts_by_model = {}
        sampled_counts_by_model = {}
        for fields in result_rows:
            if len(fields) <= score_col:
                continue
            if fields[score_col] == "":
                continue  # interrupted combo, not counted
            row_version = (
                fields[version_col]
                if version_col is not None and len(fields) > version_col
                else ""
            )
            if row_version != version:
                continue

            passed_datasets = [
                label
                for position, label in question_columns
                if label is not None and len(fields) > position
                and fields[position] == "pass"
            ]

            run_name = fields[run_col]
            if run_name in ("det", "ukn"):
                counts_by_model = (
                    det_counts_by_model if run_name == "det" else ukn_counts_by_model
                )
                dataset_counts = counts_by_model.setdefault(fields[model_col], {})
                for label in passed_datasets:
                    dataset_counts[label] = dataset_counts.get(label, 0) + 1
            else:
                run_counts = {}
                for label in passed_datasets:
                    run_counts[label] = run_counts.get(label, 0) + 1
                sampled_counts_by_model.setdefault(fields[model_col], []).append(
                    run_counts
                )
    return det_counts_by_model, ukn_counts_by_model, sampled_counts_by_model


def plot_overview(det_counts_by_model, ukn_counts_by_model,
                  sampled_counts_by_model,
                  question_totals_by_dataset, total_questions, version):
    """Grouped stacked bars: the anchor run and every sampled run per model."""
    # rank models best-first by their anchor run's total passed — the
    # det run, or the ukn one for models whose provider refuses sampling
    # control; alphabetical tiebreak keeps the order stable — reference
    # models keep their own block on the left, ranked among themselves
    anchor_counts_by_model = dict(ukn_counts_by_model)
    anchor_counts_by_model.update(det_counts_by_model)
    ranked = sorted(
        anchor_counts_by_model,
        key=lambda model: (-sum(anchor_counts_by_model[model].values()), model),
    )
    models = (
        [model for model in ranked if model in REFERENCE_MODELS]
        + [model for model in ranked if model not in REFERENCE_MODELS]
    )
    if not models:
        sys.exit(f"{settings.CSV_PATH}: no completed anchor rows yet (det or ukn)")
    # fact boxes come from the model db; a plotted model missing from
    # models.json exits loudly (load_model_facts)
    model_facts = load_model_facts(models)

    dataset_names = sorted(
        {dataset for counts in det_counts_by_model.values() for dataset in counts}
    )

    # one group of slim bars per model: the anchor bar on the left
    # (d — or u, unknown sampling), every sampled run to its right,
    # each marked just below the axis — and below the names a thin strip
    # with one fact box per model
    figure = plt.figure(figsize=(12, 7.5))
    grid = figure.add_gridspec(2, 1, height_ratios=[12, 1])
    axis = figure.add_subplot(grid[0])
    facts_axis = figure.add_subplot(grid[1])
    bar_width, bar_gap, group_spacing = 0.22, 0.05, 1.5
    below_axis = axis.get_xaxis_transform()  # x in data, y in axes units
    group_centers = []
    for group_number, model in enumerate(models):
        anchor_runs = []
        if model in det_counts_by_model:
            anchor_runs.append(("d", det_counts_by_model[model]))
        if model in ukn_counts_by_model:
            anchor_runs.append(("u", ukn_counts_by_model[model]))
        runs = anchor_runs + [
            ("s", section_counts)
            for section_counts in sampled_counts_by_model.get(model, [])
        ]
        group_center = group_number * group_spacing
        group_width = len(runs) * bar_width + (len(runs) - 1) * bar_gap
        left = group_center - group_width / 2
        for run_number, (mark, section_counts) in enumerate(runs):
            bar_x = left + run_number * (bar_width + bar_gap)
            bottom = 0
            for dataset_index, dataset in enumerate(dataset_names):
                height = section_counts.get(dataset, 0)
                axis.bar(
                    bar_x, height, bottom=bottom, width=bar_width,
                    color=f"C{dataset_index}",
                )
                # bare pass count inside each segment; the section's
                # total lives in the legend
                if height:
                    axis.text(
                        bar_x, bottom + height / 2, str(height),
                        ha="center", va="center", fontsize=8, color="white",
                    )
                bottom += height
            axis.text(
                bar_x, -0.02, mark, transform=below_axis,
                ha="center", va="top", fontsize=8, color="black",
            )

        # dotted line across the group's bars at the model's average
        # total passed over all of its runs (the anchor and every s one);
        # bar() centers each bar on x, so the group's visual edge sits
        # half a bar to the left of left
        average_total = sum(
            sum(section_counts.values()) for _, section_counts in runs
        ) / len(runs)
        group_left = left - bar_width / 2
        axis.plot(
            [group_left, group_left + group_width],
            [average_total, average_total],
            linestyle=":", color="black",
        )
        # the average's share of all questions rides the line's right end
        axis.text(
            group_left + group_width + 0.05, average_total,
            f"{100 * average_total / total_questions:.0f}%",
            va="center", ha="left", fontsize=8, color="black",
        )
        group_centers.append(group_center)

    # orange dotted line between the reference block and the ranked rest
    references_plotted = sum(model in REFERENCE_MODELS for model in models)
    if 0 < references_plotted < len(models):
        axis.axvline(
            (references_plotted - 0.5) * group_spacing,
            linestyle=":", color="orange", linewidth=3,
        )

    # dotted reference line: every question of every configured suite
    reference_line = axis.axhline(
        total_questions, linestyle=":", color="gray",
        label=f"{total_questions} questions total",
    )

    # axes, legend, and the png
    axis.set_xlim(group_centers[0] - 0.8, group_centers[-1] + 0.8)
    axis.set_xticks(group_centers)
    axis.set_xticklabels(models, rotation=90)
    axis.tick_params(axis="x", pad=18)  # model names sit below the run marks
    axis.set_ylabel("questions passed")
    axis.set_title("passed questions per model")

    # legend: one entry per section, carrying its question total (the
    # counts inside the bars are bare; the /total lives here)
    section_handles = [
        matplotlib.patches.Patch(
            color=f"C{dataset_index}",
            label=f"{dataset} ({question_totals_by_dataset[dataset]})",
        )
        for dataset_index, dataset in enumerate(dataset_names)
    ]
    # proxy handle standing in for every group's average line
    average_handle = matplotlib.lines.Line2D(
        [0], [0], linestyle=":", color="black",
        label="model average (all runs)",
    )
    axis.legend(
        handles=[*section_handles, reference_line, average_handle],
        title="dataset", loc="upper right", bbox_to_anchor=(1.0, 0.68),
    )

    # plain-words note box pinned to the top-right corner: the legend is
    # anchored just below it — loc="best" cannot see the note and would
    # collide with it
    axis.text(
        0.98, 0.97,
        "Each group: the d bar is the deterministic run,\n"
        "the s bars are its sampled runs; a u bar is a run\n"
        "whose sampling the provider leaves at its default\n"
        "(claude: no greedy pinning available). Counts inside\n"
        "the bars are passes per section; each section's\n"
        "total is in the legend. The sections are described\n"
        "in the sections below. The orange dotted line cuts\n"
        "off the reference models (left block) from the rest.",
        transform=axis.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"},
    )
    # the facts strip below the names: one rounded box per model,
    # carrying its db facts (total, active, release)
    facts_axis.set_xlim(axis.get_xlim())
    facts_axis.set_ylim(0, 1)
    facts_axis.axis("off")
    for center, model in zip(group_centers, models):
        facts_axis.text(
            center, 0.5, model_facts[model],
            ha="center", va="center", fontsize=7,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "gray"},
        )

    figure.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    figure.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH} (version {version})")

    # per-model totals in the console: the anchor run plus every
    # sampled run's total
    for model in models:
        anchor = det_counts_by_model.get(model) or ukn_counts_by_model[model]
        anchor_kind = "det" if model in det_counts_by_model else "ukn"
        anchor_total = sum(anchor.values())
        sampled_runs = sampled_counts_by_model.get(model, [])
        if sampled_runs:
            run_totals = ", ".join(
                str(sum(run.values())) for run in sampled_runs
            )
            print(f"  {model}: {anchor_total} {anchor_kind}; sampled runs: {run_totals}")
        else:
            print(f"  {model}: {anchor_total} ({anchor_kind} only)")


def main():
    labels_by_suite = load_dataset_labels()
    question_totals_by_dataset = count_questions_per_dataset(labels_by_suite)
    total_questions = sum(question_totals_by_dataset.values())
    version = load_current_version()
    det_counts_by_model, ukn_counts_by_model, sampled_counts_by_model = (
        collect_results_per_model(labels_by_suite, version)
    )
    plot_overview(
        det_counts_by_model, ukn_counts_by_model, sampled_counts_by_model,
        question_totals_by_dataset, total_questions, version,
    )


if __name__ == "__main__":
    main()
