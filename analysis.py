#!/usr/bin/env python3
"""Overview graphs from results.csv.

One stacked bar per model, ranked best-first by total questions passed in
the deterministic run: y axis = passed questions of the deterministic
run, colored segments = the datasets (one per suite, e.g. static,
linguistic_benchmark, rg). Black whiskers centered on each
bar top show det ± 5*std of the sampled runs (the README methodology), and
a dotted reference line marks the total number of configured questions.
A small note box in the top-right corner explains bars vs whiskers in
plain words (the title carries no version).
Writes plots/results_overview.svg (the OUT env var overrides).
Run: uv run analysis.py
"""

import csv
import json
import math
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")  # headless renderer: draw into a buffer, save to png
import matplotlib.pyplot as plt

from tools import settings

OUT_PATH = os.environ.get("OUT", os.path.join("plots", "results_overview.svg"))

# the README methodology: score = deterministic run +- 5 * std of the
# sampled runs
ERROR_STDEV_MULTIPLE = 5


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
    """Two aggregates from results.csv, finished rows of the given version.

    Returns (det_counts_by_model, sampled_totals_by_model):
    - det: model -> dataset -> passed count (the bar segments)
    - smp: model -> list of passed totals, one entry per sampled run

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

        # det rows become the bar segments, sampled rows only the error;
        # rows of other (or pre-versioning) question sets do not count
        det_counts_by_model = {}
        sampled_totals_by_model = {}
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

            if fields[run_col] == "det":
                dataset_counts = det_counts_by_model.setdefault(fields[model_col], {})
                for label in passed_datasets:
                    dataset_counts[label] = dataset_counts.get(label, 0) + 1
            else:
                sampled_totals_by_model.setdefault(fields[model_col], []).append(
                    len(passed_datasets)
                )
    return det_counts_by_model, sampled_totals_by_model


def plot_overview(det_counts_by_model, sampled_totals_by_model,
                 question_totals_by_dataset, total_questions, version):
    """Stacked bar chart with error whiskers and a reference line."""
    # rank models best-first by their det total passed; alphabetical
    # tiebreak keeps the order stable
    models = sorted(
        det_counts_by_model,
        key=lambda model: (-sum(det_counts_by_model[model].values()), model),
    )
    if not models:
        sys.exit(f"{settings.CSV_PATH}: no deterministic rows yet")

    dataset_names = sorted(
        {dataset for counts in det_counts_by_model.values() for dataset in counts}
    )

    # build the stacked bars: each dataset adds its segment on top of the
    # previous ones, per model, with its passed count printed inside
    figure, axis = plt.subplots(figsize=(8, 6))  # extra height for upright names
    bottom_offsets = [0] * len(models)
    for dataset in dataset_names:
        segment_heights = [
            det_counts_by_model[model].get(dataset, 0) for model in models
        ]
        axis.bar(models, segment_heights, bottom=bottom_offsets, label=dataset,
                 width=0.6)

        # the "passed/total" fraction goes beside each segment — the space
        # above the bar belongs to the error whisker (empty segments get
        # no label)
        segment_total = question_totals_by_dataset[dataset]
        for bar_position, (offset, height) in enumerate(
            zip(bottom_offsets, segment_heights)
        ):
            if height > 0:
                axis.text(
                    bar_position + 0.32, offset + height / 2,
                    f"{height}/{segment_total}",
                    ha="left", va="center", fontsize=9, color="black",
                )
        bottom_offsets = [
            offset + height
            for offset, height in zip(bottom_offsets, segment_heights)
        ]

    # black whiskers centered on the bar top: det ± 5 * std of the sampled
    # runs — the segment counts sit beside the bar, so the lower arm
    # reaching into the bar is unobstructed. Models run deterministically
    # only get NaN, which skips that bar's error bar entirely
    error_sizes = []
    sampled_run_counts = []
    for model in models:
        sampled_totals = sampled_totals_by_model.get(model, [])
        if len(sampled_totals) >= 2:
            error_sizes.append(
                ERROR_STDEV_MULTIPLE * statistics.stdev(sampled_totals)
            )
            sampled_run_counts.append(len(sampled_totals))
        else:
            error_sizes.append(float("nan"))
            sampled_run_counts.append(0)
    axis.errorbar(
        models, bottom_offsets, yerr=error_sizes, fmt="none",
        ecolor="black", capsize=4,
    )

    # mini label on each whisker: how many sampled runs it is built from
    # (sample counts vary per model, so the number travels with the whisker)
    for bar_position, (total, error, run_count) in enumerate(
        zip(bottom_offsets, error_sizes, sampled_run_counts)
    ):
        if not math.isnan(error):
            axis.text(
                bar_position, total + error, f"smp: {run_count}",
                ha="center", va="bottom", fontsize=7, color="black",
            )

    # dotted reference line: every question of every configured suite
    axis.axhline(
        total_questions, linestyle=":", color="gray",
        label=f"{total_questions} questions total",
    )

    # axes, legend, and the png
    axis.set_xlim(-0.55, len(models) - 0.5 + 0.45)  # room for the side fractions
    axis.set_ylabel("questions passed")
    axis.set_xlabel("model")
    # upright model names: horizontal ones crowd once more models join
    axis.tick_params(axis="x", labelrotation=90)
    axis.set_title("passed questions per model")

    # plain-words note box pinned to the top-right corner: bars vs
    # whiskers. The legend is anchored just below it — loc="best" cannot
    # see the note and would collide with it
    axis.text(
        0.98, 0.97,
        f"Each bar is one deterministic run.\n"
        f"Whiskers, where present, show ±{ERROR_STDEV_MULTIPLE}·std "
        f"of the sampled runs.",
        transform=axis.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"},
    )
    axis.legend(title="dataset", loc="upper right", bbox_to_anchor=(1.0, 0.86))
    figure.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    figure.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH} (version {version})")

    # per-model totals and error in the console
    for model, total, error in zip(models, bottom_offsets, error_sizes):
        if math.isnan(error):
            print(f"  {model}: {total} passed (deterministic only)")
        else:
            print(
                f"  {model}: {total} passed "
                f"(det ± {ERROR_STDEV_MULTIPLE}*std = ±{error:.1f})"
            )


def main():
    labels_by_suite = load_dataset_labels()
    question_totals_by_dataset = count_questions_per_dataset(labels_by_suite)
    total_questions = sum(question_totals_by_dataset.values())
    version = load_current_version()
    det_counts_by_model, sampled_totals_by_model = collect_results_per_model(
        labels_by_suite, version
    )
    plot_overview(
        det_counts_by_model, sampled_totals_by_model,
        question_totals_by_dataset, total_questions, version,
    )


if __name__ == "__main__":
    main()
