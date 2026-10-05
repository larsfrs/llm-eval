#!/usr/bin/env python3
"""Overview graphs from results.csv.

One group of slim stacked bars per model, ranked best-first by total
questions passed in the deterministic run: within a group the det bar
stands on the left and every sampled run to its right, each bar marked
d / s just below the axis. Colored segments = the datasets (one per
suite, e.g. static, linguistic_benchmark, rg), the bare pass count
printed inside each segment and each section's question total in the
legend. A dotted reference line marks the total number of configured
questions; inside each group a dotted black line crosses the bars at
the model's average total passed across all of its runs.
A small note box in the top-right corner explains the groups in plain
words (the title carries no version).
Writes plots/results_overview.svg (the OUT env var overrides).
Run: uv run analysis.py
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless renderer: draw into a buffer, save to png
import matplotlib.pyplot as plt

from tools import settings

OUT_PATH = os.environ.get("OUT", os.path.join("plots", "results_overview.svg"))


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

    Returns (det_counts_by_model, sampled_counts_by_model):
    - det: model -> dataset -> passed count (the bar segments)
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

        # det rows become the d bars, sampled runs the s bars;
        # rows of other (or pre-versioning) question sets do not count
        det_counts_by_model = {}
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

            if fields[run_col] == "det":
                dataset_counts = det_counts_by_model.setdefault(fields[model_col], {})
                for label in passed_datasets:
                    dataset_counts[label] = dataset_counts.get(label, 0) + 1
            else:
                run_counts = {}
                for label in passed_datasets:
                    run_counts[label] = run_counts.get(label, 0) + 1
                sampled_counts_by_model.setdefault(fields[model_col], []).append(
                    run_counts
                )
    return det_counts_by_model, sampled_counts_by_model


def plot_overview(det_counts_by_model, sampled_counts_by_model,
                 question_totals_by_dataset, total_questions, version):
    """Grouped stacked bars: the det run and every sampled run per model."""
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

    # one group of slim bars per model: the det bar on the left, every
    # sampled run to its right, each marked d / s just below the axis
    figure, axis = plt.subplots(figsize=(12, 7))
    bar_width, bar_gap, group_spacing = 0.22, 0.05, 1.5
    below_axis = axis.get_xaxis_transform()  # x in data, y in axes units
    group_centers = []
    for group_number, model in enumerate(models):
        runs = [("d", det_counts_by_model[model])] + [
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
        # total passed over all of its runs (the d and every s one);
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
        group_centers.append(group_center)

    # dotted reference line: every question of every configured suite
    reference_line = axis.axhline(
        total_questions, linestyle=":", color="gray",
        label=f"{total_questions} questions total",
    )

    # axes, legend, and the png
    axis.set_xlim(group_centers[0] - 0.8, group_centers[-1] + 0.8)
    axis.set_xticks(group_centers)
    axis.set_xticklabels(models, rotation=90)
    axis.tick_params(axis="x", pad=18)  # model names sit below the d/s marks
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
        title="dataset", loc="upper right", bbox_to_anchor=(1.0, 0.80),
    )

    # plain-words note box pinned to the top-right corner: the legend is
    # anchored just below it — loc="best" cannot see the note and would
    # collide with it
    axis.text(
        0.98, 0.97,
        "Each group: the d bar is the deterministic run,\n"
        "the s bars are its sampled runs. Counts inside the\n"
        "bars are passes per section; each section's total\n"
        "is in the legend. The sections are described in\n"
        "the sections below.",
        transform=axis.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"},
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    figure.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH} (version {version})")

    # per-model totals in the console: det plus every sampled run's total
    for model in models:
        det_total = sum(det_counts_by_model[model].values())
        sampled_runs = sampled_counts_by_model.get(model, [])
        if sampled_runs:
            run_totals = ", ".join(
                str(sum(run.values())) for run in sampled_runs
            )
            print(f"  {model}: {det_total} det; sampled runs: {run_totals}")
        else:
            print(f"  {model}: {det_total} (deterministic only)")


def main():
    labels_by_suite = load_dataset_labels()
    question_totals_by_dataset = count_questions_per_dataset(labels_by_suite)
    total_questions = sum(question_totals_by_dataset.values())
    version = load_current_version()
    det_counts_by_model, sampled_counts_by_model = collect_results_per_model(
        labels_by_suite, version
    )
    plot_overview(
        det_counts_by_model, sampled_counts_by_model,
        question_totals_by_dataset, total_questions, version,
    )


if __name__ == "__main__":
    main()
