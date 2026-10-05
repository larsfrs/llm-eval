#!/usr/bin/env python3

import json
import os
import sys

from eval import default, rg
from tools import ask as ask_tool
from tools import csv as csv_tool
from tools import extract
from tools import log as log_tool
from tools import settings

# all environment knobs (incl. the .env file) live in tools/settings.py
CONFIG_PATH = settings.CONFIG_PATH
CSV_PATH = settings.CSV_PATH

# fallback sampling params; overridden by the model's "params", then the run
DEFAULT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
    "max_tokens": 1024,
}

# the sampling knobs sent with every question: the payload keys in ask.py,
# the per-question params header in the log, the row's param columns
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repeat_penalty", "max_tokens",
)


def get_sampling_params(model, run):
    """Sampling params for a combo: DEFAULT_SAMPLING < model "params" < run.

    The model entry carries its provider-recommended sampling settings; the
    run entry pins only what it deliberately varies (det pins the greedy
    triple, smp just carries repeats).
    """
    return {**DEFAULT_SAMPLING, **model.get("params", {}), **run}


def load_config():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    for key in ("version", "models", "runs", "suites"):
        if not config.get(key):
            sys.exit(f"{CONFIG_PATH}: missing or empty '{key}'")
    for suite in config["suites"]:
        for key in ("name", "data", "eval"):
            if not suite.get(key):
                sys.exit(
                    f"{CONFIG_PATH}: suite {suite.get('name', '?')!r} "
                    f"is missing '{key}'"
                )
    return config


# the evaluators used in the run
EVALUATORS = {
    "default": default,  # canonical format (static, linguistic_benchmark, ...)
    "rg": rg,            # reasoning gym (scoring delegated to the library)
}


def get_evaluator(name):
    """Resolve an evaluator name from eval.json to its module."""
    if name not in EVALUATORS:
        sys.exit(f"unknown evaluator {name!r} (have: {', '.join(sorted(EVALUATORS))})")

    module = EVALUATORS[name]
    missing = [k for k in ("load", "render", "verify") if not hasattr(module, k)]
    if missing:
        sys.exit(f"evaluator {name!r} is missing: {', '.join(missing)}")
    return module


def resolve_suites(config):
    """Resolve every suite once: (suite, evaluator, questions, columns).

    The question columns are the row positions ("{suite}_Q{n}"), so
    questions must stay append-only: existing indices never shift, new
    questions are added at the end. Anything else is a version bump.
    """
    suites = []
    for suite in config["suites"]:
        if not os.path.isfile(suite["data"]):
            hint = f" (run {suite['prepare']} first?)" if suite.get("prepare") else ""
            sys.exit(f"{suite['data']}: not found{hint}")
        evaluator = get_evaluator(suite["eval"])
        questions = evaluator.load(suite["data"])
        for position, question in enumerate(questions, 1):
            if question.get("index") != position:
                sys.exit(
                    f"{suite['data']}: question at position {position} has "
                    f"index {question.get('index')!r}; questions are "
                    f"append-only (add new ones at the end, never reorder)"
                )
        columns = [f"{suite['name']}_Q{i}" for i in range(1, len(questions) + 1)]
        suites.append((suite, evaluator, questions, columns))
    return suites


def format_duration(seconds):
    """Rough human-readable duration: "2h 15m", "37m 20s", or "45s"."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def remaining_time_note(session, max_tokens):
    """Worst-case remaining time note, appended to each progress line.

    Covers every missing cell of the whole run (all combos), based on the
    session's observed tokens/second.
    """
    questions_left = session.get("missing_left", 0)
    if questions_left <= 0 or session["seconds"] <= 0:
        return ""
    tokens_per_second = session["tokens"] / session["seconds"]
    max_seconds = questions_left * max_tokens / tokens_per_second
    return f", ~{format_duration(max_seconds)} left"


def run_combo(model, run, repeat, config, suites, fieldnames, existing_row, session):
    """Run the combo's missing questions, keeping its earlier answers.

    The row is seeded from the combo's existing line (when it has one):
    answered questions keep their cells, only empty cells go to the model.
    The row is written after every touched suite, so an interrupted run
    keeps its answers; the score returns once every cell is filled.
    """
    # resolve sampling params and the api key
    sampling_params = get_sampling_params(model, run)
    api_key_env = model.get("api_key_env", "API_KEY")
    api_key = os.environ.get(api_key_env, "sk-none")
    if api_key_env not in os.environ:
        print(f"  (warning: env var {api_key_env} not set, sending default key)")

    # seed the row from the existing line: earlier answers survive, the
    # score stays off until the combo is complete again
    csv_row = {k: v for k, v in (existing_row or {}).items() if k != "score"}
    csv_row.update({
        "model": model["name"],
        "version": config["version"],
        "run": run["name"],
        "repeat": repeat,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "top_k": sampling_params["top_k"],
        "min_p": sampling_params["min_p"],
        "presence_penalty": sampling_params["presence_penalty"],
        "repeat_penalty": sampling_params["repeat_penalty"],
        "max_tokens": sampling_params["max_tokens"],
    })

    # per suite, run only the questions whose cells are still empty
    for suite, evaluator, questions, columns in suites:
        missing = [
            (i, column, question)
            for i, (column, question) in enumerate(zip(columns, questions), 1)
            if not csv_row.get(column)
        ]
        if not missing:
            continue  # suite already fully answered

        system_prompt = getattr(evaluator, "SYSTEM", default.SYSTEM)

        # console header + start of the transcript for this suite
        suite_label = f"{model['name']} / {suite['name']} / {run['name']}"
        if sampling_params.get("repeats", 1) > 1:
            suite_label += f" #{repeat}"
        print(f"\n=== {suite_label} ===")
        if len(missing) < len(questions):
            print(f"  filling {len(missing)} of {len(questions)} questions")
        log_tool.start(model["name"], suite["name"], run["name"], repeat)

        # ask every missing question: render -> send -> extract -> verify
        for i, column, question in missing:
            question_prompt = evaluator.render(question)
            # what goes to the LLM, joined as one text: system prompt + question
            llm_input = f"{system_prompt}\n\n{question_prompt}"
            model_output, full_output, tokens, duration = ask_tool.ask(
                model["model_id"], model["base_url"], api_key, sampling_params,
                question_prompt, system=system_prompt,
            )
            session["tokens"] += tokens
            session["seconds"] += duration
            session["missing_left"] -= 1
            extracted_answer = extract.extract_answer(model_output)
            correct = evaluator.verify(question, extracted_answer)
            csv_row[column] = "pass" if correct else "fail"

            # progress line + transcript entry (full prompt in, full answer out)
            expected = question.get(
                "answer_index", question.get("correct_answer", question.get("answer"))
            )
            log_tool.log_question(llm_input, full_output, {
                key: sampling_params[key] for key in SAMPLING_KEYS
            })
            tokens_per_second = tokens / duration if duration else 0.0
            remaining_note = remaining_time_note(
                session, sampling_params["max_tokens"]
            )
            GREEN, RED, RESET = "\x1b[32m", "\x1b[31m", "\x1b[0m"
            status = f"{GREEN}OK  {RESET}" if correct else f"{RED}FAIL{RESET}"
            print(
                f"[{status}] Q{i}: expected {expected!r}, "
                f"got {extracted_answer!r} ({tokens} tok, "
                f"{tokens_per_second:.1f} tok/s, {duration:.1f}s"
                f"{remaining_note})"
            )
            if extracted_answer is None:
                print(f"      (no <answer> tag found, last output: ...{model_output[-200:]!r})")

        # the suite's overall state now, seeded answers included
        suite_passes = sum(1 for column in columns if csv_row.get(column) == "pass")
        print(f"  {suite['name']}: {suite_passes}/{len(questions)}")
        csv_tool.upsert_row(CSV_PATH, csv_row, fieldnames)

    # every cell is filled now: the score marks the combo complete again
    question_columns = [column for _, _, _, columns in suites for column in columns]
    total_correct = sum(
        1 for column in question_columns if csv_row.get(column) == "pass"
    )
    csv_row["score"] = f"{total_correct}/{len(question_columns)}"
    csv_tool.upsert_row(CSV_PATH, csv_row, fieldnames)
    return csv_row


def main():

    # fresh start if wanted: clear the results file before anything runs
    if settings.OVERWRITE_CSV and os.path.isfile(CSV_PATH):
        os.remove(CSV_PATH)
        print(f"{CSV_PATH}: cleared (OVERWRITE_CSV set)")

    # eval.json config file
    config = load_config()

    # resolve every suite once: evaluator, questions, and its column names
    suites = resolve_suites(config)
    question_columns = [
        column for _, _, _, columns in suites for column in columns
    ]
    fieldnames = csv_tool.FIXED_FIELDS + question_columns

    # new questions become empty cells in every existing row
    added = csv_tool.sync_header(CSV_PATH, fieldnames)
    if added:
        print(f"{CSV_PATH}: {len(added)} new column(s), empty cells in all rows")
    existing_rows = csv_tool.read_rows(CSV_PATH)

    # every (model, run, repeat) combination to consider
    combos = []
    for model in config["models"]:
        for run in config["runs"]:
            for repeat in range(1, run.get("repeats", 1) + 1):
                combos.append((model, run, repeat))

    # the real workload: combos with at least one empty question cell
    session = {"tokens": 0, "seconds": 0.0, "missing_left": 0}
    pending, skipped_combos = [], 0
    for model, run, repeat in combos:
        sampling_params = get_sampling_params(model, run)
        combo_key = (
            model["name"], config["version"], run["name"], str(repeat),
            str(sampling_params["temperature"]), str(sampling_params["top_p"]),
            str(sampling_params["top_k"]), str(sampling_params["min_p"]),
            str(sampling_params["presence_penalty"]),
            str(sampling_params["repeat_penalty"]),
            str(sampling_params["max_tokens"]),
        )
        existing = existing_rows.get(combo_key)
        missing = [
            column for column in question_columns
            if existing is None or not existing.get(column)
        ]
        if not missing:
            skipped_combos += 1
            continue
        session["missing_left"] += len(missing)
        pending.append((model, run, repeat, existing))

    for model, run, repeat, existing in pending:
        csv_row = run_combo(
            model, run, repeat, config, suites, fieldnames, existing, session,
        )
        print(f"Score: {csv_row['score']}  ->  updated {CSV_PATH}")

    if skipped_combos:
        print(f"\nSkipped {skipped_combos} of {len(combos)} combos (up to date in {CSV_PATH})")


if __name__ == "__main__":
    main()
