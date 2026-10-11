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

# the canonical sampling knobs: the row's param columns and the per-question
# params block in the log — the payload itself carries exactly what
# get_sampling_params returns, nothing injected anywhere
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repeat_penalty", "max_tokens",
)

# keys ask.py owns in the payload; a model's params may never collide
RESERVED_PARAM_KEYS = {"model", "messages", "stream", "stream_options", "headers"}


def get_sampling_params(model, run):
    """Sampling params for a combo, exactly as they go on the wire.

    The model's "params" declare every knob its provider takes — only
    declared knobs are ever sent. The run overrides values of declared
    knobs; it can never inject an undeclared one onto the wire. No
    defaults: a knob nobody declares is simply not sent and the server's
    own default applies. max_tokens must be declared — the progress time
    estimate needs a budget.
    """
    params = dict(model.get("params", {}))
    for key, value in run.items():
        if key in params:
            params[key] = value
    if clashing := RESERVED_PARAM_KEYS & params.keys():
        sys.exit(
            f"model {model.get('name', '?')!r}: params may not contain "
            f"{sorted(clashing)}"
        )
    if "max_tokens" not in params:
        sys.exit(
            f"model {model.get('name', '?')!r}: params must declare "
            f"max_tokens (no defaults — without a budget the time "
            f"estimate has nothing to work with)"
        )
    return params


def load_config():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    for key in ("version", "models", "runs", "suites"):
        if not config.get(key):
            sys.exit(f"{CONFIG_PATH}: missing or empty '{key}'")
    if not isinstance(config.get("streaming", True), bool):
        sys.exit(f"{CONFIG_PATH}: 'streaming' must be true or false")
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
    # resolve sampling params and the headers (auth rides in them, as configured)
    sampling_params = get_sampling_params(model, run)
    headers = settings.resolve_headers(model.get("headers", {}))

    # seed the row from the existing line: earlier answers survive, the
    # score stays off until the combo is complete again
    csv_row = {k: v for k, v in (existing_row or {}).items() if k != "score"}
    csv_row.update({
        "model": model["name"],
        "version": config["version"],
        "run": run["name"],
        "repeat": repeat,
        # empty cell = knob not declared, so not sent (no defaults anywhere)
        **{key: sampling_params.get(key, "") for key in SAMPLING_KEYS},
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
                model["model_id"], model["base_url"], headers, sampling_params,
                question_prompt, system=system_prompt,
                streaming=config.get("streaming", True),
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
                key: sampling_params.get(key, "") for key in SAMPLING_KEYS
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

    # every (model, run, repeat) combination to consider; a model with
    # "use_ukn_only" records every run as "ukn" — unknown sampling, its
    # provider refuses the knobs — so its regimes collapse onto the same
    # row key and it measures once per repeat
    combos = []
    for model in config["models"]:
        for run in config["runs"]:
            name = "ukn" if model.get("use_ukn_only") else run["name"]
            for repeat in range(1, run.get("repeats", 1) + 1):
                combos.append((model, {**run, "name": name}, repeat))

    # the real workload: combos with at least one empty question cell;
    # a row is measured once — with use_ukn_only several regimes map onto
    # the same key, the first combo claims it, the rest skip
    session = {"tokens": 0, "seconds": 0.0, "missing_left": 0}
    pending, skipped_combos = [], 0
    claimed_keys = set()
    for model, run, repeat in combos:
        sampling_params = get_sampling_params(model, run)
        # the resume key, identical to the row's cells (csv KEY_FIELDS):
        # an undeclared knob is an empty cell on both sides
        combo_key = (
            model["name"], config["version"], run["name"], str(repeat),
            *(str(sampling_params.get(key, "")) for key in SAMPLING_KEYS),
        )
        if combo_key in claimed_keys:
            skipped_combos += 1
            continue
        existing = existing_rows.get(combo_key)
        missing = [
            column for column in question_columns
            if existing is None or not existing.get(column)
        ]
        if not missing:
            skipped_combos += 1
            continue
        claimed_keys.add(combo_key)
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
