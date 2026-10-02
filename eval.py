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
    "presence_penalty": 0.0,
    "max_tokens": 1024,
}


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


def format_duration(seconds):
    """Rough human-readable duration: "2h 15m", "37m 20s", or "45s"."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


def remaining_time_note(session, questions_left, max_tokens):
    """Worst-case remaining time note, appended to each progress line."""
    if questions_left <= 0 or session["seconds"] <= 0:
        return ""
    tokens_per_second = session["tokens"] / session["seconds"]
    max_seconds = questions_left * max_tokens / tokens_per_second
    return f", ~{format_duration(max_seconds)} left"


def run_combo(model, run, repeat, config, session, combos_after):
    """Run every suite of one combo, writing the row after each suite.

    session is the cross-combo running total {"tokens", "seconds"} used for
    the per-question remaining-time estimate; combos_after is how many
    combos still come after this one.
    """

    # resolve sampling params and the api key
    sampling_params = get_sampling_params(model, run)
    api_key_env = model.get("api_key_env", "API_KEY")
    api_key = os.environ.get(api_key_env, "sk-none")
    if api_key_env not in os.environ:
        print(f"  (warning: env var {api_key_env} not set, sending default key)")

    # resolve every suite upfront: evaluator, questions, and its column names
    # (the full header needs every suite's columns from the first write on)
    suites = []
    for suite in config["suites"]:
        if not os.path.isfile(suite["data"]):
            hint = f" (run {suite['prepare']} first?)" if suite.get("prepare") else ""
            sys.exit(f"{suite['data']}: not found{hint}")
        evaluator = get_evaluator(suite["eval"])
        questions = evaluator.load(suite["data"])
        columns = [f"{suite['name']}_Q{i}" for i in range(1, len(questions) + 1)]
        suites.append((suite, evaluator, questions, columns))
    fieldnames = csv_tool.FIXED_FIELDS + [
        column for _, _, _, columns in suites for column in columns
    ]

    # the CSV row: fixed columns first, one {suite}_Q{n} column per question
    csv_row = {
        "model": model["name"],
        "version": config["version"],
        "run": run["name"],
        "repeat": repeat,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "top_k": sampling_params["top_k"],
        "presence_penalty": sampling_params["presence_penalty"],
        "max_tokens": sampling_params["max_tokens"],
    }

    # run every suite; the row is written after each one, so an interrupted
    # combo keeps the results of the suites it already finished
    questions_per_combo = sum(len(questions) for _, _, questions, _ in suites)
    questions_done = 0
    total_correct, total_questions = 0, 0
    for suite, evaluator, questions, _ in suites:
        system_prompt = getattr(evaluator, "SYSTEM", default.SYSTEM)

        # console header + start of the transcript for this suite
        suite_label = f"{model['name']} / {suite['name']} / {run['name']}"
        if sampling_params.get("repeats", 1) > 1:
            suite_label += f" #{repeat}"
        print(f"\n=== {suite_label} ===")
        log_tool.start(model["name"], suite["name"], run["name"], repeat)

        # ask every question: render -> send -> extract -> verify
        correct_per_question = []
        for i, question in enumerate(questions, 1):
            question_prompt = evaluator.render(question)
            # what goes to the LLM, joined as one text: system prompt + question
            llm_input = f"{system_prompt}\n\n{question_prompt}"
            model_output, full_output, tokens, duration = ask_tool.ask(
                model["model_id"], model["base_url"], api_key, sampling_params,
                question_prompt, system=system_prompt,
            )
            session["tokens"] += tokens
            session["seconds"] += duration
            questions_done += 1
            extracted_answer = extract.extract_answer(model_output)
            correct = evaluator.verify(question, extracted_answer)
            correct_per_question.append(correct)

            # progress line + transcript entry (full prompt in, full answer out)
            expected = question.get(
                "answer_index", question.get("correct_answer", question.get("answer"))
            )
            log_tool.log_question(llm_input, full_output)
            tokens_per_second = tokens / duration if duration else 0.0
            questions_left = (
                questions_per_combo - questions_done
                + combos_after * questions_per_combo
            )
            remaining_note = remaining_time_note(
                session, questions_left, sampling_params["max_tokens"]
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

        # merge the suite's results into the row, then write it
        suite_correct = sum(correct_per_question)
        total_correct += suite_correct
        total_questions += len(questions)
        print(f"  {suite['name']}: {suite_correct}/{len(questions)}")
        for i, correct in enumerate(correct_per_question, 1):
            csv_row[f"{suite['name']}_Q{i}"] = "pass" if correct else "fail"
        csv_tool.upsert_row(CSV_PATH, csv_row, fieldnames)

    # the score is written last, marking the combo complete
    csv_row["score"] = f"{total_correct}/{total_questions}"
    csv_tool.upsert_row(CSV_PATH, csv_row, fieldnames)
    return csv_row


def main():

    # fresh start if wanted: clear the results file before anything runs
    if settings.OVERWRITE_CSV and os.path.isfile(CSV_PATH):
        os.remove(CSV_PATH)
        print(f"{CSV_PATH}: cleared (OVERWRITE_CSV set)")

    # eval.json config file
    config = load_config()

    # completed rows, so interrupted runs can resume
    completed_keys = csv_tool.read_completed(CSV_PATH)

    # every (model, run, repeat) combination to run
    combos = []
    for model in config["models"]:
        for run in config["runs"]:
            for repeat in range(1, run.get("repeats", 1) + 1):
                combos.append((model, run, repeat))

    # run every combo not already in the results file
    total_combos, skipped_combos, done_combos = len(combos), 0, 0
    session = {"tokens": 0, "seconds": 0.0}  # running total for the estimate
    for model, run, repeat in combos:
        sampling_params = get_sampling_params(model, run)
        combo_key = (
            model["name"], config["version"], run["name"], repeat,
            str(sampling_params["temperature"]), str(sampling_params["top_p"]),
            str(sampling_params["top_k"]), str(sampling_params["presence_penalty"]),
            str(sampling_params["max_tokens"]),
        )
        if combo_key in completed_keys:
            skipped_combos += 1
            continue

        # run all suites of the combo; the row is written after each suite
        combos_after = total_combos - skipped_combos - done_combos - 1
        csv_row = run_combo(model, run, repeat, config, session, combos_after)
        done_combos += 1
        print(f"Score: {csv_row['score']}  ->  updated {CSV_PATH}")

    if skipped_combos:
        print(f"\nSkipped {skipped_combos} of {total_combos} combos (already in {CSV_PATH})")


if __name__ == "__main__":
    main()
