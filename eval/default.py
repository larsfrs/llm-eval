#!/usr/bin/env python3
"""Default evaluator for every suite in the canonical question format.

Covers static and linguistic_benchmark (and any future JSON benchmark
adapted to the format) — no per-suite eval file needed. run.py falls back
to this module whenever eval/<suite_name>.py does not exist.

Scoring follows the hard project rule: we only ever compare indices or
numbers, never long text. The model answers MC questions with the option
number (compared to answer_index) and numeric questions with a bare number
(compared to correct_answer within 1%).
"""

import json
import sys

SYSTEM = """You are answering quiz questions. Think step by step first if you need to.
Once you are done thinking, give your final answer inside <answer></answer> tags,
using the tags precisely once — only around your final answer, nowhere else.
The tags must contain ONLY the answer and nothing else:
- Multiple choice: the number of the correct option, e.g. <answer>3</answer>
- Numerical question: the bare number, no units or words, e.g. <answer>9.81</answer>"""


def load(path):
    """Load a question file in the canonical format."""
    with open(path) as f:
        questions = json.load(f)
    if not isinstance(questions, list) or not questions:
        sys.exit(f"{path}: expected a non-empty list of questions")
    return questions


def render(q):
    """User message body: question text, plus numbered choices for MC."""
    body = q["question"]
    if q["multiple_choice"] is not None:
        body += "\n" + "\n".join(
            f"{i}. {c}" for i, c in enumerate(q["multiple_choice"], 1)
        )
    return body


def verify(q, raw):
    """True if the extracted answer matches: option index, or number."""
    if raw is None:
        return False
    try:
        if q["multiple_choice"] is not None:
            return int(raw) == q["answer_index"]
        target = float(q["correct_answer"])
        return abs(float(raw) - target) <= max(0.01, abs(target) * 0.01)
    except (TypeError, ValueError):
        return False
