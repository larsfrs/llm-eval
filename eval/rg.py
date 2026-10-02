#!/usr/bin/env python3
"""Reasoning gym evaluator: scoring delegated to the library's own scorers.

Prepared rg questions carry the library's item fields verbatim (question /
answer / metadata, plus envelope fields). verify() hands the model's
extracted answer to the matching dataset's score_answer — full credit
(>= 1.0) counts as a pass, partial credit does not. No scoring logic of our
own lives here.
"""

import json
import sys
from functools import cache

from reasoning_gym import create_dataset

SYSTEM = """You are answering puzzle questions. Think step by step first if you need to.
Follow the answer format requested by each question exactly.
Once you are done thinking, give your final answer inside <answer></answer> tags,
using the tags precisely once — only around your final answer, nowhere else.
The tags must contain ONLY the final answer and nothing else."""


def load(path):
    with open(path) as f:
        questions = json.load(f)
    if not isinstance(questions, list) or not questions:
        sys.exit(f"{path}: expected a non-empty list of questions")
    return questions


def render(q):
    return q["question"]


@cache
def scorer(name):
    """Dataset instance for calling the library's score_answer (any config)."""
    return create_dataset(name)


def verify(q, raw):
    if raw is None:
        return False
    entry = {
        "question": q["question"],
        "answer": q["answer"],
        "metadata": q["metadata"],
    }
    try:
        return scorer(q["metadata"]["source_dataset"]).score_answer(raw, entry) >= 1.0
    except Exception:
        return False  # same spirit as the library: unparseable answer scores 0
