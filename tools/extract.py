#!/usr/bin/env python3
"""Answer extraction: strip the thinking block, pull the <answer> payload."""

import re

from tools.special import special_char

THINK_RE = re.compile(
    re.escape(special_char("think_open"))
    + r".*?"
    + re.escape(special_char("think_close")),
    re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def extract_answer(text):
    """Pull the committed <answer> payload out of the final answer.

    The LAST non-empty answer tag outside the thinking block wins: models
    often quote the format or draft answers inside their working text, and
    the committed answer is the last one they write. A model that never
    writes a final answer within its budget returns None -> auto-fail, and
    that failure is part of what the benchmark measures.
    """
    matches = [answer for answer in ANSWER_RE.findall(THINK_RE.sub("", text)) if answer]
    return matches[-1] if matches else None
