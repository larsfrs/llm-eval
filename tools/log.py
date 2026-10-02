#!/usr/bin/env python3
"""One log per runner invocation: everything sent and answered, in order.

Per combo a divider with its identity, then two blocks per question: the
full prompt string that goes to the LLM (system prompt + question, joined
as one text) and the complete output the LLM returned (thinking included).
"""

import os

from tools import settings

LOG_PATH = os.path.join(settings.LOG_DIR, settings.LOG_NAME)


def start(model_name, suite_name, run_name, repeat):
    """Append a combo's identity divider to the session log."""
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n{'=' * 78}\n")
        f.write(f"{model_name} / {suite_name} / {run_name} / #{repeat}\n")
        f.write(f"{'=' * 78}\n")


def log_question(llm_input, llm_output):
    """Append the two blocks: the full prompt sent, the full answer returned."""
    with open(LOG_PATH, "a") as f:
        f.write("\n--- input ---\n")
        f.write(llm_input.rstrip() + "\n")
        f.write("--- output ---\n")
        f.write((llm_output or "").rstrip() + "\n")
