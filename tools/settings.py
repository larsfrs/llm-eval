#!/usr/bin/env python3
"""The one place environment-driven settings come from.

Priority: the real environment first, then the .env file (KEY=VALUE lines,
gitignored — local overrides like API keys), then the defaults below.
Importing this module loads the .env file once; every knob is then a plain
module attribute.
"""

import os
from datetime import datetime

ENV_PATH = os.environ.get("ENV_FILE", ".env")


def _load_env_file(path):
    """Load KEY=VALUE lines into os.environ; the real environment wins."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(ENV_PATH)


CONFIG_PATH = os.environ.get("CONFIG", "eval.json")  # runner config file
CSV_PATH = os.environ.get("CSV_PATH", "results.csv")  # results CSV
LOG_DIR = os.environ.get("LOG_DIR", "logs")  # transcript directory
LOG_NAME = f"run_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"  # session transcript
OVERWRITE_CSV = os.environ.get("OVERWRITE_CSV", "").lower() in ("1", "true", "yes")
