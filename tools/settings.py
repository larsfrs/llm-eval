#!/usr/bin/env python3
"""The one place environment-driven settings come from.

Priority: the real environment first, then the .env file (KEY=VALUE lines,
gitignored — local overrides like API keys), then the defaults below.
Importing this module loads the .env file once; every knob is then a plain
module attribute.
"""

import os
import re
import sys
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


HEADER_REF = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def resolve_headers(headers):
    """Eval.json model headers, sent as-is: $ENV_NAME references in the
    values resolve from the environment (real env first, then the .env
    file — never literal secrets in the committed config). A reference
    that does not resolve exits loudly naming the header."""
    resolved = {}
    for name, value in headers.items():
        if not isinstance(value, str):
            sys.exit(f"settings.py: header {name!r} must be a string, got {value!r}")

        def substitute(match):
            env_name = match.group(1)
            if env_name not in os.environ:
                sys.exit(f"settings.py: header {name!r} references ${env_name}, which is not set")
            return os.environ[env_name]

        resolved[name] = HEADER_REF.sub(substitute, value)
    return resolved
