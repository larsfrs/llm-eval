#!/usr/bin/env python3
"""The reserved literals, kept out of every file in raw form.

These strings can never be written literally into the codebase: the agent
that maintains this repo streams its own output as SSE and dies on the
end-of-stream sentinel, and the harness filters reasoning tags out of its
I/O. So they live here base64-encoded and get decoded once at import. No
manual substitution step exists.

To change a value, recompute its base64 with:
    python3 -c "import base64; print(base64.b64encode(b'VALUE').decode())"
and edit _ENCODED below.
"""

import base64

_ENCODED = {
    # SSE end-of-stream sentinel (bracketed, four letters)
    "end_stream": "W0RPTkVd",
    # reasoning block tags the model emits (pair from THINK_RE in legacy quiz.py)
    "think_open": "PHRoaW5rPg==",
    "think_close": "PC90aGluazQ=",
}

_SPECIAL = {k: base64.b64decode(v).decode() for k, v in _ENCODED.items()}


def special_char(name):
    """Return a reserved literal, decoded from base64. KeyError if unknown."""
    return _SPECIAL[name]
