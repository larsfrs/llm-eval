#!/usr/bin/env python3
"""Streaming call to an OpenAI-compatible chat endpoint. Transport only.

Returns (text, full_output, tokens, dt): the visible content channel (what
answer extraction runs on) and the complete stream including reasoning,
plus token count and wall time. Answer extraction lives in tools/extract.py;
scoring lives in the per-suite evaluators (eval/default.py for the canonical
format).
"""

import json
import sys
import time

import requests

from tools.special import special_char

TIMEOUT = 600  # seconds per question
END_OF_STREAM = special_char("end_stream")  # decoded once at import


def ask(model_id, base_url, api_key, params, body, system):
    """Ask one question, stream the reply, return (text, full_output, tokens, dt)."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "presence_penalty": params["presence_penalty"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.time()
    text = full_output = ""
    tokens = 0
    with requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        stream=True,
        timeout=TIMEOUT,
    ) as r:
        if not r.ok:
            sys.exit(f"HTTP {r.status_code} from {base_url}: {r.text[:500]}")
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:].decode()
            if data.strip() == END_OF_STREAM:
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                tokens = chunk.get("usage", {}).get("completion_tokens", tokens)
                continue
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                text += delta["content"]
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                tokens += 1  # each SSE chunk carries roughly one token
                full_output += piece
                tail = full_output[-60:].replace("\n", " ")
                print(f"\r    {tokens:>5} tok | {tail}", end="\x1b[K", flush=True)
    print()
    dt = time.time() - t0
    return text, full_output, tokens, dt
