#!/usr/bin/env python3
import json
import sys
import threading
import time

import requests

from tools.special import special_char

TIMEOUT = 600  # seconds per question
END_OF_STREAM = special_char("end_stream")  # decoded once at import


def _part_text(part):
    if not isinstance(part, dict) or not isinstance(part.get("text"), str):
        sys.exit(f"ask.py: unexpected content part: {part!r}")
    return part["text"]


def _content_pieces(content, reasoning):
    """One delta or message as (thinking, text)."""
    if not content:  # "", None or []: nothing visible here
        return reasoning or "", ""

    if isinstance(content, str):
        # Llama.cpp style: content is a plain string
        return reasoning or "", content

    if isinstance(content, list):
        # Else: content is a list of typed parts
        thinking = text = ""
        for part in content:
            if not isinstance(part, dict):
                sys.exit(f"ask.py: unexpected content part: {part!r}")
            if part.get("type") == "thinking":
                thinking += "".join(_part_text(inner) for inner in part.get("thinking", []))
            elif part.get("type") == "text":
                text += _part_text(part)
            else:
                sys.exit(f"ask.py: unexpected content part type: {part!r}")
        return thinking, text

    # Handle unexpected types
    sys.exit(f"ask.py: unexpected content: {content!r}")


def _sse(url, headers, payload):
    """Streamed call: live tok/tail progress line. Returns (text, full_output, tokens)."""
    text = full_output = ""
    tokens = 0
    with requests.post(url, headers=headers,
                       json=payload, stream=True, timeout=TIMEOUT) as r:
        if not r.ok:
            sys.exit(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:].decode()
            if data.strip() == END_OF_STREAM:
                break
            chunk = json.loads(data)
            if chunk.get("usage"):  # llama.cpp: separate final chunk; else: on the finish chunk
                tokens = chunk["usage"].get("completion_tokens", tokens)
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta", {})
            thought, answer = _content_pieces(delta.get("content"), delta.get("reasoning_content"))
            if answer:
                text += answer
            for piece in thought, answer:
                if not piece:
                    continue
                tokens += 1  # each streamed piece is roughly one token
                full_output += piece
                tail = full_output[-60:].replace("\n", " ")
                print(f"\r    {tokens:>5} tok | {tail}", end="\x1b[K", flush=True)
    print()
    return text, full_output, tokens


def _plain(url, headers, payload):
    """Silent call: dots blink while it waits. Returns (text, full_output, tokens)."""
    stop = threading.Event()

    def blink():
        frame = 0
        while not stop.wait(0.5):
            print("\r" + "." * (frame % 3 + 1) + "   ", end="", flush=True)
            frame += 1

    dots = threading.Thread(target=blink, daemon=True)
    dots.start()
    try:
        r = requests.post(url, headers=headers,
                          json=payload, timeout=TIMEOUT)
    finally:
        stop.set()
        dots.join()
        print()  # close the blink line
    if not r.ok:
        sys.exit(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
    data = r.json()
    if not data.get("choices"):
        sys.exit(f"ask.py: response without choices: {data!r}")
    message = data["choices"][0].get("message", {})
    thought, text = _content_pieces(message.get("content"), message.get("reasoning_content"))
    tokens = (data.get("usage") or {}).get("completion_tokens")
    if tokens is None:
        sys.exit(f"ask.py: response without usage: {data!r}")
    return text, thought + text, tokens


def ask(model_id, base_url, headers, params, body, system, streaming=True):
    """Ask one question, return (text, full_output, tokens, dt)."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": body},
        ],
    }
    payload.update(params)  # exactly the declared knobs, nothing injected
    if streaming:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    print(payload)

    url = f"{base_url.rstrip('/')}/chat/completions"
    t0 = time.time()
    if streaming:
        text, full_output, tokens = _sse(url, headers, payload)
    else:
        text, full_output, tokens = _plain(url, headers, payload)
    return text, full_output, tokens, time.time() - t0
