"""Long-dictation restructuring via Nebius (owner's token factory).

Gemini smart mode already delivers punctuation + word-level cleanup at STT
time. What it cannot do is LAYOUT: turning 45-second walls of text into
paragraphs and lists. This pass runs ONLY for recordings at or above
NEBIUS_REWRITE_MIN_SEC (default 12s), using a paid, high-throughput model -
so the free-tier bucket is untouched and short dictations gain zero latency.

Fidelity rule is absolute: never change, add, or drop words/names/numbers.
Only layout (paragraph breaks, bullets for explicit enumerations).
"""

import json
import os

import requests

_BASE = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/") + "/"
_MODEL = os.environ.get("NEBIUS_REWRITE_MODEL", "nvidia/Nemotron-3_5-Lightning")
_MIN_SEC = float(os.environ.get("NEBIUS_REWRITE_MIN_SEC", "12"))

_SYSTEM = ("You reformat dictated text. NEVER change, add, or drop words, names, numbers, or meaning. "
           "Only: split into short paragraphs, use bullet points only when the speaker is clearly enumerating, "
           "keep every technical term exact. Return only the reformatted text.")


def configured() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY"))


def min_sec() -> float:
    return _MIN_SEC


def model_name() -> str:
    return _MODEL


def rewrite(text: str) -> str:
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key or not text.strip():
        return text
    r = requests.post(
        f"{_BASE}chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": _MODEL,
              "messages": [{"role": "user", "content": f"{_SYSTEM}\n\nDictated text:\n{text}"}],
              "temperature": 0.2, "max_tokens": 1500,
              # measured: default thinking mode = 1200 reasoning tokens, 4.7-90s+;
              # reasoning off = 0.86-0.97s with identical structure quality
              "chat_template_kwargs": {"enable_thinking": False}},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"nebius {r.status_code}: {r.text[:100]}")
    out = (r.json()["choices"][0]["message"].get("content") or "").strip()
    return out or text
