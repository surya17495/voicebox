"""Gemini STT provider (custom fork patch, 2026-08-30).

Replaces local Whisper with Google's gemini-3.5-transcribe using the
multipart-upload + /v1beta/interactions flow (NOT generateContent). Supports a
key chain (GEMINI_API_KEY + GEMINI_API_KEY_2 + GEMINI_API_KEYS comma list) with
per-key 429 cooldowns parsed from Google's own retry-after countdown, language
pinning (default en-US), and smart/verbatim modes.

Streamed chunks bill against the same free-tier bucket as batch calls, so this
service deliberately uses ONE batch call per utterance (no client-side
streaming) - the Voicebox frontend posts the whole recording here.
"""

import json
import time as _time
from pathlib import Path

import requests

_GB = "https://generativelanguage.googleapis.com"

# key gated 400 = account-level rejection (model not enabled for that account).
# cap it for 10 minutes instead of retrying every call.
_KEY_INVALID_CAP_S = 600

_state = {"keys": [], "cur": 0, "cap": {}}


def _cfg_keys():
    import os
    keys = []
    for k in [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        *(x.strip() for x in os.environ.get("GEMINI_API_KEYS", "").split(",") if x.strip()),
    ]:
        if k and k not in keys:
            keys.append(k)
    return keys


def enabled() -> bool:
    """True when at least one Gemini key is configured (STT/refine routed here)."""
    return bool(_cfg_keys())


def gemini_refine_text(system_prompt: str, user_text: str) -> str:
    """Refinement via gemini-3.5-flash (separate free-tier bucket from STT)."""
    keys = _cfg_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY not configured")
    last = ""
    for i in range(len(keys)):
        key = keys[(i + _state["cur"]) % len(keys)]
        try:
            r = requests.post(
                f"{_GB}/v1beta/models/gemini-3.5-flash:generateContent",
                headers={"x-goog-api-key": key},
                json={"contents": [
                    {"role": "user", "parts": [
                        {"text": f"{system_prompt}\n\nTranscript:\n{user_text}"},
                    ]},
                ]},
                timeout=40,
            )
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if r.status_code == 429:
                import re
                m = re.search(r"retry in ([0-9.]+)s", r.text)
                _state["cap"][i] = _time.time() + (float(m.group(1)) if m else 45.0)
                last = f"key#{i+1} 429"
            else:
                _state["cap"][i] = _time.time() + _KEY_INVALID_CAP_S
                last = f"key#{i+1} {r.status_code}: {r.text[:90]}"
        except requests.RequestException as e:
            last = f"key#{i+1} {str(e)[:90]}"
    raise RuntimeError(f"refine failed: {last}")


def _pick_key():
    now = _time.time()
    for i in range(len(_state["keys"])):
        if now >= _state["cap"].get(i, 0):
            _state["cur"] = i
            return i, 0.0
    waits = [max(0.0, _state["cap"].get(i, 0) - now) for i in range(len(_state["keys"]))]
    j = min(range(len(waits)), key=lambda i: waits[i])
    _state["cur"] = j
    return j, waits[j]


def gemini_transcribe_file(path: str, language: str | None = None,
                           smart: bool = True, diarize: bool = False) -> tuple[str, str]:
    """Returns (text, provider_tag). Raises RuntimeError with a readable message."""
    import re
    keys = _cfg_keys()
    _state["keys"] = keys
    if not keys:
        raise RuntimeError("GEMINI_API_KEY not configured")
    data = Path(path).read_bytes()
    last_err = ""
    attempts = 0
    retry_same = False
    max_attempts = max(3, len(keys) * 2)
    while attempts < max_attempts:
        idx, wait = _pick_key()
        if wait > 0:
            raise RuntimeError(f"all {len(keys)} keys on quota - soonest unlock in {int(wait)}s")
        key = keys[idx]
        headers = {"x-goog-api-key": key}
        try:
            up = requests.post(
                f"{_GB}/upload/v1beta/files", params={"key": key, "uploadType": "multipart"},
                headers=headers,
                files={"file": ("audio.wav", data, "audio/wav")},
                data={"request": json.dumps({"file": {"display_name": "voicebox"}})}, timeout=30)
            if up.status_code != 200 or "file" not in up.json():
                last_err = f"upload {up.status_code}: {up.text[:100]}"
                _state["cap"][idx] = _time.time() + 30
            else:
                fj = up.json()["file"]
                if smart and not diarize:
                    tconf = {"mode": "smart"}
                elif diarize:
                    tconf = {"mode": {"type": "verbatim", "diarization_mode": "speaker"}}
                else:
                    tconf = {"mode": "verbatim"}
                tconf["language_codes"] = [(language or "en-US").replace("_", "-")]
                r = requests.post(f"{_GB}/v1beta/interactions", headers=headers, timeout=60,
                    json={"model": "gemini-3.5-transcribe",
                          "input": [{"type": "audio", "uri": fj["uri"], "mime_type": fj.get("mimeType", "audio/wav")}],
                          "generation_config": {"transcription_config": tconf}})
                if r.status_code == 200:
                    texts = []
                    for st in r.json().get("steps", []):
                        if st.get("type") == "model_output":
                            for p in st.get("content", []):
                                if p.get("type") == "text" and p.get("text"):
                                    texts.append(p["text"])
                    text = " ".join(texts).strip()
                    if text:
                        return text, f"gemini(key#{idx+1})"
                    # 200-with-empty-output is transient (seen on garbage/silent
                    # audio); retry same key - a hard error here wedges the app's
                    # pill state machine, so never 500 faster than we must.
                    last_err = "200 with empty output"
                    retry_same = True
                    _time.sleep(1.5)
                    attempts += 1
                    continue
                elif r.status_code == 429:
                    m = re.search(r"retry in ([0-9.]+)s", r.text)
                    cd = float(m.group(1)) if m else 45.0
                    _state["cap"][idx] = _time.time() + max(cd, 3.0)
                    last_err = f"key#{idx+1} quota window ({cd:.0f}s)"
                else:
                    _state["cap"][idx] = _time.time() + _KEY_INVALID_CAP_S
                    last_err = f"key#{idx+1} {r.status_code}: {r.text[:100]}"
        except requests.RequestException as e:
            last_err = f"key#{idx+1} network: {str(e)[:90]}"
            _state["cap"][idx] = _time.time() + 20
        attempts += 1
        if retry_same:
            retry_same = False          # same key: empty-200 is transient, not key health
        else:
            _state["cur"] = (idx + 1) % max(1, len(keys))
    raise RuntimeError(last_err or "all Gemini keys exhausted")
