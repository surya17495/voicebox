# Gemini STT (this fork)

Dictation transcription runs on Google's `gemini-3.5-transcribe` via the
`/transcribe` route (multipart upload + /v1beta/interactions). Local Whisper
remains the fallback when no key is configured.

## Environment
- `GEMINI_API_KEY`        (required to enable; primary account)
- `GEMINI_API_KEY_2`      (optional second account; rotated on 429/err)
- `GEMINI_API_KEYS`       (optional comma list, merges with the two above)

## Behavior
- ONE batch call per utterance (whole recording by the app). See services/gemini_stt.py:
  bread-and-butter caveat — streamed chunks bill against the same 25 req/min
  free bucket, so wholesale client streaming would exhaust it in ~3s; keep it batch.
- Language pinned en-US (override via the route's `language` form field).
- Key rotation: on 429 the key caps for Google's own retry countdown; on 400
  invalid-argument the key caps 10min (account gate).
- `model` form field: `smart` (default; filler/self-correction cleanup) or `verbatim`.
