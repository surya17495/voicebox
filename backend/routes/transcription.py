"""Transcription endpoints."""

import asyncio
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import models
from ..services import transcribe
from ..services.task_queue import create_background_task
from ..utils.tasks import get_task_manager

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB

# Same set profiles.py accepts for voice samples. librosa picks its decoder from the
# file extension, so the temp file has to keep the uploaded one.
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}


@router.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
):
    """Transcribe audio file to text.

    Fork patch (gemini-stt branch, 2026-08-30): when GEMINI_API_KEY is set,
    transcribe via Google's gemini-3.5-transcribe (multipart upload +
    interactions; key-chain + quota pacing in services.gemini_stt). Falls back
    to local Whisper when no key is configured, preserving upstream behavior.
    """
    uploaded_ext = Path(file.filename or "").suffix.lower()
    file_suffix = uploaded_ext if uploaded_ext in ALLOWED_AUDIO_EXTS else ".wav"

    with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            tmp.write(chunk)
        tmp_path = tmp.name

    from .. import config as _cfg  # noqa: PLC0415 - runtime env lookup
    if getattr(_cfg, "GEMINI_API_KEY", None) or __import__("os").environ.get("GEMINI_API_KEY"):
        from ..services.gemini_stt import gemini_transcribe_file  # noqa: PLC0415

        smart_mode = (model or "smart").lower() != "verbatim"
        try:
            # wav keeps its real duration; containers get best-effort 0.0
            duration = 0.0
            if file_suffix == ".wav":
                import wave as _wave  # noqa: PLC0415

                try:
                    with _wave.open(tmp_path, "rb") as w:
                        duration = w.getnframes() / float(w.getframerate() or 1)
                except Exception:
                    duration = 0.0
            text, tag = await asyncio.to_thread(
                gemini_transcribe_file, tmp_path, language, smart_mode, False)
            return models.TranscriptionResponse(text=text, duration=duration)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini STT: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    stt_path = tmp_path
    try:
        from ..utils.audio import load_audio, save_audio
        from ..backends import WHISPER_HF_REPOS

        audio, sr = await asyncio.to_thread(load_audio, tmp_path)
        duration = len(audio) / sr

        # The STT backend (mlx_audio.stt -> miniaudio) only decodes
        # WAV/FLAC/MP3/Vorbis, so browser recordings uploaded as WebM/Opus
        # fail with "unsupported file format" (issue: web-mode dictation).
        # librosa already decoded the file above (it falls back to
        # audioread/ffmpeg for exotic containers), so re-encode that PCM to a
        # temp WAV and hand *that* to Whisper. WAV inputs pass through
        # unchanged.
        if file_suffix != ".wav":
            stt_path = f"{tmp_path}.stt.wav"
            await asyncio.to_thread(save_audio, audio, stt_path, sr)

        whisper_model = transcribe.get_whisper_model()
        model_size = model if model else whisper_model.model_size

        valid_sizes = list(WHISPER_HF_REPOS.keys())
        if model_size not in valid_sizes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model size '{model_size}'. Must be one of: {', '.join(valid_sizes)}",
            )

        already_loaded = whisper_model.is_loaded() and whisper_model.model_size == model_size
        if not already_loaded and not whisper_model._is_model_cached(model_size):
            progress_model_name = f"whisper-{model_size}"
            task_manager = get_task_manager()

            async def download_whisper_background():
                try:
                    await whisper_model.load_model_async(model_size)
                    task_manager.complete_download(progress_model_name)
                except Exception as e:
                    task_manager.error_download(progress_model_name, str(e))

            task_manager.start_download(progress_model_name)
            create_background_task(download_whisper_background())

            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"Whisper model {model_size} is being downloaded. Please wait and try again.",
                    "model_name": progress_model_name,
                    "downloading": True,
                },
            )

        text = await whisper_model.transcribe(stt_path, language, model_size)

        return models.TranscriptionResponse(
            text=text,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if stt_path != tmp_path:
            Path(stt_path).unlink(missing_ok=True)
