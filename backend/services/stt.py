"""Speech-to-text: local Whisper, or OpenAI Whisper API if OPENAI_API_KEY is set."""

import io
import os
import tempfile

from config import OPENAI_API_KEY


def convert(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    if not audio_bytes:
        return ""

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            ext = os.path.splitext(filename)[1] or ".webm"
            buf = io.BytesIO(audio_bytes)
            buf.name = f"recording{ext}"
            tr = client.audio.transcriptions.create(model="whisper-1", file=buf)
            return (tr.text or "").strip()
        except Exception:
            pass

    try:
        import whisper

        suffix = os.path.splitext(filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
            result = model.transcribe(tmp.name)
            return (result.get("text") or "").strip()
    except Exception:
        return "Unable to transcribe (install openai-whisper or set OPENAI_API_KEY)."
