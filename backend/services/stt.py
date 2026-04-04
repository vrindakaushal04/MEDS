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
        import os as _os

        suffix = os.path.splitext(filename)[1] or ".webm"
        tmp_path = None
        try:
            # Windows fix: delete=False so whisper can open the file after we close it
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            model = whisper.load_model(_os.getenv("WHISPER_MODEL", "base"))
            result = model.transcribe(tmp_path)
            return (result.get("text") or "").strip()
        finally:
            if tmp_path and _os.path.exists(tmp_path):
                try:
                    _os.unlink(tmp_path)
                except Exception:
                    pass
    except Exception:
        return "Unable to transcribe (install openai-whisper or set OPENAI_API_KEY)."
