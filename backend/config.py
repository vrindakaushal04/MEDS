import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

USE_LLM = os.getenv("USE_LLM", "true").lower() in ("1", "true", "yes")

# Oumi fine-tuned model served via any OpenAI-compatible API (vLLM, LM Studio, mlx, etc.)
OUMI_BASE_URL = os.getenv("OUMI_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
OUMI_MODEL = os.getenv("OUMI_MODEL", "default")
OUMI_API_KEY = os.getenv("OUMI_API_KEY", "dummy")

# Optional: cloud Whisper when local STT is not installed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
