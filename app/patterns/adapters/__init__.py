"""Concrete adapter implementations for each pipeline strategy."""

from app.patterns.adapters.transcription import WhisperXLocalAdapter
from app.patterns.adapters.summarization import LocalLLMAdapter
from app.patterns.adapters.tts import FPTCloudTTSAdapter, LocalTTSAdapter

__all__ = [
    "WhisperXLocalAdapter",
    "LocalLLMAdapter",
    "FPTCloudTTSAdapter",
    "LocalTTSAdapter",
]
