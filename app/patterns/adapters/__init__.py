"""Concrete adapter implementations for each pipeline strategy."""

from app.patterns.adapters.transcription import (
    AssemblyAIAdapter,
    GeminiTranscriptionAdapter,
    OpenAIWhisperAdapter,
    WhisperXLocalAdapter,
)
from app.patterns.adapters.summarization import (
    GeminiSummarizationAdapter,
    LocalLLMAdapter,
    OpenAISummarizationAdapter,
)
from app.patterns.adapters.tts import FPTCloudTTSAdapter, LocalTTSAdapter

__all__ = [
    # Transcription
    "WhisperXLocalAdapter",
    "OpenAIWhisperAdapter",
    "AssemblyAIAdapter",
    "GeminiTranscriptionAdapter",
    # Summarization
    "LocalLLMAdapter",
    "OpenAISummarizationAdapter",
    "GeminiSummarizationAdapter",
    # TTS
    "FPTCloudTTSAdapter",
    "LocalTTSAdapter",
]
