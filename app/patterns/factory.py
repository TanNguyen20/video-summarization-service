"""Factory for creating pipeline components.

Maps the ``env`` string (from the API request) to the correct concrete
adapter.  Adding a new provider requires only a new ``elif`` branch.
"""

from app.core.config import settings
from app.core.logging import get_logger
from app.patterns.adapters import (
    AssemblyAIAdapter,
    ElevenLabsTTSAdapter,
    FPTCloudTTSAdapter,
    GeminiSummarizationAdapter,
    GeminiTranscriptionAdapter,
    LocalLLMAdapter,
    LocalTTSAdapter,
    OpenAISummarizationAdapter,
    OpenAITTSAdapter,
    OpenAIWhisperAdapter,
    WhisperXLocalAdapter,
)
from app.patterns.interfaces import (
    SummarizationStrategy,
    TTSStrategy,
    TranscriptionStrategy,
)

logger = get_logger("factory")


class ComponentFactory:
    """Factory for creating pipeline components.

    All configuration flows from ``settings`` so that .env values
    are respected automatically.

    Supported ``env`` values:
        Transcription : "local" (WhisperX) | "openai" | "assemblyai" | "gemini"
        Summarization : "local" (Ollama)   | "openai" | "gemini"
        TTS           : "local" (gTTS)     | "fpt"    | "openai" | "elevenlabs"
    """

    @staticmethod
    def create_transcriber(env: str = "local") -> TranscriptionStrategy:
        logger.info("Creating transcriber: env=%s", env)
        if env == "local":
            return WhisperXLocalAdapter()
        if env == "openai":
            return OpenAIWhisperAdapter()
        if env == "assemblyai":
            return AssemblyAIAdapter()
        if env == "gemini":
            return GeminiTranscriptionAdapter()
        raise ValueError(
            f"Unsupported transcriber env: '{env}'. "
            f"Choose from: local, openai, assemblyai, gemini"
        )

    @staticmethod
    def create_summarizer(env: str = "local") -> SummarizationStrategy:
        logger.info("Creating summarizer: env=%s", env)
        if env == "local":
            return LocalLLMAdapter()
        if env == "openai":
            return OpenAISummarizationAdapter()
        if env == "gemini":
            return GeminiSummarizationAdapter()
        raise ValueError(
            f"Unsupported summarizer env: '{env}'. "
            f"Choose from: local, openai, gemini"
        )

    @staticmethod
    def create_tts(
        env: str = "local", language: str | None = None,
    ) -> TTSStrategy:
        logger.info("Creating TTS: env=%s  lang=%s", env, language)
        if env == "local":
            return LocalTTSAdapter(lang=language)
        if env in ("fpt", "cloud"):  # "cloud" kept as legacy alias
            api_key = settings.FPT_API_KEY
            if not api_key:
                raise ValueError("FPT_API_KEY is not configured for FPT TTS")
            return FPTCloudTTSAdapter(api_key=api_key)
        if env == "openai":
            return OpenAITTSAdapter()
        if env == "elevenlabs":
            return ElevenLabsTTSAdapter()
        raise ValueError(
            f"Unsupported TTS env: '{env}'. "
            f"Choose from: local, fpt, openai, elevenlabs"
        )