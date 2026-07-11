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
                        | "cloud" (legacy alias — first configured cloud provider)
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
    def create_summarizer(
        env: str = "local", language: str | None = None,
    ) -> SummarizationStrategy:
        logger.info("Creating summarizer: env=%s  lang=%s", env, language)
        if env == "local":
            return LocalLLMAdapter(language=language)
        if env == "openai":
            return OpenAISummarizationAdapter(language=language)
        if env == "gemini":
            return GeminiSummarizationAdapter(language=language)
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
        if env == "cloud":  # legacy alias — first cloud provider with a key
            if settings.FPT_API_KEY:
                logger.info("TTS env 'cloud' resolved to: fpt")
                return FPTCloudTTSAdapter(api_key=settings.FPT_API_KEY)
            if settings.ELEVENLABS_API_KEY:
                logger.info("TTS env 'cloud' resolved to: elevenlabs")
                return ElevenLabsTTSAdapter(language=language)
            if settings.OPENAI_API_KEY:
                logger.info("TTS env 'cloud' resolved to: openai")
                return OpenAITTSAdapter()
            raise ValueError(
                "No cloud TTS provider is configured. Set FPT_API_KEY, "
                "ELEVENLABS_API_KEY, or OPENAI_API_KEY — or use tts_env=local"
            )
        if env == "fpt":
            api_key = settings.FPT_API_KEY
            if not api_key:
                raise ValueError("FPT_API_KEY is not configured for FPT TTS")
            return FPTCloudTTSAdapter(api_key=api_key)
        if env == "openai":
            return OpenAITTSAdapter()
        if env == "elevenlabs":
            return ElevenLabsTTSAdapter(language=language)
        raise ValueError(
            f"Unsupported TTS env: '{env}'. "
            f"Choose from: local, fpt, openai, elevenlabs"
        )