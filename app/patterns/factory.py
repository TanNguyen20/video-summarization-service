from app.core.config import settings
from app.core.logging import get_logger
from app.patterns.adapters import (
    FPTCloudTTSAdapter,
    LocalLLMAdapter,
    LocalTTSAdapter,
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
    """

    @staticmethod
    def create_transcriber(env: str = "local") -> TranscriptionStrategy:
        logger.info("Creating transcriber: env=%s", env)
        if env == "local":
            return WhisperXLocalAdapter()
        raise ValueError(f"Unsupported transcriber env: {env}")

    @staticmethod
    def create_summarizer(env: str = "local") -> SummarizationStrategy:
        logger.info("Creating summarizer: env=%s", env)
        if env == "local":
            return LocalLLMAdapter()
        raise ValueError(f"Unsupported summarizer env: {env}")

    @staticmethod
    def create_tts(
        env: str = "local", language: str | None = None,
    ) -> TTSStrategy:
        logger.info("Creating TTS: env=%s  lang=%s", env, language)
        if env == "local":
            return LocalTTSAdapter(lang=language)
        if env == "cloud":
            api_key = settings.FPT_API_KEY
            if not api_key:
                raise ValueError("FPT_API_KEY is not configured for cloud TTS")
            return FPTCloudTTSAdapter(api_key=api_key)
        raise ValueError(f"Unsupported TTS env: {env}")