from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    # ── Server ──────────────────────────────────────────────
    PROJECT_NAME: str = "Video Summarization API"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database ────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/video_summarization"

    # ── LLM (Ollama — local) ────────────────────────────────
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_TIMEOUT: int = 120
    LLM_MODEL: str = "qwen3:14b"
    # Thinking models (qwen3) emit "{}" under format=json unless thinking is off
    LLM_THINK: bool = False

    # ── Cloud Providers ────────────────────────────────────
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_TTS_VOICE: str = "alloy"
    # gpt-4o-mini-tts supports emotion instructions; tts-1/tts-1-hd do not
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.0-flash"
    ASSEMBLYAI_API_KEY: str | None = None
    FPT_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None
    ELEVENLABS_VOICE_ID: str = "JBFqnCBsd6RMkjVDRZzb"
    # multilingual_v2 does not support Vietnamese; flash_v2_5 does
    ELEVENLABS_MODEL_ID: str = "eleven_flash_v2_5"

    # ── Whisper ─────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "large-v2"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "float16"

    # ── TTS ─────────────────────────────────────────────────
    DEFAULT_TTS_LANG: str = "vi"

    # ── Directories ─────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"

    # ── Upload limits ───────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: str = ".mp4,.mkv,.avi,.mov,.webm"

    # ── Logging ─────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Computed properties ─────────────────────────────────

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def resolved_device(self) -> str:
        """Return the actual compute device, auto-detecting CUDA availability."""
        if self.WHISPER_DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.WHISPER_DEVICE

    @property
    def resolved_compute_type(self) -> str:
        """Downgrade float16 to int8 when running on CPU."""
        if self.resolved_device == "cpu" and self.WHISPER_COMPUTE_TYPE == "float16":
            return "int8"
        return self.WHISPER_COMPUTE_TYPE


settings = Settings()