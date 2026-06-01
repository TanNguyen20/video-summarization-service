"""Transcription adapters — local (WhisperX) and cloud (OpenAI Whisper API)."""

from typing import Dict, List

from app.core.config import settings
from app.core.logging import get_logger
from app.patterns.interfaces import TranscriptionStrategy

logger = get_logger("adapters.transcription")


# ═══════════════════════════════════════════════════════════
#  Local — WhisperX
# ═══════════════════════════════════════════════════════════

class WhisperXLocalAdapter(TranscriptionStrategy):
    """Transcribe audio to timestamped segments using WhisperX.

    Supports automatic CPU fallback when CUDA is unavailable.
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        import whisperx

        self.device = device or settings.resolved_device
        _compute_type = compute_type or settings.resolved_compute_type
        _model_size = model_size or settings.WHISPER_MODEL_SIZE

        logger.info(
            "Loading WhisperX  model=%s  device=%s  compute=%s",
            _model_size,
            self.device,
            _compute_type,
        )
        self.model = whisperx.load_model(
            _model_size, self.device, compute_type=_compute_type,
        )

    def transcribe(self, audio_path: str) -> List[Dict]:
        import whisperx

        logger.info("Transcribing: %s", audio_path)
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=16)

        language = result.get("language", "en")
        logger.info("Detected language: %s", language)

        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=self.device,
        )
        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        segments = aligned["segments"]
        logger.info("Transcription complete: %d segments", len(segments))
        return segments


# ═══════════════════════════════════════════════════════════
#  Cloud — OpenAI Whisper API
# ═══════════════════════════════════════════════════════════

class OpenAIWhisperAdapter(TranscriptionStrategy):
    """Transcribe audio via the OpenAI Whisper API.

    Uses the ``openai`` SDK to call the ``audio.transcriptions`` endpoint
    with segment-level timestamps.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the 'openai' package: pip install openai"
            ) from exc

        _api_key = api_key or settings.OPENAI_API_KEY
        if not _api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI transcription")

        self.client = OpenAI(api_key=_api_key)
        self.model = model

    def transcribe(self, audio_path: str) -> List[Dict]:
        logger.info("Transcribing via OpenAI Whisper: %s", audio_path)

        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments: List[Dict] = []
        for seg in result.segments or []:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })

        logger.info(
            "OpenAI transcription complete: %d segments", len(segments),
        )
        return segments
