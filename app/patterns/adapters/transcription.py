"""WhisperX transcription adapter."""

from typing import Dict, List

import whisperx

from app.core.config import settings
from app.core.logging import get_logger
from app.patterns.interfaces import TranscriptionStrategy

logger = get_logger("adapters.transcription")


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
