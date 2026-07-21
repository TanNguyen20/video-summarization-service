"""Transcription adapters — local and cloud providers.

Providers:
    - WhisperX   (local, GPU/CPU)
    - OpenAI     (cloud, Whisper API)
    - AssemblyAI (cloud, sentence-level timestamps)
    - Gemini     (cloud, multimodal audio processing)
"""

import json
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
        self.device = device or settings.resolved_device
        self._compute_type = compute_type or settings.resolved_compute_type
        self._model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        import whisperx

        logger.info(
            "Loading WhisperX  model=%s  device=%s  compute=%s",
            self._model_size,
            self.device,
            self._compute_type,
        )
        self.model = whisperx.load_model(
            self._model_size, self.device, compute_type=self._compute_type,
        )

    def _free_gpu_memory(self) -> None:
        """Force garbage collection and return cached VRAM to the OS."""
        import gc

        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

    def transcribe(self, audio_path: str) -> List[Dict]:
        import whisperx

        if self.model is None:  # reload after a previous cleanup()
            self._load_model()

        logger.info("Transcribing: %s", audio_path)
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=16)

        language = result.get("language", "en")
        logger.info("Detected language: %s", language)

        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=self.device,
        )
        try:
            aligned = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        finally:
            # The align model is single-use — release its VRAM immediately
            del model_a
            self._free_gpu_memory()

        segments = aligned["segments"]
        logger.info("Transcription complete: %d segments", len(segments))
        return segments

    def cleanup(self) -> None:
        """Release the Whisper model and return its VRAM to the OS."""
        if self.model is None:
            return
        logger.info("Releasing WhisperX model (device=%s)", self.device)
        self.model = None
        self._free_gpu_memory()


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


# ═══════════════════════════════════════════════════════════
#  Cloud — AssemblyAI
# ═══════════════════════════════════════════════════════════

class AssemblyAIAdapter(TranscriptionStrategy):
    """Transcribe audio via the AssemblyAI API.

    Returns sentence-level timestamped segments.  Automatically detects
    the spoken language.
    """

    def __init__(self, api_key: str | None = None):
        try:
            import assemblyai as aai
        except ImportError as exc:
            raise ImportError(
                "Install the 'assemblyai' package: pip install assemblyai"
            ) from exc

        _api_key = api_key or settings.ASSEMBLYAI_API_KEY
        if not _api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY is required for AssemblyAI transcription"
            )

        aai.settings.api_key = _api_key
        self._aai = aai

    def transcribe(self, audio_path: str) -> List[Dict]:
        logger.info("Transcribing via AssemblyAI: %s", audio_path)

        config = self._aai.TranscriptionConfig(
            language_detection=True,
        )
        transcriber = self._aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == self._aai.TranscriptStatus.error:
            raise RuntimeError(
                f"AssemblyAI transcription failed: {transcript.error}"
            )

        sentences = transcript.get_sentences()
        segments: List[Dict] = []
        for sentence in sentences:
            segments.append({
                "start": sentence.start / 1000.0,   # ms → seconds
                "end": sentence.end / 1000.0,
                "text": sentence.text,
            })

        logger.info(
            "AssemblyAI transcription complete: %d segments", len(segments),
        )
        return segments


# ═══════════════════════════════════════════════════════════
#  Cloud — Google Gemini (multimodal)
# ═══════════════════════════════════════════════════════════

class GeminiTranscriptionAdapter(TranscriptionStrategy):
    """Transcribe audio via the Google Gemini multimodal API.

    Uploads the audio file and prompts Gemini to return timestamped
    segments as structured JSON.

    Note: Timestamps are LLM-estimated and may be less precise than
    dedicated speech-to-text services (WhisperX, OpenAI, AssemblyAI).
    Best suited for short-to-medium audio where convenience outweighs
    frame-level accuracy.
    """

    _PROMPT = (
        "Transcribe this audio file into timestamped segments.\n\n"
        "Return a JSON object with a single key \"segments\" containing an array. "
        "Each element must have exactly:\n"
        '  - "start": float (seconds)\n'
        '  - "end":   float (seconds)\n'
        '  - "text":  string\n\n'
        "Example:\n"
        '{"segments": [{"start": 0.0, "end": 5.2, "text": "Hello world"}]}'
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Install the 'google-genai' package: pip install google-genai"
            ) from exc

        _api_key = api_key or settings.GEMINI_API_KEY
        if not _api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for Gemini transcription"
            )

        self.client = genai.Client(api_key=_api_key)
        self.model = model or settings.GEMINI_MODEL
        self._genai = genai

    def transcribe(self, audio_path: str) -> List[Dict]:
        logger.info("Transcribing via Gemini (%s): %s", self.model, audio_path)

        # Upload audio file to Gemini
        uploaded = self.client.files.upload(file=audio_path)
        logger.debug("Uploaded audio: name=%s", uploaded.name)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded, self._PROMPT],
            config=self._genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        # Parse structured JSON response
        try:
            parsed = json.loads(response.text)
            raw_segments = parsed.get("segments", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            raise RuntimeError(
                f"Gemini returned invalid transcription JSON: {exc}"
            ) from exc

        segments: List[Dict] = []
        for seg in raw_segments:
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": str(seg["text"]),
            })

        # Clean up uploaded file
        try:
            self.client.files.delete(name=uploaded.name)
        except Exception:
            logger.debug("Could not delete uploaded file: %s", uploaded.name)

        logger.info(
            "Gemini transcription complete: %d segments", len(segments),
        )
        return segments

