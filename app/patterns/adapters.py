import json
import time
from typing import List, Dict

import requests
import whisperx
from gtts import gTTS
from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import SummaryScene
from app.patterns.interfaces import (
    TranscriptionStrategy,
    SummarizationStrategy,
    TTSStrategy,
)

logger = get_logger("adapters")


# ═══════════════════════════════════════════════════════════
#  Transcription
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


# ═══════════════════════════════════════════════════════════
#  Summarization
# ═══════════════════════════════════════════════════════════

class LocalLLMAdapter(SummarizationStrategy):
    """Summarize transcripts via a local Ollama LLM."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.url = ollama_url or settings.OLLAMA_URL
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def summarize(self, transcript: str) -> List[Dict]:
        logger.info(
            "Summarizing transcript (%d chars) with model=%s",
            len(transcript),
            self.model,
        )

        prompt = self._build_prompt(transcript)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }

        # ── Call Ollama ────────────────────────────────────
        try:
            response = requests.post(
                self.url, json=payload, timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("LLM request failed: %s", exc)
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        # ── Parse response ────────────────────────────────
        try:
            raw = response.json()["response"]
            parsed = json.loads(raw)
            scenes_raw = parsed.get("scenes", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse LLM response: %s", exc)
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        # ── Validate each scene ───────────────────────────
        scenes: List[Dict] = []
        for idx, raw_scene in enumerate(scenes_raw):
            try:
                scene = SummaryScene(**raw_scene)
                scenes.append(scene.model_dump())
            except ValidationError as exc:
                logger.warning("Skipping invalid scene %d: %s", idx, exc)

        if not scenes:
            raise RuntimeError("LLM produced zero valid summary scenes")

        logger.info("Summarization complete: %d scenes", len(scenes))
        return scenes

    @staticmethod
    def _build_prompt(transcript: str) -> str:
        return (
            "You are a video summarization assistant. "
            "Given the following timestamped transcript, extract the key scenes.\n\n"
            'Return a JSON object with a single key "scenes" containing an array of objects. '
            "Each object must have exactly these keys:\n"
            '  - "start_time": float (seconds)\n'
            '  - "end_time":   float (seconds)\n'
            '  - "summary_text": string (a concise narration of that scene)\n\n'
            "Example output:\n"
            '{"scenes": [{"start_time": 0.0, "end_time": 15.5, '
            '"summary_text": "Introduction to the topic"}]}\n\n'
            f"Transcript:\n{transcript}"
        )


# ═══════════════════════════════════════════════════════════
#  Text-to-Speech
# ═══════════════════════════════════════════════════════════

class FPTCloudTTSAdapter(TTSStrategy):
    """Generate speech audio via the FPT.AI TTS cloud API.

    Implements polling for the asynchronous audio URL.
    """

    POLL_INTERVAL: float = 1.0
    POLL_TIMEOUT: float = 60.0

    def __init__(self, api_key: str, voice: str = "banmai"):
        if not api_key:
            raise ValueError("FPT API key is required for cloud TTS")
        self.api_key = api_key
        self.voice = voice
        self.url = "https://api.fpt.ai/hmi/tts/v5"

    def generate_audio(self, text: str, output_path: str) -> str:
        logger.info("Requesting FPT TTS for %d chars", len(text))
        headers = {"api-key": self.api_key}
        payload = {"text": text, "voice": self.voice, "speed": "0"}

        try:
            response = requests.post(
                self.url, headers=headers, data=payload, timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"FPT TTS request failed: {exc}") from exc

        audio_url = data.get("async")
        if not audio_url:
            raise RuntimeError(f"FPT TTS did not return async URL: {data}")

        audio_data = self._poll_for_audio(audio_url)
        with open(output_path, "wb") as f:
            f.write(audio_data)

        logger.info("FPT TTS audio saved: %s", output_path)
        return output_path

    def _poll_for_audio(self, url: str) -> bytes:
        """Poll the async URL until the audio is ready or timeout."""
        elapsed = 0.0
        while elapsed < self.POLL_TIMEOUT:
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 0:
                    return resp.content
            except requests.RequestException:
                logger.debug("Polling FPT TTS … (%.1fs)", elapsed)

        raise RuntimeError(
            f"FPT TTS polling timed out after {self.POLL_TIMEOUT}s"
        )


class LocalTTSAdapter(TTSStrategy):
    """Generate speech audio locally using Google TTS (gTTS)."""

    def __init__(self, lang: str | None = None):
        self.lang = lang or settings.DEFAULT_TTS_LANG

    def generate_audio(self, text: str, output_path: str) -> str:
        logger.info("Generating local TTS (lang=%s, %d chars)", self.lang, len(text))
        tts = gTTS(text, lang=self.lang)
        tts.save(output_path)
        logger.info("Local TTS saved: %s", output_path)
        return output_path