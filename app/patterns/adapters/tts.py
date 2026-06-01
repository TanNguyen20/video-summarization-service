"""Text-to-Speech adapters — local and cloud providers.

Providers:
    - gTTS         (local/free, Google Translate TTS)
    - FPT.AI       (cloud, Vietnamese-focused)
    - OpenAI TTS   (cloud, high-quality HD voices)
    - ElevenLabs   (cloud, premium multilingual voices)
"""

import time

import requests
from gtts import gTTS

from app.core.config import settings
from app.core.logging import get_logger
from app.patterns.interfaces import TTSStrategy

logger = get_logger("adapters.tts")


# ═══════════════════════════════════════════════════════════
#  Local — gTTS (Google Translate TTS)
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  Cloud — FPT.AI
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


# ═══════════════════════════════════════════════════════════
#  Cloud — OpenAI TTS
# ═══════════════════════════════════════════════════════════

class OpenAITTSAdapter(TTSStrategy):
    """Generate speech audio via the OpenAI TTS API.

    Supports models ``tts-1`` (fast) and ``tts-1-hd`` (high quality).

    Available voices: alloy, ash, ballad, coral, echo, fable,
    onyx, nova, sage, shimmer.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "tts-1",
        voice: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the 'openai' package: pip install openai"
            ) from exc

        _api_key = api_key or settings.OPENAI_API_KEY
        if not _api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI TTS")

        self.client = OpenAI(api_key=_api_key)
        self.model = model
        self.voice = voice or settings.OPENAI_TTS_VOICE

    def generate_audio(self, text: str, output_path: str) -> str:
        logger.info(
            "Generating OpenAI TTS (model=%s, voice=%s, %d chars)",
            self.model,
            self.voice,
            len(text),
        )

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="mp3",
        )
        response.stream_to_file(output_path)

        logger.info("OpenAI TTS saved: %s", output_path)
        return output_path


# ═══════════════════════════════════════════════════════════
#  Cloud — ElevenLabs
# ═══════════════════════════════════════════════════════════

class ElevenLabsTTSAdapter(TTSStrategy):
    """Generate speech audio via the ElevenLabs API.

    Premium multilingual voices with natural intonation.
    Uses the ``eleven_multilingual_v2`` model by default.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = "eleven_multilingual_v2",
    ):
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise ImportError(
                "Install the 'elevenlabs' package: pip install elevenlabs"
            ) from exc

        _api_key = api_key or settings.ELEVENLABS_API_KEY
        if not _api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is required for ElevenLabs TTS"
            )

        self.client = ElevenLabs(api_key=_api_key)
        self.voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
        self.model_id = model_id

    def generate_audio(self, text: str, output_path: str) -> str:
        logger.info(
            "Generating ElevenLabs TTS (voice=%s, %d chars)",
            self.voice_id,
            len(text),
        )

        audio_generator = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )

        with open(output_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        logger.info("ElevenLabs TTS saved: %s", output_path)
        return output_path

