"""Text-to-Speech adapters (local gTTS and FPT Cloud)."""

import time

import requests
from gtts import gTTS

from app.core.config import settings
from app.core.logging import get_logger
from app.patterns.interfaces import TTSStrategy

logger = get_logger("adapters.tts")


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
