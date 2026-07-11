"""Summarization adapters — local (Ollama), OpenAI, and Google Gemini."""

import json
from typing import Dict, List

import requests
from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import SummaryScene
from app.patterns.adapters.prompts import (
    build_summarization_prompt,
    build_system_instruction,
)
from app.patterns.interfaces import SummarizationStrategy

logger = get_logger("adapters.summarization")


# ─── Shared helpers ───────────────────────────────────────

def _validate_scenes(raw_json: str, source: str) -> List[Dict]:
    """Parse JSON and validate each scene through the Pydantic model.

    Shared across all summarization adapters to ensure consistent output.
    """
    try:
        parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        scenes_raw = parsed.get("scenes", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error("[%s] Failed to parse response JSON: %s", source, exc)
        raise RuntimeError(f"{source} returned invalid JSON: {exc}") from exc

    scenes: List[Dict] = []
    for idx, raw_scene in enumerate(scenes_raw):
        try:
            scene = SummaryScene(**raw_scene)
            scenes.append(scene.model_dump())
        except ValidationError as exc:
            logger.warning("[%s] Skipping invalid scene %d: %s", source, idx, exc)

    if not scenes:
        raise RuntimeError(f"{source} produced zero valid summary scenes")

    logger.info("[%s] Summarization complete: %d scenes", source, len(scenes))
    return scenes


# ═══════════════════════════════════════════════════════════
#  Local — Ollama
# ═══════════════════════════════════════════════════════════

class LocalLLMAdapter(SummarizationStrategy):
    """Summarize transcripts via a local Ollama LLM."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        language: str | None = None,
    ):
        self.url = ollama_url or settings.OLLAMA_URL
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.language = language

    def summarize(self, transcript: str) -> List[Dict]:
        logger.info(
            "Summarizing transcript (%d chars) with Ollama model=%s lang=%s",
            len(transcript),
            self.model,
            self.language,
        )

        payload = {
            "model": self.model,
            "prompt": build_summarization_prompt(transcript, self.language),
            "format": "json",
            "stream": False,
            "think": settings.LLM_THINK,
        }

        try:
            response = requests.post(
                self.url, json=payload, timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            raw = response.json()["response"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(f"Ollama returned invalid response: {exc}") from exc

        return _validate_scenes(raw, source="Ollama")


# ═══════════════════════════════════════════════════════════
#  Cloud — OpenAI (GPT-4o / GPT-4o-mini)
# ═══════════════════════════════════════════════════════════

class OpenAISummarizationAdapter(SummarizationStrategy):
    """Summarize transcripts via the OpenAI Chat Completions API.

    Uses JSON mode for reliable structured output.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the 'openai' package: pip install openai"
            ) from exc

        _api_key = api_key or settings.OPENAI_API_KEY
        if not _api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI summarization")

        self.client = OpenAI(api_key=_api_key)
        self.model = model or settings.OPENAI_MODEL
        self.language = language

    def summarize(self, transcript: str) -> List[Dict]:
        logger.info(
            "Summarizing transcript (%d chars) with OpenAI model=%s lang=%s",
            len(transcript),
            self.model,
            self.language,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": build_system_instruction(self.language)},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        return _validate_scenes(raw, source="OpenAI")


# ═══════════════════════════════════════════════════════════
#  Cloud — Google Gemini
# ═══════════════════════════════════════════════════════════

class GeminiSummarizationAdapter(SummarizationStrategy):
    """Summarize transcripts via the Google Gemini API.

    Uses the ``google-genai`` SDK with JSON response mode.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Install the 'google-genai' package: pip install google-genai"
            ) from exc

        _api_key = api_key or settings.GEMINI_API_KEY
        if not _api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini summarization")

        self.client = genai.Client(api_key=_api_key)
        self.model = model or settings.GEMINI_MODEL
        self._genai = genai
        self.language = language

    def summarize(self, transcript: str) -> List[Dict]:
        logger.info(
            "Summarizing transcript (%d chars) with Gemini model=%s lang=%s",
            len(transcript),
            self.model,
            self.language,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=build_summarization_prompt(transcript, self.language),
            config=self._genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )

        raw = response.text
        return _validate_scenes(raw, source="Gemini")
