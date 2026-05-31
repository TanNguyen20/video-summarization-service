"""Local LLM summarization adapter (Ollama)."""

import json
from typing import Dict, List

import requests
from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import SummaryScene
from app.patterns.interfaces import SummarizationStrategy

logger = get_logger("adapters.summarization")


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
