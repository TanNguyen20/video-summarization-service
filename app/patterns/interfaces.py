from abc import ABC, abstractmethod
from typing import List, Dict


class TranscriptionStrategy(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict]:
        """Transcribe an audio file and return timestamped segments.

        Each segment dict must contain at least:
            - 'start': float  (seconds)
            - 'end':   float  (seconds)
            - 'text':  str
        """

    def cleanup(self) -> None:
        """Release resources held by the adapter (e.g. GPU model weights).

        Called by the pipeline as soon as the stage is finished, and again
        defensively on teardown — implementations must be idempotent.
        Default is a no-op; adapters holding local models should override.
        """


class SummarizationStrategy(ABC):
    """Abstract interface for transcript summarization."""

    @abstractmethod
    def summarize(self, transcript: str) -> List[Dict]:
        """Summarize a flat transcript into key scenes.

        Returns a list of dicts, each containing:
            - 'start_time':    float
            - 'end_time':      float
            - 'summary_text':  str
            - 'emotion':       str  (SceneEmotion value, e.g. "neutral")
        """

    def cleanup(self) -> None:
        """Release resources held by the adapter. Idempotent no-op by default."""


class TTSStrategy(ABC):
    """Abstract interface for text-to-speech generation."""

    @abstractmethod
    def generate_audio(
        self, text: str, output_path: str, emotion: str | None = None,
    ) -> str:
        """Convert *text* to speech and save to *output_path*.

        *emotion* is an optional tone hint (a SceneEmotion value taken from
        the scene summary). Adapters without emotion control may ignore it.

        Returns the path to the generated audio file.
        """

    def cleanup(self) -> None:
        """Release resources held by the adapter. Idempotent no-op by default."""