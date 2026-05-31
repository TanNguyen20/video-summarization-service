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


class SummarizationStrategy(ABC):
    """Abstract interface for transcript summarization."""

    @abstractmethod
    def summarize(self, transcript: str) -> List[Dict]:
        """Summarize a flat transcript into key scenes.

        Returns a list of dicts, each containing:
            - 'start_time':    float
            - 'end_time':      float
            - 'summary_text':  str
        """


class TTSStrategy(ABC):
    """Abstract interface for text-to-speech generation."""

    @abstractmethod
    def generate_audio(self, text: str, output_path: str) -> str:
        """Convert *text* to speech and save to *output_path*.

        Returns the path to the generated audio file.
        """