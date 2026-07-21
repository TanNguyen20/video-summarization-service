"""Shared prompt builder for summarization adapters.

Extracted so all LLM adapters (Ollama, OpenAI, Gemini) use the same
prompt structure without duplication.
"""

from app.models.schemas import SceneEmotion

# ISO 639-1 codes mapped to English language names for the LLM prompt.
# Falls back to the raw code for anything not listed.
LANGUAGE_NAMES = {
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
    "id": "Indonesian",
}

DEFAULT_LANGUAGE = "vi"

# Approximate spoken pace used to size each scene's narration so it roughly
# fills the scene's video segment (end_time - start_time). Values map a
# language code to (unit, rate):
#   - "words" per second for space-delimited languages
#   - "chars" per second for languages written without word spacing
# These are rough averages of natural TTS pacing — tune per your TTS voice
# if narration consistently runs short or long. [Inference: approximate]
SPEAKING_RATE: dict[str, tuple[str, float]] = {
    "vi": ("words", 2.3),
    "en": ("words", 2.5),
    "fr": ("words", 2.6),
    "de": ("words", 2.2),
    "es": ("words", 2.7),
    "id": ("words", 2.3),
    "ko": ("words", 2.4),
    "zh": ("chars", 4.5),
    "ja": ("chars", 5.5),
    "th": ("chars", 5.0),
}
DEFAULT_RATE: tuple[str, float] = ("words", 2.4)

_EMOTION_VALUES = ", ".join(e.value for e in SceneEmotion)

_BASE_INSTRUCTION = (
    "You are a video summarization assistant. "
    "Given a timestamped transcript, extract the key scenes.\n\n"
    'Return a JSON object with a single key "scenes" containing an array. '
    "Each object must have exactly these keys:\n"
    '  - "start_time": float (seconds)\n'
    '  - "end_time":   float (seconds)\n'
    '  - "summary_text": string (a concise narration of that scene)\n'
    f'  - "emotion": string, exactly one of: {_EMOTION_VALUES}\n\n'
    "Infer each scene's emotion from the transcript context — word choice, "
    "events described, and overall mood — and write summary_text in a tone "
    "that matches it: energetic phrasing for excited scenes, gentle and "
    "somber phrasing for sad ones, composed phrasing for serious ones. "
    "Do not assign an emotion the content does not support; "
    'use "neutral" when in doubt.\n\n'
    "Example:\n"
    '{"scenes": [{"start_time": 0.0, "end_time": 15.5, '
    '"summary_text": "Introduction to the topic", "emotion": "neutral"}, '
    '{"start_time": 15.5, "end_time": 42.0, '
    '"summary_text": "The home team scores a stunning last-minute goal!", '
    '"emotion": "excited"}]}'
)


def language_name(language: str | None) -> str:
    """Resolve a language code to its English name (falls back to the code)."""
    code = (language or DEFAULT_LANGUAGE).lower()
    return LANGUAGE_NAMES.get(code, code)


def speaking_rate(language: str | None) -> tuple[str, float]:
    """Return the (unit, rate) spoken-pace estimate for *language*."""
    code = (language or DEFAULT_LANGUAGE).lower()
    return SPEAKING_RATE.get(code, DEFAULT_RATE)


def build_pacing_instruction(language: str | None = None) -> str:
    """Tell the LLM to size each scene's narration to its segment duration.

    The narration is spoken over its own video segment, so ``summary_text``
    should take roughly as long to speak as ``end_time - start_time``.
    Without this, the model tends to write terse text that finishes early
    and leaves the rest of the segment playing in silence.
    """
    unit, rate = speaking_rate(language)
    return (
        "PACING: Each scene's narration is spoken aloud over its own video "
        "segment, so it should take about as long to say as the segment "
        f"lasts. Assume a speaking pace of roughly {rate:g} {unit} per "
        "second. For a scene lasting D seconds (D = end_time - start_time), "
        f"write summary_text of about D x {rate:g} {unit}. Expand with "
        "relevant detail drawn from the transcript to fill longer segments "
        "rather than leaving silent gaps, and keep it concise for short "
        "ones — but never pad with filler, repetition, or invented facts."
    )


def build_system_instruction(language: str | None = None) -> str:
    """Build the system instruction, forcing narration into *language*.

    The transcript may be in any language; ``summary_text`` must always
    come back in the requested target language so downstream TTS speaks
    the language the user asked for.
    """
    target = language_name(language)
    return (
        f"{_BASE_INSTRUCTION}\n\n"
        f"{build_pacing_instruction(language)}\n\n"
        f'IMPORTANT: Write every "summary_text" value in {target}. '
        f"If the transcript is in a different language, translate the "
        f"narration into {target}. Keep timestamps and JSON keys unchanged."
    )


def build_summarization_prompt(
    transcript: str, language: str | None = None,
) -> str:
    """Build the full user prompt including the transcript."""
    return f"{build_system_instruction(language)}\n\nTranscript:\n{transcript}"


# Backwards-compatible module-level constant (default language).
SYSTEM_INSTRUCTION = build_system_instruction()
