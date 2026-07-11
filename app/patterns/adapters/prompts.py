"""Shared prompt builder for summarization adapters.

Extracted so all LLM adapters (Ollama, OpenAI, Gemini) use the same
prompt structure without duplication.
"""


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

_BASE_INSTRUCTION = (
    "You are a video summarization assistant. "
    "Given a timestamped transcript, extract the key scenes.\n\n"
    'Return a JSON object with a single key "scenes" containing an array. '
    "Each object must have exactly these keys:\n"
    '  - "start_time": float (seconds)\n'
    '  - "end_time":   float (seconds)\n'
    '  - "summary_text": string (a concise narration of that scene)\n\n'
    "Example:\n"
    '{"scenes": [{"start_time": 0.0, "end_time": 15.5, '
    '"summary_text": "Introduction to the topic"}]}'
)


def language_name(language: str | None) -> str:
    """Resolve a language code to its English name (falls back to the code)."""
    code = (language or DEFAULT_LANGUAGE).lower()
    return LANGUAGE_NAMES.get(code, code)


def build_system_instruction(language: str | None = None) -> str:
    """Build the system instruction, forcing narration into *language*.

    The transcript may be in any language; ``summary_text`` must always
    come back in the requested target language so downstream TTS speaks
    the language the user asked for.
    """
    target = language_name(language)
    return (
        f"{_BASE_INSTRUCTION}\n\n"
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
