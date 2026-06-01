"""Shared prompt builder for summarization adapters.

Extracted so all LLM adapters (Ollama, OpenAI, Gemini) use the same
prompt structure without duplication.
"""


SYSTEM_INSTRUCTION = (
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


def build_summarization_prompt(transcript: str) -> str:
    """Build the full user prompt including the transcript."""
    return f"{SYSTEM_INSTRUCTION}\n\nTranscript:\n{transcript}"
