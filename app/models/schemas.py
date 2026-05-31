from enum import Enum

from pydantic import BaseModel, Field


class TaskStatusEnum(str, Enum):
    """Possible states for a summarization task."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryScene(BaseModel):
    """A single scene extracted by the LLM summarizer."""

    start_time: float = Field(..., description="Scene start time in seconds")
    end_time: float = Field(..., description="Scene end time in seconds")
    summary_text: str = Field(..., min_length=1, description="Narration text for this scene")


class TaskResponse(BaseModel):
    """API response model for task status queries."""

    task_id: str
    status: TaskStatusEnum
    output_url: str | None = None
    error: str | None = None


class SummarizationOptions(BaseModel):
    """User-provided options for the summarization pipeline."""

    transcriber_env: str = Field(default="local", description="Transcription backend")
    summarizer_env: str = Field(default="local", description="Summarization backend")
    tts_env: str = Field(default="local", description="TTS backend: 'local' or 'cloud'")
    language: str = Field(default="vi", description="Language code for TTS output")