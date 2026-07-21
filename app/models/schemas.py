from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskStatusEnum(str, Enum):
    """Possible states for a summarization task."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SceneEmotion(str, Enum):
    """Emotional tone of a scene, detected by the LLM from transcript context.

    Bounded vocabulary so every TTS adapter can map each value to concrete
    voice parameters (OpenAI instructions, ElevenLabs voice settings).
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    SAD = "sad"
    SERIOUS = "serious"
    TENSE = "tense"
    CALM = "calm"
    HUMOROUS = "humorous"


class VideoResolution(str, Enum):
    """Output resolution / aspect-ratio presets tuned for social platforms.

    Portrait presets (mobile, tablet, square) are what TikTok, Facebook/
    Instagram Reels and Stories expect; ``desktop`` is standard landscape.
    """

    MOBILE = "mobile"      # 1080x1920, 9:16  — TikTok, FB/IG Reels & Stories
    TABLET = "tablet"      # 1080x1350, 4:5   — FB/IG portrait feed
    SQUARE = "square"      # 1080x1080, 1:1   — square feed
    DESKTOP = "desktop"    # 1920x1080, 16:9  — YouTube, FB feed, landscape
    ORIGINAL = "original"  # keep the source dimensions unchanged


# Target (width, height) in pixels for each preset. ``None`` = keep source.
RESOLUTION_DIMENSIONS: dict[str, tuple[int, int] | None] = {
    VideoResolution.MOBILE.value: (1080, 1920),
    VideoResolution.TABLET.value: (1080, 1350),
    VideoResolution.SQUARE.value: (1080, 1080),
    VideoResolution.DESKTOP.value: (1920, 1080),
    VideoResolution.ORIGINAL.value: None,
}


class VideoFit(str, Enum):
    """How the source frame is mapped into the target resolution."""

    BLUR = "blur"        # fit whole frame, fill margins with a blurred copy
    COVER = "cover"      # scale to fill and center-crop (no bars, crops edges)
    CONTAIN = "contain"  # scale to fit, letterbox with black bars (no cropping)


class SummaryScene(BaseModel):
    """A single scene extracted by the LLM summarizer."""

    model_config = ConfigDict(use_enum_values=True)

    start_time: float = Field(..., description="Scene start time in seconds")
    end_time: float = Field(..., description="Scene end time in seconds")
    summary_text: str = Field(..., min_length=1, description="Narration text for this scene")
    emotion: SceneEmotion = Field(
        default=SceneEmotion.NEUTRAL,
        validate_default=True,  # so use_enum_values also applies to the default
        description="Emotional tone of the scene, used to style TTS narration",
    )

    @field_validator("emotion", mode="before")
    @classmethod
    def _normalize_emotion(cls, value):
        """Coerce free-form LLM output into the bounded vocabulary.

        An unexpected label must not invalidate an otherwise good scene,
        so anything unrecognized falls back to neutral.
        """
        if value is None:
            return SceneEmotion.NEUTRAL
        if isinstance(value, str):
            value = value.strip().lower()
            if value not in SceneEmotion._value2member_map_:
                return SceneEmotion.NEUTRAL
        return value


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
    tts_env: str = Field(
        default="local",
        description="TTS backend: 'local', 'fpt', 'openai', 'elevenlabs', or 'cloud' (first configured cloud provider)",
    )
    language: str = Field(
        default="vi",
        description=(
            "Target language code (ISO 639-1). Summary narration is "
            "translated into this language and spoken by TTS"
        ),
    )
    resolution: VideoResolution = Field(
        default=VideoResolution.MOBILE,
        description=(
            "Output resolution preset. 'mobile' (1080x1920, 9:16) is tuned "
            "for TikTok / Facebook Reels; 'tablet' (4:5), 'square' (1:1), "
            "'desktop' (16:9), or 'original' to keep the source size"
        ),
    )
    fit: VideoFit = Field(
        default=VideoFit.BLUR,
        description=(
            "How the source frame fills the target aspect ratio: 'blur' "
            "(blurred background, no content lost), 'cover' (center-crop to "
            "fill), or 'contain' (letterbox with black bars)"
        ),
    )