import os
import shutil
import tempfile

import ffmpeg
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)
from PIL import Image, ImageFilter

from app.core.logging import get_logger
from app.patterns.interfaces import (
    SummarizationStrategy,
    TTSStrategy,
    TranscriptionStrategy,
)

logger = get_logger("pipeline")


class VideoSummarizationPipeline:
    """Orchestrates the full video-summarization workflow.

    Steps
    -----
    1. Extract audio from the input video.
    2. Transcribe the audio into timestamped segments.
    3. Summarize the transcript into key scenes via an LLM.
    4. Generate TTS narration for each scene.
    5. Compose the final video with narration and crossfade transitions.
    """

    def __init__(
        self,
        transcriber: TranscriptionStrategy,
        summarizer: SummarizationStrategy,
        tts: TTSStrategy,
    ) -> None:
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.tts = tts

    def process(
        self,
        input_video_path: str,
        output_video_path: str,
        task_id: str = "default",
        target_size: tuple[int, int] | None = None,
        fit: str = "blur",
    ) -> None:
        """Run the full pipeline, writing the result to *output_video_path*.

        Parameters
        ----------
        target_size:
            Desired ``(width, height)`` of the exported video. ``None`` keeps
            the source dimensions. Presets live in ``RESOLUTION_DIMENSIONS``;
            ``(1080, 1920)`` is the 9:16 mobile format for TikTok/Facebook.
        fit:
            How the source frame maps into ``target_size`` — ``"blur"``,
            ``"cover"``, or ``"contain"``. Ignored when ``target_size`` is None.
        """

        work_dir = tempfile.mkdtemp(prefix=f"vsapi_{task_id}_")
        logger.info("Pipeline started | task=%s | workdir=%s", task_id, work_dir)

        clips_to_close: list = []

        try:
            # Step 1 — Extract audio
            audio_path = os.path.join(work_dir, "audio.wav")
            logger.info("Step 1/4: Extracting audio")
            ffmpeg.input(input_video_path).output(
                audio_path, ac=1, ar="16k", loglevel="quiet",
            ).run(overwrite_output=True)

            # Step 2 — Transcribe
            logger.info("Step 2/4: Transcribing")
            segments = self.transcriber.transcribe(audio_path)
            self.transcriber.cleanup()  # free GPU before the LLM stage
            flat_transcript = " ".join(
                f"[{s['start']:.2f} -> {s['end']:.2f}] {s['text']}"
                for s in segments
            )

            # Step 3 — Summarize
            logger.info("Step 3/4: Summarizing")
            summary_scenes = self.summarizer.summarize(flat_transcript)

            # Step 4 — Compose video
            logger.info(
                "Step 4/4: Composing video (%d scenes)", len(summary_scenes),
            )
            video = VideoFileClip(input_video_path)
            clips_to_close.append(video)

            final_clips: list = []

            for i, scene in enumerate(summary_scenes):
                tts_path = os.path.join(work_dir, f"tts_{i}.mp3")
                self.tts.generate_audio(
                    scene["summary_text"],
                    tts_path,
                    emotion=scene.get("emotion"),
                )

                tts_audio = AudioFileClip(tts_path)
                clips_to_close.append(tts_audio)

                clip = video.subclipped(scene["start_time"], scene["end_time"])
                clips_to_close.append(clip)

                video_duration = scene["end_time"] - scene["start_time"]
                tts_duration = tts_audio.duration

                if tts_duration > video_duration:
                    extra = tts_duration - video_duration
                    last_frame = clip.get_frame(clip.duration - 0.01)
                    freeze = ImageClip(last_frame, duration=extra)
                    clips_to_close.append(freeze)
                    clip = concatenate_videoclips([clip, freeze])
                    clips_to_close.append(clip)

                clip = clip.with_audio(tts_audio)
                final_clips.append(clip)

            # Crossfade transitions
            if len(final_clips) > 1:
                transition = 0.5
                processed = [final_clips[0]]
                for c in final_clips[1:]:
                    processed.append(
                        c.with_effects([vfx.CrossFadeIn(transition)]),
                    )
                final_video = concatenate_videoclips(
                    processed, padding=-transition, method="compose",
                )
            elif final_clips:
                final_video = final_clips[0]
            else:
                raise RuntimeError("No clips to compose")

            clips_to_close.append(final_video)

            # Reshape to the requested export resolution (e.g. 9:16 mobile)
            if target_size is not None:
                logger.info(
                    "Fitting output to %dx%d (fit=%s)",
                    target_size[0], target_size[1], fit,
                )
                final_video = self._fit_to_resolution(
                    final_video, target_size, fit,
                )
                clips_to_close.append(final_video)

            final_video.write_videofile(
                output_video_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                logger=None,
            )
            logger.info("Pipeline complete | output=%s", output_video_path)

        finally:
            for clip in reversed(clips_to_close):
                try:
                    clip.close()
                except Exception:
                    pass
            # Release adapter resources (GPU models) even on failure
            for component in (self.transcriber, self.summarizer, self.tts):
                try:
                    component.cleanup()
                except Exception as exc:
                    logger.warning("Component cleanup failed: %s", exc)
            self._cleanup_dir(work_dir)

    @staticmethod
    def _fit_to_resolution(clip, target_size: tuple[int, int], fit: str):
        """Reshape *clip* to *target_size* using the chosen *fit* strategy.

        - ``cover``   : scale to fill the frame, then center-crop (crops edges).
        - ``contain`` : scale to fit inside the frame, letterbox with black.
        - ``blur``    : fit the whole frame, fill the margins with a blurred,
          upscaled copy of the source — the standard look for reposting
          landscape footage to a vertical feed without losing content.
        """
        tw, th = target_size
        sw, sh = clip.w, clip.h

        # Already the right size — nothing to do.
        if (sw, sh) == (tw, th):
            return clip

        def cover(c):
            """Scale by the larger ratio and center-crop to exactly target."""
            scale = max(tw / c.w, th / c.h)
            resized = c.resized(scale)
            return resized.cropped(
                x_center=resized.w / 2,
                y_center=resized.h / 2,
                width=tw,
                height=th,
            )

        if fit == "cover":
            return cover(clip).with_audio(clip.audio)

        # Foreground: scale to fit entirely inside the target frame.
        fit_scale = min(tw / sw, th / sh)
        foreground = clip.resized(fit_scale)

        if fit == "contain":
            background = ColorClip(
                size=(tw, th), color=(0, 0, 0), duration=clip.duration,
            )
        else:  # "blur" (default) — blurred, cover-cropped copy of the source
            def blur_frame(frame):
                img = Image.fromarray(frame)
                w, h = img.size
                # Blur a downscaled copy (fast) then scale back up — the
                # detail loss from the blur makes the upscaling invisible.
                small = img.resize((max(1, w // 8), max(1, h // 8)))
                small = small.filter(ImageFilter.GaussianBlur(radius=6))
                return np.asarray(small.resize((w, h)))

            background = cover(clip).image_transform(blur_frame)

        return CompositeVideoClip(
            [background, foreground.with_position("center")],
            size=(tw, th),
        ).with_audio(clip.audio)

    @staticmethod
    def _cleanup_dir(dir_path: str) -> None:
        try:
            shutil.rmtree(dir_path, ignore_errors=True)
            logger.info("Cleaned up workdir: %s", dir_path)
        except Exception as exc:
            logger.warning("Failed to clean workdir %s: %s", dir_path, exc)