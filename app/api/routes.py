import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.session import dispose_db, init_db
from app.models.schemas import (
    SummarizationOptions,
    TaskResponse,
    TaskStatusEnum,
)
from app.patterns.factory import ComponentFactory
from app.services.pipeline import VideoSummarizationPipeline
from app.services.task_store import TaskRepository

# ── Bootstrap ──────────────────────────────────────────────
setup_logging(settings.LOG_LEVEL)
logger = get_logger("api")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage database connection pool lifecycle."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")
    yield
    logger.info("Shutting down database...")
    await dispose_db()
    logger.info("Database connection closed")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Upload a video and receive a concise summarized version with TTS narration.",
    lifespan=lifespan,
)

task_store = TaskRepository()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)


# ── Background worker ─────────────────────────────────────

async def _run_pipeline(
    task_id: str,
    input_path: str,
    output_path: str,
    options: SummarizationOptions,
) -> None:
    """Async background task that runs the video summarization pipeline.

    DB updates are native async.  The blocking pipeline.process() call is
    offloaded to a thread via ``asyncio.to_thread`` so the event loop stays
    responsive.
    """
    try:
        await task_store.update(task_id, status=TaskStatusEnum.PROCESSING)
        logger.info("Pipeline started for task %s", task_id)

        transcriber = ComponentFactory.create_transcriber(options.transcriber_env)
        summarizer = ComponentFactory.create_summarizer(options.summarizer_env)
        tts = ComponentFactory.create_tts(options.tts_env, language=options.language)

        pipeline = VideoSummarizationPipeline(transcriber, summarizer, tts)

        # Run the blocking, CPU-heavy pipeline in a worker thread
        await asyncio.to_thread(
            pipeline.process, input_path, output_path, task_id,
        )

        await task_store.update(
            task_id,
            status=TaskStatusEnum.COMPLETED,
            output_url=f"/api/v1/tasks/{task_id}/download",
        )
        logger.info("Task %s completed", task_id)

    except Exception as exc:
        logger.exception("Task %s failed: %s", task_id, exc)
        await task_store.update(
            task_id,
            status=TaskStatusEnum.FAILED,
            error=str(exc),
        )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
            logger.debug("Removed upload: %s", input_path)


# ── Routes ─────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health-check endpoint with database connectivity status."""
    db_ok = False
    try:
        from sqlalchemy import text

        from app.db.session import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        logger.warning("Health check: database unreachable")

    return {
        "status": "healthy" if db_ok else "degraded",
        "version": settings.VERSION,
        "database": "connected" if db_ok else "disconnected",
    }


@app.post(
    "/api/v1/summarize",
    response_model=TaskResponse,
    status_code=202,
    tags=["Summarization"],
)
async def upload_and_summarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcriber_env: str = Query("local", description="Transcription backend"),
    summarizer_env: str = Query("local", description="Summarization backend"),
    tts_env: str = Query("local", description="TTS backend: 'local' or 'cloud'"),
    language: str = Query("vi", description="Language code for TTS"),
):
    """Upload a video and start the summarization pipeline."""

    # ── Validate extension ────────────────────────────────
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid format '{ext}'. "
                f"Allowed: {settings.allowed_extensions_list}"
            ),
        )

    # ── Validate content type ─────────────────────────────
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {content_type}",
        )

    # ── Persist upload (streaming, with size guard) ───────
    task_id = str(uuid.uuid4())
    safe_name = f"{task_id}{ext}"
    input_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    output_path = os.path.join(settings.OUTPUT_DIR, f"{task_id}_summary.mp4")

    total_bytes = 0
    try:
        with open(input_path, "wb") as buf:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > settings.max_file_size_bytes:
                    buf.close()
                    os.remove(input_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit",
                    )
                buf.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to save upload: {exc}",
        ) from exc

    logger.info(
        "Upload received: %s (%.1f MB) -> task %s",
        filename,
        total_bytes / (1024 * 1024),
        task_id,
    )

    # ── Enqueue background pipeline ───────────────────────
    options = SummarizationOptions(
        transcriber_env=transcriber_env,
        summarizer_env=summarizer_env,
        tts_env=tts_env,
        language=language,
    )
    await task_store.create(task_id)
    background_tasks.add_task(_run_pipeline, task_id, input_path, output_path, options)

    return TaskResponse(task_id=task_id, status=TaskStatusEnum.QUEUED)


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
)
async def get_task_status(task_id: str):
    """Check the status of a summarization task."""
    task = await task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskResponse(task_id=task_id, **task)


@app.get("/api/v1/tasks/{task_id}/download", tags=["Tasks"])
async def download_result(task_id: str):
    """Download the summarized video for a completed task."""
    task = await task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatusEnum.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Task is not completed (status: {task['status'].value})",
        )

    output_path = os.path.join(settings.OUTPUT_DIR, f"{task_id}_summary.mp4")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"{task_id}_summary.mp4",
    )