"""FastAPI application factory, lifespan, and shared state."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.session import dispose_db, init_db
from app.services.task_store import TaskRepository

# ── Bootstrap ──────────────────────────────────────────────
setup_logging(settings.LOG_LEVEL)
logger = get_logger("api")

# Shared task repository (used by endpoint routers)
task_store = TaskRepository()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage database connection pool lifecycle."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    yield
    logger.info("Shutting down database...")
    await dispose_db()
    logger.info("Database connection closed")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Upload a video and receive a concise summarized version with TTS narration.",
        lifespan=lifespan,
    )

    # Register routers
    from app.api.endpoints.system import router as system_router
    from app.api.endpoints.summarization import router as summarization_router
    from app.api.endpoints.tasks import router as tasks_router

    application.include_router(system_router)
    application.include_router(summarization_router)
    application.include_router(tasks_router)

    return application


app = create_app()
