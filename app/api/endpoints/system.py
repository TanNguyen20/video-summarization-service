"""System endpoints (health check, etc.)."""

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import engine

logger = get_logger("api.system")
router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check():
    """Health-check endpoint with database connectivity status."""
    db_ok = False
    try:
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
