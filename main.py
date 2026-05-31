"""Application entry point.

Usage:
    python main.py
    uvicorn app.api.routes:app --reload
"""

import uvicorn

from app.core.config import settings


def main() -> None:
    uvicorn.run(
        "app.api.routes:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
