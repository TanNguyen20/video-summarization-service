import logging
import sys

_INITIALIZED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root application logger.

    Safe to call multiple times — only the first call takes effect.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger("app")
    root.setLevel(log_level)
    root.addHandler(handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'app' namespace."""
    return logging.getLogger(f"app.{name}")
