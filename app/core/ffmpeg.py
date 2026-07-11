"""Locate the ffmpeg executable for processes launched with a stale PATH.

Windows keeps a process's environment frozen at launch time, so an ffmpeg
installed later (e.g. via winget) stays invisible to already-running
terminals and IDEs until every parent process is restarted. Re-reading
PATH from the registry lets the service resolve ffmpeg regardless of how
it was launched.
"""

import os
import shutil
import sys

from app.core.logging import get_logger

logger = get_logger("core.ffmpeg")


def ensure_ffmpeg_on_path() -> None:
    """Make ``ffmpeg`` resolvable, refreshing PATH from the registry on Windows."""
    if shutil.which("ffmpeg"):
        return

    if sys.platform == "win32":
        import winreg

        registry_paths: list[str] = []
        for hive, key in (
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
            (winreg.HKEY_CURRENT_USER, "Environment"),
        ):
            try:
                with winreg.OpenKey(hive, key) as reg_key:
                    value, _ = winreg.QueryValueEx(reg_key, "Path")
                    registry_paths.append(os.path.expandvars(value))
            except OSError:
                continue

        os.environ["PATH"] = os.pathsep.join(
            [os.environ.get("PATH", ""), *registry_paths]
        )

    found = shutil.which("ffmpeg")
    if found:
        logger.info("ffmpeg resolved: %s", found)
    else:
        logger.warning(
            "ffmpeg executable not found — audio extraction will fail. "
            "Install FFmpeg (e.g. `winget install Gyan.FFmpeg`) and restart."
        )
