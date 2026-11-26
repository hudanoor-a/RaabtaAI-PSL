"""Stand-ins for extract_clip_in_worker, importable by spawned worker processes (tests only)."""

import os
import time


def finish_quickly_or_hang(task: dict) -> dict:
    """Clip 00_0002 hangs (like a worker stuck inside MediaPipe); every other clip finishes at once."""
    if task["clip_identifier"] == "00_0002":
        time.sleep(600)
    return {"clip_identifier": task["clip_identifier"], "frames_read": 1}


def finish_quickly_or_crash(task: dict) -> dict:
    """Clip 00_0002 kills its worker process (like the system killing it for memory); the pool breaks."""
    if task["clip_identifier"] == "00_0002":
        os._exit(1)
    return {"clip_identifier": task["clip_identifier"], "frames_read": 1}
