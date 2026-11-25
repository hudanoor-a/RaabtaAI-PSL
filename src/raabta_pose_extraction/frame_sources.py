"""Read the frames of one clip, in order, from a video file or a frame folder.

Videos are decoded with OpenCV, which applies the rotation stored by phones (orientation
metadata) by default, so portrait recordings come out upright. The frame rate reported by the
container and the frame size are recorded for every clip: PSL comes from several phones with
different rates and sizes, and the data pipeline needs both (frame rate for resampling, size to
turn normalised x, y into an aspect-correct scale).

Frame folders are read in natural order (2 before 10), the order the original Isharah extraction
used; their frame rate is unknown (None).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from raabta_pose_extraction.clip_discovery import is_image_file

NUMBER_PATTERN = re.compile(r"(\d+)")


@dataclass
class ClipFrameInformation:
    frames_per_second: float | None
    frame_width_in_pixels: int | None = None
    frame_height_in_pixels: int | None = None
    frames_read: int = 0


def build_natural_sort_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in NUMBER_PATTERN.split(name)]


def list_frame_files(frame_folder: Path) -> list[Path]:
    files = [path for path in Path(frame_folder).iterdir() if path.is_file() and is_image_file(path.name)]
    return sorted(files, key=lambda path: build_natural_sort_key(path.name))


def iterate_clip_frames(source_path: Path, source_kind: str, information: ClipFrameInformation) -> Iterator[np.ndarray]:
    """Yield BGR uint8 frames; fills `information` (size, frames read) as it goes."""
    import cv2
    if source_kind == "video":
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise OSError(f"Could not open video {source_path}.")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                information.frames_read += 1
                if information.frame_width_in_pixels is None:
                    information.frame_height_in_pixels, information.frame_width_in_pixels = frame.shape[:2]
                yield frame
        finally:
            capture.release()
    elif source_kind == "frame_folder":
        for frame_file in list_frame_files(source_path):
            frame = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"Could not read frame image {frame_file}.")
            information.frames_read += 1
            if information.frame_width_in_pixels is None:
                information.frame_height_in_pixels, information.frame_width_in_pixels = frame.shape[:2]
            yield frame
    else:
        raise ValueError(f"Unknown source kind {source_kind}.")


def read_frames_per_second(source_path: Path, source_kind: str) -> float | None:
    """The container's frame rate for a video (None when absent or zero); None for frame folders."""
    if source_kind != "video":
        return None
    import cv2
    capture = cv2.VideoCapture(str(source_path))
    try:
        rate = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return float(rate) if rate and rate > 0 else None
