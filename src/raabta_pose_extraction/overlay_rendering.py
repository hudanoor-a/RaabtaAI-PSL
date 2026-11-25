"""Side-by-side review images: [reference skeleton, or the plain frame | new extraction].

What is drawn (BGR colours): upper body (pose 0-24 connections) white, left-hand slot green,
right-hand slot orange, a hand recovered by MediaPipe Hands red, the 468 face-mesh points as small
grey dots and the 22 face points the training schema uses (mouth, eyes, eyebrows, forehead,
chin) as larger yellow dots. Panels are enlarged so the short side is at least 480 px.

A reviewer checks, per image: the skeleton sits on the signer (not on a poster or another person),
both hands are on the hands and on the correct side, and the face points sit on the mouth, eyes
and brows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from raabta_pose_extraction.missing_hand_recovery import HAND_SOURCE_RECOVERED

UPPER_BODY_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                          (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
                          (16, 22), (18, 20), (11, 23), (12, 24), (23, 24))
HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))
SCHEMA_FACE_MESH_POINTS = (10, 152, 61, 291, 0, 17, 13, 14, 33, 133, 159, 145, 263, 362, 386, 374, 107, 105, 70, 336, 334, 300)
MINIMUM_PANEL_SHORT_SIDE_IN_PIXELS = 480
WHITE, GREEN, ORANGE, RED, GREY, YELLOW = (255, 255, 255), (0, 200, 0), (0, 140, 255), (0, 0, 255), (160, 160, 160), (0, 220, 255)


def enlarge_for_review(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2
    factor = max(1.0, MINIMUM_PANEL_SHORT_SIDE_IN_PIXELS / min(frame_bgr.shape[:2]))
    if factor == 1.0:
        return frame_bgr.copy()
    return cv2.resize(frame_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def draw_raw_row(image: np.ndarray, raw_row: np.ndarray, hand_source_row: np.ndarray | None) -> np.ndarray:
    """Draw one 1662-value row on an image (in place) and return it."""
    import cv2
    height, width = image.shape[:2]

    def pixel(point) -> tuple[int, int]:
        return int(round(point[0] * width)), int(round(point[1] * height))

    pose = raw_row[:132].reshape(33, 4)
    if np.any(pose):
        for first, second in UPPER_BODY_CONNECTIONS:
            cv2.line(image, pixel(pose[first]), pixel(pose[second]), WHITE, 2, cv2.LINE_AA)
    face = raw_row[258:].reshape(468, 3)
    if np.any(face):
        for point in face:
            cv2.circle(image, pixel(point), 1, GREY, -1, cv2.LINE_AA)
        for index in SCHEMA_FACE_MESH_POINTS:
            cv2.circle(image, pixel(face[index]), 3, YELLOW, -1, cv2.LINE_AA)
    for slot_index, first_value in enumerate((132, 195)):
        hand = raw_row[first_value:first_value + 63].reshape(21, 3)
        if not np.any(hand):
            continue
        recovered = hand_source_row is not None and hand_source_row[slot_index] == HAND_SOURCE_RECOVERED
        colour = RED if recovered else (GREEN if slot_index == 0 else ORANGE)
        for first, second in HAND_CONNECTIONS:
            cv2.line(image, pixel(hand[first]), pixel(hand[second]), colour, 2, cv2.LINE_AA)
    return image


def write_side_by_side_review_image(frame_bgr: np.ndarray, new_row: np.ndarray, new_hand_source_row: np.ndarray,
                                    reference_row: np.ndarray | None, caption: str, output_file: Path) -> Path:
    """[reference (or plain frame) | new extraction] with a caption bar; written as JPEG."""
    import cv2
    base = enlarge_for_review(frame_bgr)
    left = draw_raw_row(base.copy(), reference_row, None) if reference_row is not None else base.copy()
    right = draw_raw_row(base.copy(), new_row, new_hand_source_row)
    bar_height = 34
    canvas = np.zeros((base.shape[0] + bar_height, base.shape[1] * 2 + 8, 3), dtype=np.uint8)
    canvas[bar_height:, :base.shape[1]] = left
    canvas[bar_height:, base.shape[1] + 8:] = right
    left_title = "reference extraction" if reference_row is not None else "frame"
    cv2.putText(canvas, f"{left_title}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"new extraction  {caption}", (base.shape[1] + 16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output_file
