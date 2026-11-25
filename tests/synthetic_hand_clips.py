"""Synthetic raw Isharah clips with hands at the pose wrists, shared by the hand-recovery tests."""

import numpy as np

SHOULDER_WIDTH = 0.2


def make_hand(wrist_x: float, wrist_y: float, size: float = 0.06) -> np.ndarray:
    """A (21, 3) hand whose wrist is at (wrist_x, wrist_y) and middle knuckle `size` above it."""
    hand = np.zeros((21, 3), dtype=np.float32)
    offsets = np.linspace(0.0, 1.0, 21)
    hand[:, 0] = wrist_x + 0.01 * offsets
    hand[:, 1] = wrist_y - size * offsets * 1.5
    hand[0, :2] = (wrist_x, wrist_y)
    hand[9, :2] = (wrist_x, wrist_y - size)
    return hand


def make_clip(frame_count: int = 40, rng=None, left_missing=(), right_missing=(), pose_missing=()) -> np.ndarray:
    """Synthetic raw clip: shoulders 0.2 apart, wrists moving slowly, Holistic hands at the pose wrists."""
    rng = rng or np.random.default_rng(0)
    clip = np.zeros((frame_count, 1662), dtype=np.float32)
    for frame in range(frame_count):
        if frame in pose_missing:
            continue
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 3] = 1.0
        pose[11, :2] = (0.6, 0.4)                      # left shoulder (image right)
        pose[12, :2] = (0.4, 0.4)
        pose[15, :2] = (0.65 + 0.002 * frame, 0.6)     # left wrist
        pose[16, :2] = (0.35 - 0.002 * frame, 0.6)
        clip[frame, :132] = pose.reshape(-1)
        jitter = rng.normal(0, 0.003, size=4)
        if frame not in left_missing:
            clip[frame, 132:195] = make_hand(pose[15, 0] + jitter[0], pose[15, 1] + jitter[1]).reshape(-1)
        if frame not in right_missing:
            clip[frame, 195:258] = make_hand(pose[16, 0] + jitter[2], pose[16, 1] + jitter[3]).reshape(-1)
    return clip


