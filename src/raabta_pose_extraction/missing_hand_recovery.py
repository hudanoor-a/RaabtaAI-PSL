"""Recover hands that MediaPipe Holistic missed, using MediaPipe Hands, without touching anything else.

Pipeline position: used by extraction_pipeline.py after Holistic has run on every clip. It takes
a clip's Holistic rows (T, 1662) and the candidate hands MediaPipe Hands found on the frames where
a hand slot is empty, and returns new rows in the SAME layout. Pose, face and every hand Holistic
found stay byte-identical; only empty hand slots can be filled. numpy only (no MediaPipe import),
so it is tested without MediaPipe. Same code as RaabtaAI (src/pose_extraction/missing_hand_recovery.py).

Raw layout per frame (unchanged): pose 33 x (x, y, z, visibility) = values 0-131,
left-hand slot 21 x (x, y, z) = 132-194, right-hand slot 21 x (x, y, z) = 195-257,
face 468 x (x, y, z) = 258-1661. An empty part is all zeros.

Why a candidate hand is trusted (all distances in shoulder widths of that frame, x and y only):
  1. slot assignment: its wrist is close to the pose wrist of the empty slot
     (maximum_wrist_distance_in_shoulder_widths);
  2. not a duplicate: it is not the hand Holistic already found in the other slot;
  3. plausible size: wrist to middle-finger knuckle length within hand_size_range_in_shoulder_widths;
  4. no teleporting: its wrist moves at most maximum_wrist_speed_in_shoulder_widths_per_frame
     per frame from the nearest Holistic hand of the same slot within the neighbour window.
All thresholds come from calibrate_recovery_settings(), measured on hands Holistic DID find, so
they describe this dataset, not guesses. Which pose wrist belongs to which hand slot is also
measured (measure_hand_slot_to_pose_wrist_convention) instead of assumed.

hand_source per frame and slot: 0 = empty, 1 = found by Holistic (original), 2 = recovered here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

RAW_VALUES_PER_FRAME = 1662
POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
HAND_SLOT_NAMES = ("left_hand", "right_hand")
HAND_SLOT_FIRST_VALUE = {"left_hand": 132, "right_hand": 195}
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
HAND_WRIST, HAND_MIDDLE_FINGER_KNUCKLE = 0, 9

HAND_SOURCE_EMPTY, HAND_SOURCE_HOLISTIC, HAND_SOURCE_RECOVERED = 0, 1, 2
MINIMUM_SHOULDER_WIDTH = 1e-3
DUPLICATE_HAND_MAXIMUM_WRIST_DISTANCE_IN_SHOULDER_WIDTHS = 0.15
POSE_WRIST_MAXIMUM_DISTANCE_OUTSIDE_FRAME = 0.1   # normalised image units; a wrist further outside has no visible hand


@dataclass
class HandRecoverySettings:
    """Acceptance thresholds for a recovered hand. Produced by calibrate_recovery_settings()."""

    maximum_wrist_distance_in_shoulder_widths: float = 0.5
    hand_size_range_in_shoulder_widths: tuple[float, float] = (0.1, 0.8)
    maximum_wrist_speed_in_shoulder_widths_per_frame: float = 0.3
    neighbour_window_in_frames: int = 10
    pose_wrist_index_for_slot: dict = field(default_factory=lambda: {"left_hand": POSE_LEFT_WRIST, "right_hand": POSE_RIGHT_WRIST})
    calibration_statistics: dict = field(default_factory=dict)

    def to_dictionary(self) -> dict:
        values = asdict(self)
        values["hand_size_range_in_shoulder_widths"] = list(self.hand_size_range_in_shoulder_widths)
        return values

    @classmethod
    def from_dictionary(cls, values: dict) -> "HandRecoverySettings":
        values = dict(values)
        values["hand_size_range_in_shoulder_widths"] = tuple(values["hand_size_range_in_shoulder_widths"])
        return cls(**values)


# ---- reading the raw layout ---------------------------------------------------------------

def check_raw_clip_shape(raw_clip: np.ndarray) -> None:
    if raw_clip.ndim != 2 or raw_clip.shape[1] != RAW_VALUES_PER_FRAME:
        raise ValueError(f"Expected a (T, {RAW_VALUES_PER_FRAME}) raw clip, got shape {raw_clip.shape}.")


def read_pose_landmarks(raw_clip: np.ndarray) -> np.ndarray:
    """(T, 1662) -> (T, 33, 4): x, y, z, visibility."""
    return raw_clip[:, :POSE_LANDMARK_COUNT * 4].reshape(-1, POSE_LANDMARK_COUNT, 4)


def read_hand_slot(raw_clip: np.ndarray, slot_name: str) -> np.ndarray:
    """(T, 1662) -> (T, 21, 3) for one hand slot."""
    first = HAND_SLOT_FIRST_VALUE[slot_name]
    return raw_clip[:, first:first + HAND_LANDMARK_COUNT * 3].reshape(-1, HAND_LANDMARK_COUNT, 3)


def find_frames_with_pose(raw_clip: np.ndarray) -> np.ndarray:
    """(T,) bool: Holistic found a pose (the pose block is not all zeros)."""
    return np.any(raw_clip[:, :POSE_LANDMARK_COUNT * 4] != 0, axis=1)


def find_detected_hand_slots(raw_clip: np.ndarray) -> np.ndarray:
    """(T, 2) bool: slot 0 = left_hand, slot 1 = right_hand; True when the slot is not all zeros."""
    return np.stack([np.any(read_hand_slot(raw_clip, name) != 0, axis=(1, 2)) for name in HAND_SLOT_NAMES], axis=1)


def compute_shoulder_widths(raw_clip: np.ndarray) -> np.ndarray:
    """(T,) shoulder width in normalised image units (x, y), NaN where there is no pose."""
    pose = read_pose_landmarks(raw_clip)
    widths = np.linalg.norm(pose[:, POSE_LEFT_SHOULDER, :2] - pose[:, POSE_RIGHT_SHOULDER, :2], axis=1)
    widths = widths.astype(np.float64)
    widths[~find_frames_with_pose(raw_clip) | (widths < MINIMUM_SHOULDER_WIDTH)] = np.nan
    return widths


def measure_hand_size_in_shoulder_widths(hand: np.ndarray, shoulder_width: float) -> float:
    """Wrist to middle-finger knuckle length of one (21, 3) hand, in shoulder widths."""
    return float(np.linalg.norm(hand[HAND_MIDDLE_FINGER_KNUCKLE, :2] - hand[HAND_WRIST, :2]) / shoulder_width)


def measure_mean_landmark_distance_in_shoulder_widths(first_hand: np.ndarray, second_hand: np.ndarray, shoulder_width: float) -> float:
    """Mean x, y distance over the 21 landmarks of two (21, 3) hands, in shoulder widths."""
    return float(np.linalg.norm(first_hand[:, :2] - second_hand[:, :2], axis=1).mean() / shoulder_width)


# ---- calibration on hands Holistic found ---------------------------------------------------

def measure_hand_slot_to_pose_wrist_convention(raw_clips: list[np.ndarray]) -> dict:
    """For Holistic hands, how often each slot's wrist is closer to pose 15 than to pose 16."""
    closer_to_pose_15 = {name: 0 for name in HAND_SLOT_NAMES}
    counted = {name: 0 for name in HAND_SLOT_NAMES}
    for raw_clip in raw_clips:
        pose = read_pose_landmarks(raw_clip)
        detected = find_detected_hand_slots(raw_clip) & find_frames_with_pose(raw_clip)[:, None]
        for slot_index, name in enumerate(HAND_SLOT_NAMES):
            frames = np.flatnonzero(detected[:, slot_index])
            if frames.size == 0:
                continue
            wrists = read_hand_slot(raw_clip, name)[frames, HAND_WRIST, :2]
            to_15 = np.linalg.norm(wrists - pose[frames, POSE_LEFT_WRIST, :2], axis=1)
            to_16 = np.linalg.norm(wrists - pose[frames, POSE_RIGHT_WRIST, :2], axis=1)
            closer_to_pose_15[name] += int((to_15 < to_16).sum())
            counted[name] += int(frames.size)
    fractions = {name: (closer_to_pose_15[name] / counted[name] if counted[name] else None) for name in HAND_SLOT_NAMES}
    return {"fraction_of_slot_wrists_closer_to_pose_15": fractions, "frames_counted": counted}


def decide_pose_wrist_index_for_slot(convention: dict, minimum_agreement: float = 0.9) -> dict:
    """Map each slot to pose 15 or 16 from the measured convention; stop if the data is ambiguous."""
    fractions = convention["fraction_of_slot_wrists_closer_to_pose_15"]
    left, right = fractions["left_hand"], fractions["right_hand"]
    if left is None or right is None:
        raise ValueError(f"Cannot measure the slot convention, a slot has no Holistic hands: {convention}")
    if left >= minimum_agreement and right <= 1 - minimum_agreement:
        return {"left_hand": POSE_LEFT_WRIST, "right_hand": POSE_RIGHT_WRIST}
    if left <= 1 - minimum_agreement and right >= minimum_agreement:
        return {"left_hand": POSE_RIGHT_WRIST, "right_hand": POSE_LEFT_WRIST}
    raise ValueError(f"Hand slots do not follow one pose wrist consistently (needed {minimum_agreement}): {convention}")


def calibrate_recovery_settings(raw_clips: list[np.ndarray], neighbour_window_in_frames: int = 10,
                                distance_margin: float = 1.25, size_margin: float = 0.2, speed_margin: float = 1.5) -> HandRecoverySettings:
    """Thresholds from Holistic hands: 99th percentile wrist offset and speed, 1st-99th percentile size, plus margins."""
    convention = measure_hand_slot_to_pose_wrist_convention(raw_clips)
    pose_wrist_index_for_slot = decide_pose_wrist_index_for_slot(convention)
    wrist_offsets, hand_sizes, wrist_speeds = [], [], []
    for raw_clip in raw_clips:
        pose = read_pose_landmarks(raw_clip)
        widths = compute_shoulder_widths(raw_clip)
        detected = find_detected_hand_slots(raw_clip)
        for slot_index, name in enumerate(HAND_SLOT_NAMES):
            hands = read_hand_slot(raw_clip, name)
            usable = detected[:, slot_index] & ~np.isnan(widths)
            frames = np.flatnonzero(usable)
            if frames.size == 0:
                continue
            wrists = hands[frames, HAND_WRIST, :2]
            pose_wrists = pose[frames, pose_wrist_index_for_slot[name], :2]
            wrist_offsets.extend(np.linalg.norm(wrists - pose_wrists, axis=1) / widths[frames])
            hand_sizes.extend(np.linalg.norm(hands[frames, HAND_MIDDLE_FINGER_KNUCKLE, :2] - wrists, axis=1) / widths[frames])
            consecutive = frames[1:][np.diff(frames) == 1]
            if consecutive.size:
                steps = np.linalg.norm(hands[consecutive, HAND_WRIST, :2] - hands[consecutive - 1, HAND_WRIST, :2], axis=1)
                wrist_speeds.extend(steps / widths[consecutive])
    if min(len(wrist_offsets), len(hand_sizes), len(wrist_speeds)) < 100:
        raise ValueError("Too few Holistic hands to calibrate (need at least 100 of each measurement); use more clips.")
    offsets, sizes, speeds = np.asarray(wrist_offsets), np.asarray(hand_sizes), np.asarray(wrist_speeds)
    statistics = {
        "holistic_hand_frames_used": int(offsets.size),
        "slot_convention": convention,
        "wrist_to_pose_wrist_in_shoulder_widths_percentiles_50_95_99": np.percentile(offsets, [50, 95, 99]).round(4).tolist(),
        "hand_size_in_shoulder_widths_percentiles_1_50_99": np.percentile(sizes, [1, 50, 99]).round(4).tolist(),
        "wrist_speed_in_shoulder_widths_per_frame_percentiles_50_95_99": np.percentile(speeds, [50, 95, 99]).round(4).tolist(),
    }
    return HandRecoverySettings(
        maximum_wrist_distance_in_shoulder_widths=float(np.percentile(offsets, 99) * distance_margin),
        hand_size_range_in_shoulder_widths=(float(np.percentile(sizes, 1) * (1 - size_margin)),
                                            float(np.percentile(sizes, 99) * (1 + size_margin))),
        maximum_wrist_speed_in_shoulder_widths_per_frame=float(np.percentile(speeds, 99) * speed_margin),
        neighbour_window_in_frames=neighbour_window_in_frames,
        pose_wrist_index_for_slot=pose_wrist_index_for_slot,
        calibration_statistics=statistics,
    )


# ---- recovery ----------------------------------------------------------------------------------

def assign_candidate_hands_to_empty_slots(candidate_hands: np.ndarray, pose_frame: np.ndarray, shoulder_width: float,
                                          holistic_hands: dict, settings: HandRecoverySettings) -> tuple[dict, list[str]]:
    """Choose which candidate (if any) fills each EMPTY slot of one frame.

    candidate_hands (N, 21, 3); pose_frame (33, 4); holistic_hands {slot name: (21, 3) or None}.
    Returns ({slot name: candidate index}, rejection reasons for candidates that were not used).
    """
    empty_slots = [name for name in HAND_SLOT_NAMES if holistic_hands.get(name) is None]
    reasons: list[str] = []
    if not empty_slots or len(candidate_hands) == 0:
        return {}, reasons
    usable = []
    for candidate_index, hand in enumerate(candidate_hands):
        wrist = hand[HAND_WRIST, :2]
        if any(holistic is not None and np.linalg.norm(wrist - holistic[HAND_WRIST, :2]) / shoulder_width
               <= DUPLICATE_HAND_MAXIMUM_WRIST_DISTANCE_IN_SHOULDER_WIDTHS for holistic in holistic_hands.values()):
            reasons.append("duplicate_of_holistic_hand")
            continue
        size = measure_hand_size_in_shoulder_widths(hand, shoulder_width)
        low, high = settings.hand_size_range_in_shoulder_widths
        if not low <= size <= high:
            reasons.append("implausible_hand_size")
            continue
        usable.append(candidate_index)
    pairs = []
    for candidate_index in usable:
        for name in empty_slots:
            pose_wrist = pose_frame[settings.pose_wrist_index_for_slot[name], :2]
            if np.any(pose_wrist < -POSE_WRIST_MAXIMUM_DISTANCE_OUTSIDE_FRAME) or np.any(pose_wrist > 1 + POSE_WRIST_MAXIMUM_DISTANCE_OUTSIDE_FRAME):
                continue
            distance = np.linalg.norm(candidate_hands[candidate_index][HAND_WRIST, :2] - pose_wrist) / shoulder_width
            pairs.append((distance, candidate_index, name))
    assignment: dict = {}
    used_candidates: set = set()
    for distance, candidate_index, name in sorted(pairs):
        if name in assignment or candidate_index in used_candidates:
            continue
        if distance > settings.maximum_wrist_distance_in_shoulder_widths:
            continue
        assignment[name] = candidate_index
        used_candidates.add(candidate_index)
    reasons.extend("too_far_from_pose_wrist" for candidate_index in usable if candidate_index not in used_candidates)
    return assignment, reasons


def find_nearest_holistic_wrist(holistic_detected: np.ndarray, hands: np.ndarray, frame_index: int, window: int):
    """Wrist (x, y) and frame distance of the nearest Holistic hand of this slot within the window, or (None, None)."""
    for offset in range(1, window + 1):
        for neighbour in (frame_index - offset, frame_index + offset):
            if 0 <= neighbour < len(holistic_detected) and holistic_detected[neighbour]:
                return hands[neighbour, HAND_WRIST, :2], offset
    return None, None


def recover_missing_hands_in_clip(raw_clip: np.ndarray, candidate_hands_per_frame: dict,
                                  settings: HandRecoverySettings) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fill empty hand slots from candidate hands.

    candidate_hands_per_frame: {frame index: (N, 21, 3) float array} for the frames that were run
    through the hand detector (coordinates normalised to the original frame, like Holistic's).
    Returns (new raw clip, hand_source (T, 2) uint8, counts of recovered hands and rejection reasons).
    """
    check_raw_clip_shape(raw_clip)
    recovered_clip = raw_clip.copy()
    holistic_detected = find_detected_hand_slots(raw_clip)
    hand_source = np.where(holistic_detected, HAND_SOURCE_HOLISTIC, HAND_SOURCE_EMPTY).astype(np.uint8)
    pose = read_pose_landmarks(raw_clip)
    widths = compute_shoulder_widths(raw_clip)
    original_hands = {name: read_hand_slot(raw_clip, name) for name in HAND_SLOT_NAMES}
    counts = {"recovered_left_hand": 0, "recovered_right_hand": 0, "rejected_teleporting": 0}
    for frame_index, candidates in sorted(candidate_hands_per_frame.items()):
        if np.isnan(widths[frame_index]) or candidates is None or len(candidates) == 0:
            continue
        holistic_hands = {name: (original_hands[name][frame_index] if holistic_detected[frame_index, slot_index] else None)
                          for slot_index, name in enumerate(HAND_SLOT_NAMES)}
        assignment, reasons = assign_candidate_hands_to_empty_slots(np.asarray(candidates, dtype=np.float32), pose[frame_index],
                                                                    float(widths[frame_index]), holistic_hands, settings)
        for reason in reasons:
            counts[f"rejected_{reason}"] = counts.get(f"rejected_{reason}", 0) + 1
        for name, candidate_index in assignment.items():
            slot_index = HAND_SLOT_NAMES.index(name)
            hand = np.asarray(candidates[candidate_index], dtype=np.float32)
            neighbour_wrist, frame_gap = find_nearest_holistic_wrist(holistic_detected[:, slot_index], original_hands[name],
                                                                     frame_index, settings.neighbour_window_in_frames)
            if neighbour_wrist is not None:
                speed = np.linalg.norm(hand[HAND_WRIST, :2] - neighbour_wrist) / widths[frame_index] / frame_gap
                if speed > settings.maximum_wrist_speed_in_shoulder_widths_per_frame:
                    counts["rejected_teleporting"] += 1
                    continue
            first = HAND_SLOT_FIRST_VALUE[name]
            recovered_clip[frame_index, first:first + HAND_LANDMARK_COUNT * 3] = hand.reshape(-1)
            hand_source[frame_index, slot_index] = HAND_SOURCE_RECOVERED
            counts[f"recovered_{name}"] += 1
    return recovered_clip, hand_source, counts


def list_frames_needing_hand_detection(raw_clip: np.ndarray) -> np.ndarray:
    """Frame indices with a pose and at least one empty hand slot (the only frames the detector runs on)."""
    return np.flatnonzero(find_frames_with_pose(raw_clip) & ~find_detected_hand_slots(raw_clip).all(axis=1))


def summarise_hand_coverage(hand_source: np.ndarray, frames_with_pose: np.ndarray) -> dict:
    """Frame counts with each slot filled, before (Holistic only) and after recovery, over frames with a pose."""
    before = (hand_source == HAND_SOURCE_HOLISTIC)[frames_with_pose]
    after = (hand_source != HAND_SOURCE_EMPTY)[frames_with_pose]
    return {
        "frames_with_pose": int(frames_with_pose.sum()),
        "left_hand_before": int(before[:, 0].sum()), "left_hand_after": int(after[:, 0].sum()),
        "right_hand_before": int(before[:, 1].sum()), "right_hand_after": int(after[:, 1].sum()),
        "any_hand_before": int(before.any(axis=1).sum()), "any_hand_after": int(after.any(axis=1).sum()),
        "both_hands_before": int(before.all(axis=1).sum()), "both_hands_after": int(after.all(axis=1).sum()),
    }


def measure_detector_agreement_with_holistic(raw_clip: np.ndarray, frame_index: int, slot_name: str,
                                             candidate_hands: np.ndarray, settings: HandRecoverySettings) -> dict:
    """Accuracy proxy on a frame where Holistic FOUND this slot: pretend it is empty and see what recovery picks.

    Returns found (a candidate was assigned to the slot), and the mean landmark distance to the
    Holistic hand in shoulder widths when found.
    """
    width = float(compute_shoulder_widths(raw_clip[frame_index:frame_index + 1])[0])
    if np.isnan(width):
        return {"found": False, "mean_landmark_distance_in_shoulder_widths": None}
    slot_index = HAND_SLOT_NAMES.index(slot_name)
    detected = find_detected_hand_slots(raw_clip[frame_index:frame_index + 1])[0]
    holistic_hands = {name: (read_hand_slot(raw_clip, name)[frame_index] if detected[index] and name != slot_name else None)
                      for index, name in enumerate(HAND_SLOT_NAMES)}
    assignment, _ = assign_candidate_hands_to_empty_slots(np.asarray(candidate_hands, dtype=np.float32),
                                                          read_pose_landmarks(raw_clip)[frame_index], width, holistic_hands, settings)
    if slot_name not in assignment:
        return {"found": False, "mean_landmark_distance_in_shoulder_widths": None}
    holistic_hand = read_hand_slot(raw_clip, slot_name)[frame_index]
    distance = measure_mean_landmark_distance_in_shoulder_widths(np.asarray(candidate_hands[assignment[slot_name]]), holistic_hand, width)
    return {"found": True, "mean_landmark_distance_in_shoulder_widths": distance, "slot_index": slot_index}
