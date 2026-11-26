"""Quality measures for an extracted clip, the per-clip retry rule, and the dataset gates.

All percentages of hands and face are over frames WITH a pose (a frame without a pose cannot
hold hands or a face in MediaPipe Holistic); the pose percentage is over all frames.

Pose outliers catch a pose that jumped to another person, a poster or a mirror: a frame is an
outlier when its shoulder midpoint is far from the clip's median shoulder midpoint, or its
shoulder width is far from the clip's median width (thresholds in configs/pose_extraction.yaml).

With a reference extraction (the original Isharah poses), the new extraction is compared side by
side, clip by clip: same frame count, detection rates before and after, and how far the new
wrists sit from the reference wrists on frames where both found a pose.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from raabta_pose_extraction.missing_hand_recovery import (
    HAND_SOURCE_EMPTY,
    HAND_SOURCE_RECOVERED,
    RAW_VALUES_PER_FRAME,
    compute_shoulder_widths,
    find_detected_hand_slots,
    find_frames_with_pose,
    read_pose_landmarks,
)

FACE_FIRST_VALUE = 258
POSE_SHOULDERS = (11, 12)
POSE_WRISTS = (15, 16)


def check_clip_integrity(raw_clip: np.ndarray) -> list[str]:
    """Problems that make a clip unusable: wrong shape, NaN or infinity, coordinates far outside the frame."""
    problems = []
    if raw_clip.ndim != 2 or raw_clip.shape[1] != RAW_VALUES_PER_FRAME or raw_clip.shape[0] == 0:
        return [f"shape {raw_clip.shape} is not (T > 0, {RAW_VALUES_PER_FRAME})"]
    if not np.isfinite(raw_clip).all():
        problems.append("NaN or infinite values")
    pose = read_pose_landmarks(raw_clip)[find_frames_with_pose(raw_clip)]
    if pose.size and (np.abs(pose[:, :25, :2]).max() > 3.0):
        problems.append("upper-body coordinates more than 3 frame sizes outside the frame")
    return problems


def find_pose_outlier_frames(raw_clip: np.ndarray, maximum_offset_in_shoulder_widths: float,
                             shoulder_width_ratio_range: tuple[float, float]) -> np.ndarray:
    """(T,) bool: frames whose pose is far from the clip's typical body position or size."""
    widths = compute_shoulder_widths(raw_clip)
    valid = ~np.isnan(widths)
    outliers = np.zeros(raw_clip.shape[0], dtype=bool)
    if valid.sum() < 3:
        return outliers
    pose = read_pose_landmarks(raw_clip)
    midpoints = pose[:, list(POSE_SHOULDERS), :2].mean(axis=1)
    median_midpoint = np.median(midpoints[valid], axis=0)
    median_width = float(np.median(widths[valid]))
    offsets = np.linalg.norm(midpoints - median_midpoint, axis=1) / median_width
    ratios = widths / median_width
    low, high = shoulder_width_ratio_range
    outliers[valid] = (offsets[valid] > maximum_offset_in_shoulder_widths) | (ratios[valid] < low) | (ratios[valid] > high)
    return outliers


def measure_clip_quality(raw_clip: np.ndarray, hand_source: np.ndarray | None, outlier_settings: dict) -> dict:
    """Frame counts and percentages for one clip."""
    frame_count = int(raw_clip.shape[0])
    with_pose = find_frames_with_pose(raw_clip)
    pose_frames = int(with_pose.sum())
    hands = (hand_source != HAND_SOURCE_EMPTY) if hand_source is not None else find_detected_hand_slots(raw_clip)
    hands = hands & with_pose[:, None]
    face = np.any(raw_clip[:, FACE_FIRST_VALUE:] != 0, axis=1) & with_pose
    outliers = find_pose_outlier_frames(raw_clip, outlier_settings["maximum_shoulder_midpoint_offset_in_shoulder_widths"],
                                        tuple(outlier_settings["shoulder_width_ratio_range"]))
    denominator = max(pose_frames, 1)
    return {
        "frame_count": frame_count,
        "pose_frames": pose_frames,
        "left_hand_frames": int(hands[:, 0].sum()),
        "right_hand_frames": int(hands[:, 1].sum()),
        "any_hand_frames": int(hands.any(axis=1).sum()),
        "both_hands_frames": int(hands.all(axis=1).sum()),
        "face_frames": int(face.sum()),
        "pose_outlier_frames": int(outliers.sum()),
        "recovered_hands": int((hand_source == HAND_SOURCE_RECOVERED).sum()) if hand_source is not None else 0,
        "pose_frame_percent": round(100 * pose_frames / max(frame_count, 1), 2),
        "any_hand_frame_percent": round(100 * hands.any(axis=1).sum() / denominator, 2),
        "both_hands_frame_percent": round(100 * hands.all(axis=1).sum() / denominator, 2),
        "face_frame_percent": round(100 * face.sum() / denominator, 2),
        "pose_outlier_frame_percent": round(100 * outliers.sum() / denominator, 2),
    }


def list_retry_reasons(quality: dict, retry_settings: dict) -> list[str]:
    """Why a clip should be re-extracted with the enhanced settings (empty list = keep it)."""
    reasons = []
    if quality["pose_frame_percent"] < retry_settings["minimum_pose_frame_percent"]:
        reasons.append("pose_frames_low")
    if quality["any_hand_frame_percent"] < retry_settings["minimum_any_hand_frame_percent"]:
        reasons.append("hand_frames_low")
    if quality["pose_outlier_frame_percent"] > retry_settings["maximum_pose_outlier_frame_percent"]:
        reasons.append("pose_outliers_high")
    return reasons


def score_extraction(quality: dict) -> float:
    """Higher is better; used to keep the better of the primary and enhanced extraction of a clip."""
    return quality["pose_frame_percent"] + quality["any_hand_frame_percent"] - 5 * quality["pose_outlier_frame_percent"]


def compare_with_reference(new_clip: np.ndarray, reference_clip: np.ndarray, outlier_settings: dict) -> dict:
    """Side-by-side numbers for one clip against its reference extraction."""
    reference_quality = measure_clip_quality(reference_clip, None, outlier_settings)
    comparison = {f"reference_{key}": value for key, value in reference_quality.items() if key.endswith(("percent", "frames"))}
    comparison["reference_frame_count"] = int(reference_clip.shape[0])
    comparison["frame_count_matches_reference"] = bool(reference_clip.shape[0] == new_clip.shape[0])
    if comparison["frame_count_matches_reference"]:
        both = find_frames_with_pose(new_clip) & find_frames_with_pose(reference_clip)
        widths = compute_shoulder_widths(reference_clip)
        usable = both & ~np.isnan(widths)
        if usable.any():
            new_pose, reference_pose = read_pose_landmarks(new_clip), read_pose_landmarks(reference_clip)
            offsets = np.linalg.norm(new_pose[usable][:, list(POSE_WRISTS), :2] - reference_pose[usable][:, list(POSE_WRISTS), :2],
                                     axis=2) / widths[usable][:, None]
            comparison["median_wrist_offset_from_reference_in_shoulder_widths"] = round(float(np.median(offsets)), 4)
    return comparison


def aggregate_by_signer(clip_reports: list[dict]) -> dict:
    """Frame-weighted percentages per signer (and 'all')."""
    totals: dict = defaultdict(lambda: defaultdict(float))
    offsets: dict = defaultdict(list)
    for report in clip_reports:
        quality = report.get("quality")
        if not quality:
            continue
        for group in (report["signer_folder_name"], "all"):
            for key, value in quality.items():
                if key.endswith("_frames") or key in ("frame_count", "recovered_hands"):
                    totals[group][key] += value
            comparison = report.get("reference_comparison") or {}
            for key in ("reference_pose_frames", "reference_any_hand_frames", "reference_face_frames"):
                if key in comparison:
                    totals[group][key] += comparison[key]
            if comparison:
                totals[group]["reference_clips"] += 1
                totals[group]["reference_frame_count_matches"] += int(comparison.get("frame_count_matches_reference", False))
            if comparison.get("median_wrist_offset_from_reference_in_shoulder_widths") is not None:
                offsets[group].append(comparison["median_wrist_offset_from_reference_in_shoulder_widths"])
            totals[group]["clips"] += 1
    summary = {}
    for group, values in totals.items():
        pose, frames = values["pose_frames"], max(values["frame_count"], 1)
        entry = {
            "clips": int(values["clips"]),
            "frames": int(values["frame_count"]),
            "pose_frame_percent": round(100 * pose / frames, 2),
            "any_hand_frame_percent": round(100 * values["any_hand_frames"] / max(pose, 1), 2),
            "both_hands_frame_percent": round(100 * values["both_hands_frames"] / max(pose, 1), 2),
            "face_frame_percent": round(100 * values["face_frames"] / max(pose, 1), 2),
            "pose_outlier_frame_percent": round(100 * values["pose_outlier_frames"] / max(pose, 1), 2),
            "recovered_hands": int(values["recovered_hands"]),
        }
        if values.get("reference_clips"):
            entry["reference_clips"] = int(values["reference_clips"])
            entry["frame_count_match_with_reference_percent"] = round(100 * values["reference_frame_count_matches"]
                                                                      / values["reference_clips"], 2)
        if values.get("reference_pose_frames"):
            reference_pose = values["reference_pose_frames"]
            entry["reference_pose_frame_percent"] = round(100 * reference_pose / frames, 2)
            entry["reference_any_hand_frame_percent"] = round(100 * values["reference_any_hand_frames"] / max(reference_pose, 1), 2)
            entry["reference_face_frame_percent"] = round(100 * values["reference_face_frames"] / max(reference_pose, 1), 2)
        if offsets[group]:
            entry["median_wrist_offset_from_reference_in_shoulder_widths"] = round(float(np.median(offsets[group])), 4)
        summary[group] = entry
    return summary


def evaluate_dataset_gates(signer_summary: dict, gates: dict, has_reference: bool) -> dict:
    """Every gate per signer; a signer passes only when all of ITS gates pass (thresholds fixed in the config).

    The overall any-hand gain over the reference is reported for the whole dataset but blocks no signer
    (a signer already near 95% cannot gain 5 points); per signer, coverage must not drop below the reference.
    """
    results = []

    def record(gate: str, group: str, value, threshold, passed: bool, blocking: bool = True):
        results.append({"gate": gate, "group": group, "value": value, "threshold": threshold, "passed": bool(passed),
                        "blocking": blocking})

    face_covered = []
    signers = sorted(group for group in signer_summary if group != "all")
    for group in signers:
        entry = signer_summary[group]
        record("pose_frame_percent", group, entry["pose_frame_percent"], f">= {gates['minimum_pose_frame_percent']}",
               entry["pose_frame_percent"] >= gates["minimum_pose_frame_percent"])
        record("pose_outlier_frame_percent", group, entry["pose_outlier_frame_percent"],
               f"<= {gates['maximum_pose_outlier_frame_percent']}",
               entry["pose_outlier_frame_percent"] <= gates["maximum_pose_outlier_frame_percent"])
        if entry["face_frame_percent"] < gates["face_covered_below_face_frame_percent"]:
            face_covered.append(group)
        else:
            record("face_frame_percent", group, entry["face_frame_percent"], f">= {gates['minimum_face_frame_percent']}",
                   entry["face_frame_percent"] >= gates["minimum_face_frame_percent"])
        if has_reference:
            # Every clip must be compared with its original: a missing (or partly attached) reference dataset
            # would otherwise let a signer pass on the few clips that happen to have one.
            covered = round(100 * entry.get("reference_clips", 0) / max(entry["clips"], 1), 2)
            limit = gates["minimum_clips_with_reference_percent"]
            record("clips_with_reference_percent", group, covered, f">= {limit}", covered >= limit)
            matched = entry.get("frame_count_match_with_reference_percent", 0.0)
            limit = gates["minimum_frame_count_match_with_reference_percent"]
            record("frame_count_match_with_reference_percent", group, matched, f">= {limit}", matched >= limit)
        if has_reference and "reference_any_hand_frame_percent" in entry:
            if gates["reference_any_hand_must_not_drop"]:
                record("any_hand_not_below_reference", group, entry["any_hand_frame_percent"],
                       f">= reference {entry['reference_any_hand_frame_percent']}",
                       entry["any_hand_frame_percent"] >= entry["reference_any_hand_frame_percent"])
            record("pose_not_below_reference", group, entry["pose_frame_percent"],
                   f">= reference {entry['reference_pose_frame_percent']}",
                   entry["pose_frame_percent"] >= entry["reference_pose_frame_percent"])
            offset = entry.get("median_wrist_offset_from_reference_in_shoulder_widths")
            if offset is not None:
                limit = gates["maximum_median_wrist_offset_from_reference_in_shoulder_widths"]
                record("median_wrist_offset_from_reference", group, offset, f"<= {limit}", offset <= limit)
        else:
            record("any_hand_frame_percent", group, entry["any_hand_frame_percent"], f">= {gates['minimum_any_hand_frame_percent']}",
                   entry["any_hand_frame_percent"] >= gates["minimum_any_hand_frame_percent"])
    overall = signer_summary.get("all", {})
    if has_reference and "reference_any_hand_frame_percent" in overall:
        gain = round(overall["any_hand_frame_percent"] - overall["reference_any_hand_frame_percent"], 2)
        record("overall_any_hand_gain_points", "all", gain, f">= {gates['minimum_overall_any_hand_gain_in_points']}",
               gain >= gates["minimum_overall_any_hand_gain_in_points"], blocking=False)
    signer_passed = {group: all(result["passed"] for result in results if result["group"] == group) for group in signers}
    return {"passed": bool(signers) and all(signer_passed.values()), "signer_passed": signer_passed, "results": results,
            "face_covered_signers": face_covered}
