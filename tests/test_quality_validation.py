"""Clip quality, pose outliers, retry rule, reference comparison and dataset gates."""

import numpy as np

from raabta_pose_extraction.quality_validation import (
    aggregate_by_signer,
    check_clip_integrity,
    compare_with_reference,
    evaluate_dataset_gates,
    find_pose_outlier_frames,
    list_retry_reasons,
    measure_clip_quality,
)
from synthetic_hand_clips import make_clip

OUTLIER_SETTINGS = {"maximum_shoulder_midpoint_offset_in_shoulder_widths": 1.0, "shoulder_width_ratio_range": [0.6, 1.6]}
RETRY = {"minimum_pose_frame_percent": 90.0, "minimum_any_hand_frame_percent": 60.0, "maximum_pose_outlier_frame_percent": 2.0}
GATES = {"minimum_pose_frame_percent": 95.0, "maximum_pose_outlier_frame_percent": 1.0, "minimum_any_hand_frame_percent": 80.0,
         "minimum_face_frame_percent": 90.0, "face_covered_below_face_frame_percent": 20.0,
         "reference_any_hand_must_not_drop": True, "minimum_overall_any_hand_gain_in_points": 5.0,
         "maximum_median_wrist_offset_from_reference_in_shoulder_widths": 0.10,
         "minimum_frame_count_match_with_reference_percent": 99.0, "minimum_clips_with_reference_percent": 99.0,
         "maximum_unwritten_clip_percent": 1.0}


def with_face(clip):
    clip = clip.copy()
    clip[:, 258:] = 0.5
    return clip


def test_a_jump_to_another_body_is_an_outlier_and_triggers_a_retry():
    clip = with_face(make_clip(40))
    clip[30, 44:52] += np.array([0.5, 0, 0, 0, 0.5, 0, 0, 0], dtype=np.float32)   # shoulders 11 and 12 move 2.5 widths right
    outliers = find_pose_outlier_frames(clip, 1.0, (0.6, 1.6))
    assert outliers.tolist().count(True) == 1 and outliers[30]
    quality = measure_clip_quality(clip, None, OUTLIER_SETTINGS)
    assert quality["pose_outlier_frame_percent"] == 2.5 and list_retry_reasons(quality, RETRY) == ["pose_outliers_high"]


def test_quality_percentages_and_integrity():
    clip = with_face(make_clip(40, left_missing=set(range(20)), right_missing=set(range(20)), pose_missing={39}))
    quality = measure_clip_quality(clip, None, OUTLIER_SETTINGS)
    assert quality["pose_frame_percent"] == 97.5 and quality["any_hand_frame_percent"] == round(100 * 19 / 39, 2)
    assert "hand_frames_low" in list_retry_reasons(quality, RETRY)
    broken = clip.copy()
    broken[0, 0] = np.nan
    assert check_clip_integrity(broken) == ["NaN or infinite values"] and check_clip_integrity(np.zeros((3, 10))) != []


def test_gates_with_and_without_reference():
    good = with_face(make_clip(40))
    reference = with_face(make_clip(40, left_missing=set(range(10)), right_missing=set(range(10))))
    reports = [{"signer_folder_name": "signer00", "quality": measure_clip_quality(good, None, OUTLIER_SETTINGS),
                "reference_comparison": compare_with_reference(good, reference, OUTLIER_SETTINGS)}]
    summary = aggregate_by_signer(reports)
    assert summary["all"]["reference_any_hand_frame_percent"] == 75.0 and summary["all"]["any_hand_frame_percent"] == 100.0
    assert summary["all"]["median_wrist_offset_from_reference_in_shoulder_widths"] == 0.0
    assert evaluate_dataset_gates(summary, GATES, has_reference=True)["passed"]
    worse = with_face(make_clip(40, left_missing=set(range(30)), right_missing=set(range(30))))
    summary = aggregate_by_signer([{"signer_folder_name": "signer00", "quality": measure_clip_quality(worse, None, OUTLIER_SETTINGS)}])
    gates = evaluate_dataset_gates(summary, GATES, has_reference=False)
    assert not gates["passed"] and [g["gate"] for g in gates["results"] if not g["passed"]] == ["any_hand_frame_percent"]
    assert gates["signer_passed"] == {"signer00": False}


def test_a_face_covered_signer_is_reported_not_failed():
    clip = make_clip(40)   # no face values
    summary = aggregate_by_signer([{"signer_folder_name": "signer06", "quality": measure_clip_quality(clip, None, OUTLIER_SETTINGS)}])
    gates = evaluate_dataset_gates(summary, GATES, has_reference=False)
    assert gates["face_covered_signers"] == ["signer06"] and gates["passed"]


def test_one_failing_signer_does_not_fail_the_others():
    good, bad = with_face(make_clip(40)), with_face(make_clip(40, left_missing=set(range(30)), right_missing=set(range(30))))
    summary = aggregate_by_signer([{"signer_folder_name": "signer00", "quality": measure_clip_quality(good, None, OUTLIER_SETTINGS)},
                                   {"signer_folder_name": "signer01", "quality": measure_clip_quality(bad, None, OUTLIER_SETTINGS)}])
    gates = evaluate_dataset_gates(summary, GATES, has_reference=False)
    assert gates["signer_passed"] == {"signer00": True, "signer01": False} and not gates["passed"]


def test_reference_gates_catch_a_missing_reference_and_a_wrong_mapping():
    good = with_face(make_clip(40))
    no_reference = aggregate_by_signer([{"signer_folder_name": "signer03", "quality": measure_clip_quality(good, None, OUTLIER_SETTINGS)}])
    gates = evaluate_dataset_gates(no_reference, GATES, has_reference=True)
    assert [g["gate"] for g in gates["results"] if not g["passed"]] == ["clips_with_reference_percent", "frame_count_match_with_reference_percent"]
    shorter_reference = with_face(make_clip(35))     # another clip's poses: the frame count differs
    summary = aggregate_by_signer([{"signer_folder_name": "signer03", "quality": measure_clip_quality(good, None, OUTLIER_SETTINGS),
                                    "reference_comparison": compare_with_reference(good, shorter_reference, OUTLIER_SETTINGS)}])
    assert summary["signer03"]["frame_count_match_with_reference_percent"] == 0.0
    assert not evaluate_dataset_gates(summary, GATES, has_reference=True)["signer_passed"]["signer03"]


def test_a_partly_attached_reference_fails_the_signer():
    good = with_face(make_clip(40))
    compared = {"signer_folder_name": "signer04", "quality": measure_clip_quality(good, None, OUTLIER_SETTINGS),
                "reference_comparison": compare_with_reference(good, good, OUTLIER_SETTINGS)}
    uncompared = {"signer_folder_name": "signer04", "quality": measure_clip_quality(good, None, OUTLIER_SETTINGS)}
    gates = evaluate_dataset_gates(aggregate_by_signer([compared, uncompared]), GATES, has_reference=True)
    assert [g["gate"] for g in gates["results"] if not g["passed"] and g["blocking"]] == ["clips_with_reference_percent"]
    assert evaluate_dataset_gates(aggregate_by_signer([compared, compared]), GATES, has_reference=True)["signer_passed"]["signer04"]
