"""Hand recovery logic on synthetic raw clips: slot convention, calibration, assignment, checks, coverage."""

import numpy as np
import pytest

from raabta_pose_extraction.missing_hand_recovery import (
    HAND_SOURCE_EMPTY,
    HAND_SOURCE_HOLISTIC,
    HAND_SOURCE_RECOVERED,
    HandRecoverySettings,
    calibrate_recovery_settings,
    decide_pose_wrist_index_for_slot,
    find_detected_hand_slots,
    list_frames_needing_hand_detection,
    measure_detector_agreement_with_holistic,
    measure_hand_slot_to_pose_wrist_convention,
    read_hand_slot,
    recover_missing_hands_in_clip,
    summarise_hand_coverage,
)

from synthetic_hand_clips import SHOULDER_WIDTH, make_clip, make_hand  # noqa: E402


def calibrated_settings() -> HandRecoverySettings:
    rng = np.random.default_rng(1)
    return calibrate_recovery_settings([make_clip(60, rng) for _ in range(5)])


def test_slot_convention_is_measured_and_a_swapped_convention_is_detected():
    clip = make_clip()
    convention = measure_hand_slot_to_pose_wrist_convention([clip])
    assert convention["fraction_of_slot_wrists_closer_to_pose_15"] == {"left_hand": 1.0, "right_hand": 0.0}
    assert decide_pose_wrist_index_for_slot(convention) == {"left_hand": 15, "right_hand": 16}
    swapped = clip.copy()
    swapped[:, 132:195], swapped[:, 195:258] = clip[:, 195:258], clip[:, 132:195]
    assert decide_pose_wrist_index_for_slot(measure_hand_slot_to_pose_wrist_convention([swapped])) == {"left_hand": 16, "right_hand": 15}
    mixed = {"fraction_of_slot_wrists_closer_to_pose_15": {"left_hand": 0.6, "right_hand": 0.4}, "frames_counted": {}}
    with pytest.raises(ValueError, match="consistently"):
        decide_pose_wrist_index_for_slot(mixed)


def test_calibration_thresholds_come_from_holistic_hands():
    settings = calibrated_settings()
    low, high = settings.hand_size_range_in_shoulder_widths
    assert low < 0.3 < high                                   # hand size 0.06 / 0.2 = 0.3 shoulder widths
    assert 0 < settings.maximum_wrist_distance_in_shoulder_widths < 0.2
    assert settings.calibration_statistics["holistic_hand_frames_used"] == 600
    with pytest.raises(ValueError, match="Too few"):
        calibrate_recovery_settings([make_clip(10)])
    assert HandRecoverySettings.from_dictionary(settings.to_dictionary()) == settings


def test_a_good_candidate_fills_only_the_empty_slot_and_everything_else_is_unchanged():
    settings = calibrated_settings()
    clip = make_clip(left_missing={10})
    pose_wrist = clip[10, :132].reshape(33, 4)[15, :2]
    right_hand = read_hand_slot(clip, "right_hand")[10]
    candidates = {10: np.stack([right_hand, make_hand(pose_wrist[0] + 0.004, pose_wrist[1])])}   # duplicate + the missing hand
    recovered, hand_source, counts = recover_missing_hands_in_clip(clip, candidates, settings)
    assert counts["recovered_left_hand"] == 1 and counts["rejected_duplicate_of_holistic_hand"] == 1
    assert hand_source[10].tolist() == [HAND_SOURCE_RECOVERED, HAND_SOURCE_HOLISTIC]
    assert np.allclose(read_hand_slot(recovered, "left_hand")[10], candidates[10][1])
    changed = np.argwhere(recovered != clip)
    assert set(changed[:, 0]) == {10} and changed[:, 1].min() >= 132 and changed[:, 1].max() < 195


def test_far_wrong_size_and_teleporting_candidates_are_rejected():
    settings = calibrated_settings()
    clip = make_clip(left_missing={10, 11, 12})
    wrist = clip[10, :132].reshape(33, 4)[15, :2]
    candidates = {
        10: make_hand(wrist[0] + 0.15, wrist[1])[None],                   # 0.75 shoulder widths away
        11: make_hand(wrist[0], wrist[1], size=0.3)[None],                # 1.5 shoulder widths long
    }
    _, hand_source, counts = recover_missing_hands_in_clip(clip, candidates, settings)
    assert hand_source[10, 0] == hand_source[11, 0] == HAND_SOURCE_EMPTY
    assert counts["rejected_too_far_from_pose_wrist"] == 1 and counts["rejected_implausible_hand_size"] == 1
    teleporting = HandRecoverySettings.from_dictionary({**settings.to_dictionary(),
                                                        "maximum_wrist_distance_in_shoulder_widths": 1.0,
                                                        "maximum_wrist_speed_in_shoulder_widths_per_frame": 0.05})
    _, hand_source, counts = recover_missing_hands_in_clip(clip, {12: make_hand(wrist[0] + 0.05, wrist[1])[None]}, teleporting)
    assert hand_source[12, 0] == HAND_SOURCE_EMPTY and counts["rejected_teleporting"] == 1


def test_frames_without_pose_are_never_filled_and_never_run():
    settings = calibrated_settings()
    clip = make_clip(left_missing={5}, pose_missing={6})
    assert list_frames_needing_hand_detection(clip).tolist() == [5]
    _, hand_source, _ = recover_missing_hands_in_clip(clip, {6: make_hand(0.65, 0.6)[None]}, settings)
    assert hand_source[6].tolist() == [HAND_SOURCE_EMPTY, HAND_SOURCE_EMPTY]


def test_coverage_summary_and_agreement_measure():
    settings = calibrated_settings()
    clip = make_clip(left_missing={3, 4})
    wrist = clip[3, :132].reshape(33, 4)[15, :2]
    _, hand_source, _ = recover_missing_hands_in_clip(clip, {3: make_hand(wrist[0], wrist[1])[None]}, settings)
    coverage = summarise_hand_coverage(hand_source, np.ones(len(clip), dtype=bool))
    assert (coverage["left_hand_before"], coverage["left_hand_after"]) == (38, 39)
    assert coverage["both_hands_after"] == 39 and coverage["any_hand_before"] == 40
    holistic_left = read_hand_slot(clip, "left_hand")[20]
    agreement = measure_detector_agreement_with_holistic(clip, 20, "left_hand", (holistic_left + 0.002)[None], settings)
    assert agreement["found"] and agreement["mean_landmark_distance_in_shoulder_widths"] == pytest.approx(0.002 * 2 ** 0.5 / SHOULDER_WIDTH, rel=1e-3)
    assert not measure_detector_agreement_with_holistic(clip, 20, "left_hand", np.zeros((0, 21, 3)), settings)["found"]


def test_shapes_are_checked():
    with pytest.raises(ValueError, match="1662"):
        recover_missing_hands_in_clip(np.zeros((5, 10), dtype=np.float32), {}, HandRecoverySettings())
    assert find_detected_hand_slots(make_clip(4)).all()
