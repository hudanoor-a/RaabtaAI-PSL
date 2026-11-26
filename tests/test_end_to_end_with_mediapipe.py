"""End to end with real MediaPipe on a small synthetic video dataset (skipped without mediapipe).

The video frames are a photo of a person (scikit-image's astronaut, stored in tests/data) shifted a
few pixels per frame. Model complexity 1 is used here because the heavy model (complexity 2) is
downloaded at first use, which needs internet; on Kaggle the shipped configuration is used as is.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mediapipe")
cv2 = pytest.importorskip("cv2")

from raabta_pose_extraction.configuration import load_extraction_configuration  # noqa: E402
from raabta_pose_extraction.extraction_pipeline import run_dataset_extraction  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
PERSON_IMAGE = REPOSITORY / "tests" / "data" / "person_upper_body.png"


def write_video(path: Path, image: np.ndarray, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (image.shape[1], image.shape[0]))
    for frame in range(frame_count):
        writer.write(np.roll(image, frame, axis=1))
    writer.release()


@pytest.fixture(scope="module")
def offline_configuration():
    configuration = copy.deepcopy(load_extraction_configuration(REPOSITORY / "configs" / "pose_extraction.yaml"))
    for name in ("primary", "enhanced"):
        configuration["holistic"][name]["model_complexity"] = 1
    configuration["runtime"]["worker_processes"] = 2
    configuration["review"]["clips_per_signer"] = 1
    configuration["review"]["frames_per_clip"] = 2
    return configuration


def test_videos_are_extracted_validated_and_written_in_the_agreed_layout(tmp_path, offline_configuration):
    image = cv2.imread(str(PERSON_IMAGE))
    for signer in (1, 2):
        for sentence in (1, 2):
            write_video(tmp_path / "psl" / f"Signer{signer}" / f"sentence{sentence}.mp4", image, 12)
    summary = run_dataset_extraction("psl", [tmp_path / "psl"], tmp_path / "out", tmp_path / "work", offline_configuration)
    # The photo shows no hands, so every signer fails the any-hand gate and goes to the failed folder.
    assert summary["clips_written"] == 4 and summary["failed_signers"] == ["signer01", "signer02"] and summary["passed_folder"] is None
    final = Path(summary["failed_folder"])
    failure = json.loads((final / "signer01" / "signer_failure.json").read_text())
    assert [gate["gate"] for gate in failure["failed_gates"]] == ["any_hand_frame_percent"] and failure["clips_not_written"] == []
    assert (final / "run_summaries" / "signers_01_02.json").exists()
    clip_folder = final / "signer01" / "01_01"
    rows = np.load(clip_folder / "01_01.npy")
    assert rows.shape == (12, 1662) and rows.dtype == np.float32
    assert np.any(rows[:, :132]) and np.any(rows[:, 258:])          # pose and face found on the person photo
    report = json.loads((clip_folder / "extraction_report.json").read_text())
    assert report["frames_per_second"] == 25.0 and report["frame_width_in_pixels"] == 512
    assert np.load(clip_folder / "hand_source.npy").shape == (12, 2)
    reviews = sorted(final.glob("signer*/*/pose*.jpg"))
    assert len(reviews) == 4 and all(len(path.stem) == len("pose0000") for path in reviews)
    assert (final / "extraction_summary.json").exists() and (final / "clip_reports.jsonl").exists()
    assert json.loads((final / "signer_outcomes.json").read_text())["failed_signers"] == ["signer01", "signer02"]


def test_frame_folders_with_a_reference_extraction(tmp_path, offline_configuration):
    image = cv2.resize(cv2.imread(str(PERSON_IMAGE)), (256, 256))
    for sentence in ("0001", "0002"):
        folder = tmp_path / "isharah" / "00" / sentence
        folder.mkdir(parents=True)
        for frame in range(10):
            cv2.imwrite(str(folder / f"{frame}.jpg"), np.roll(image, frame, axis=1))
    first = run_dataset_extraction("isharah", [tmp_path / "isharah"], tmp_path / "out_first", tmp_path / "work_first", offline_configuration)
    reference_folder = Path(first["passed_folder"] or first["failed_folder"])
    second = run_dataset_extraction("isharah", [tmp_path / "isharah"], tmp_path / "out_second", tmp_path / "work_second",
                                    offline_configuration, reference_pose_folders=[reference_folder])
    assert second["reference_clips_found"] == 2
    report = json.loads((Path(second["passed_folder"] or second["failed_folder"]) / "signer00" / "00_0001" / "extraction_report.json").read_text())
    assert report["reference_comparison"]["frame_count_matches_reference"]
    assert report["reference_comparison"]["median_wrist_offset_from_reference_in_shoulder_widths"] < 0.05


def test_a_second_run_keeps_the_signers_of_the_first(tmp_path, offline_configuration):
    image = cv2.imread(str(PERSON_IMAGE))
    for signer in (1, 2):
        write_video(tmp_path / "psl" / f"Signer{signer}" / "sentence1.mp4", image, 6)
    run_dataset_extraction("psl", [tmp_path / "psl"], tmp_path / "out", tmp_path / "work", offline_configuration, signers=[1])
    summary = run_dataset_extraction("psl", [tmp_path / "psl"], tmp_path / "out", tmp_path / "work2", offline_configuration, signers=[2])
    failed = Path(summary["failed_folder"])
    assert (failed / "signer01").is_dir() and (failed / "signer02").is_dir()
    assert json.loads((failed / "signer_outcomes.json").read_text())["failed_signers"] == ["signer01", "signer02"]
    with pytest.raises(RuntimeError, match="Selected signers \\[7\\]"):
        run_dataset_extraction("psl", [tmp_path / "psl"], tmp_path / "out", tmp_path / "work3", offline_configuration, signers=[7])
