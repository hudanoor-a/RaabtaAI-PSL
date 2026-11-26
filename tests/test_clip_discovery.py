"""Clip discovery and naming on synthetic folder layouts."""

from pathlib import Path

import pytest

from raabta_pose_extraction.clip_discovery import (
    ClipDiscoveryError,
    compute_number_width,
    find_folder_signer_conflicts,
    name_clips,
    scan_dataset_folder,
    scan_dataset_folders,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_isharah_style_frame_folders_keep_their_padding(tmp_path):
    for signer, sentence in (("00", "0001"), ("00", "0002"), ("17", "1000")):
        for frame in range(3):
            touch(tmp_path / "frames" / signer / sentence / f"{frame}.jpg")
    named, pattern = name_clips(scan_dataset_folder(tmp_path))
    assert [clip.clip_identifier for clip in named] == ["00_0001", "00_0002", "17_1000"]
    assert named[0].signer_folder_name == "signer00" and named[0].source_kind == "frame_folder"


def test_psl_style_videos_get_two_digit_minimum_and_repetitions(tmp_path):
    for signer in (1, 12):
        for sentence in (1, 65):
            touch(tmp_path / f"Signer{signer}" / f"sentence{sentence}.mp4")
    named, _ = name_clips(scan_dataset_folder(tmp_path))
    assert {clip.clip_identifier for clip in named} == {"01_01", "01_65", "12_01", "12_65"}
    assert {clip.signer_folder_name for clip in named} == {"signer01", "signer12"}
    (tmp_path / "Signer1" / "sentence1.mp4").unlink()
    touch(tmp_path / "videos" / "3_12_1.mp4")
    touch(tmp_path / "videos" / "3_12_2.mp4")
    named, _ = name_clips(scan_dataset_folder(tmp_path / "videos"))
    assert [clip.clip_identifier for clip in named] == ["03_12_01", "03_12_02"]


def test_unreadable_names_stop_with_the_unmatched_paths(tmp_path):
    touch(tmp_path / "a" / "hello.mp4")
    touch(tmp_path / "b" / "world.mp4")
    with pytest.raises(ClipDiscoveryError, match="No clip-name pattern"):
        name_clips(scan_dataset_folder(tmp_path))
    touch(tmp_path / "c" / "s5_t7.mp4")
    named, pattern = name_clips(scan_dataset_folder(tmp_path / "c"), r"s(?P<signer>\d+)_t(?P<sentence>\d+)$")
    assert named[0].clip_identifier == "05_07" and pattern.startswith("s(")


def test_duplicate_identities_and_mixed_sources_are_refused(tmp_path):
    touch(tmp_path / "x" / "01_02.mp4")
    touch(tmp_path / "y" / "01_02.mp4")
    with pytest.raises(ClipDiscoveryError, match="not unique"):
        name_clips(scan_dataset_folder(tmp_path))
    touch(tmp_path / "z" / "01_03" / "0.png")
    touch(tmp_path / "z" / "01_03" / "1.png")
    with pytest.raises(ClipDiscoveryError, match="mixes"):
        scan_dataset_folder(tmp_path)


def test_explicit_pattern_and_number_width():
    assert compute_number_width(["1", "65"]) == 2 and compute_number_width(["0001", "1000"]) == 4 and compute_number_width(["7", "123"]) == 3


def make_frames(folder: Path, count: int = 2) -> None:
    for frame in range(count):
        touch(folder / f"{frame}.jpg")


@pytest.mark.parametrize("inner", ["{sentence}", "{signer}/{sentence}", "frames/{signer}_{sentence}"])
def test_isharah_split_over_one_kaggle_dataset_per_signer(tmp_path, inner):
    """Kaggle layout: /kaggle/input/datasets/<owner>/signer02-isharah1000, signer0-isharah1000, ..."""
    folders = []
    for slug, signer in (("signer0-isharah1000", 0), ("signer02-isharah1000", 2), ("signer4-isharah1000", 4)):
        folder = tmp_path / "owner" / slug
        for sentence in (1, 2, 500):
            make_frames(folder / inner.format(signer=f"{signer:02d}", sentence=f"{sentence:04d}"))
        folders.append(folder)
    named, _ = name_clips(scan_dataset_folders(folders), None, {"signer": 2, "sentence": 4})
    assert sorted(clip.clip_identifier for clip in named) == ["00_0001", "00_0002", "00_0500", "02_0001", "02_0002", "02_0500",
                                                              "04_0001", "04_0002", "04_0500"]
    assert find_folder_signer_conflicts(named) == []


def test_minimum_widths_keep_isharah_identifiers_for_a_subset_of_signers(tmp_path):
    make_frames(tmp_path / "signer04-isharah1000" / "1")
    make_frames(tmp_path / "signer04-isharah1000" / "500")
    without, _ = name_clips(scan_dataset_folders([tmp_path / "signer04-isharah1000"]))
    fixed, _ = name_clips(scan_dataset_folders([tmp_path / "signer04-isharah1000"]), None, {"signer": 2, "sentence": 4})
    assert [clip.clip_identifier for clip in without] == ["04_001", "04_500"]
    assert [clip.clip_identifier for clip in fixed] == ["04_0001", "04_0500"]


def test_a_mislabelled_dataset_folder_is_reported(tmp_path):
    make_frames(tmp_path / "signer00-isharah1000" / "02" / "0001")      # the folder says 0, the clips say 2
    named, _ = name_clips(scan_dataset_folders([tmp_path / "signer00-isharah1000"]))
    assert named[0].clip_identifier == "02_0001"
    assert find_folder_signer_conflicts(named) == ["signer00-isharah1000: folder name says signer 0, clip names say signer 2"]


def test_folder_errors_are_clear(tmp_path):
    with pytest.raises(ClipDiscoveryError, match="is not a folder"):
        scan_dataset_folders([tmp_path / "missing"])
    with pytest.raises(ClipDiscoveryError, match="No input folder"):
        scan_dataset_folders([])
