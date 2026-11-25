"""Clip discovery and naming on synthetic folder layouts."""

from pathlib import Path

import pytest

from raabta_pose_extraction.clip_discovery import ClipDiscoveryError, compute_number_width, name_clips, scan_dataset_folder


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
