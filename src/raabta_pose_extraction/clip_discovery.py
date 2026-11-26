"""Find every clip under a dataset folder and give it a signer number and a clip identifier.

A clip is either a VIDEO file (.mp4, .mov, .avi, .mkv, .webm, .m4v) or a FRAME FOLDER (a folder
that directly holds image files). Which kind a dataset uses is detected; mixing both is an error.

A dataset may be split over several input folders (Isharah on Kaggle: one dataset per signer,
for example /kaggle/input/datasets/<owner>/signer02-isharah1000). Every clip's name therefore
starts with the name of the input folder it was found in ("signer02-isharah1000/0001"), so the
signer number can come from either the folder name or the path inside it.

Naming. The signer and sentence numbers are read from each clip's path relative to the dataset
folder, with one regular expression that has the named groups `signer` and `sentence` (and,
optionally, `repetition`). Nothing is guessed silently: unless a pattern is given, the candidate
patterns below are tried in order and a pattern is accepted only if it parses EVERY clip and gives
every clip a different identity. Otherwise discovery stops and lists the paths that did not fit,
so the right pattern can be passed explicitly.

Identifiers keep the zero padding the data already uses, never less than 2 digits:
    signer folder  signer<SS>            e.g. signer00
    clip           <SS>_<CCCC>[_<RR>]    e.g. 00_0001, or 03_12_2 with a repetition
The width of each number = max(2, longest token as written, digits of the largest value).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_FILE_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MINIMUM_IMAGES_FOR_FRAME_FOLDER = 2
CANDIDATE_CLIP_NAME_PATTERNS = (
    # .../00_0001 or .../00_0001_2 (file or folder named signer_sentence[_repetition])
    r"(?:^|/)(?P<signer>\d+)_(?P<sentence>\d+)(?:_(?P<repetition>\d+))?$",
    # .../00/0001 (signer folder, then sentence file or folder)
    r"(?:^|/)(?P<signer>\d+)/(?P<sentence>\d+)$",
    # .../signer3/.../12 or .../Signer_03/.../sentence12
    r"(?i)(?:^|/)signer[ _-]?(?P<signer>\d+)/(?:.*/)?(?:sentence[ _-]?)?(?P<sentence>\d+)$",
    # .../00/0001/2 (signer, sentence, repetition)
    r"(?:^|/)(?P<signer>\d+)/(?P<sentence>\d+)/(?P<repetition>\d+)$",
    # signer02-isharah1000/.../0001 (signer number inside a longer folder name, e.g. a Kaggle dataset)
    r"(?i)(?:^|/)signer[ _-]?(?P<signer>\d+)[^/]*/(?:.*/)?(?:sentence[ _-]?)?(?P<sentence>\d+)$",
)
FOLDER_SIGNER_PATTERN = re.compile(r"(?i)signer[ _-]?(\d+)")


class ClipDiscoveryError(RuntimeError):
    """The dataset layout or the clip names could not be read without guessing."""


@dataclass(frozen=True)
class DiscoveredClip:
    source_path: Path          # video file or frame folder
    source_kind: str           # "video" | "frame_folder"
    relative_name: str         # <input folder name>/<path inside it>, without extension, "/" separated


@dataclass(frozen=True)
class NamedClip:
    source_path: Path
    source_kind: str
    relative_name: str
    signer_number: int
    sentence_number: int
    repetition_number: int | None
    signer_folder_name: str    # signer00
    clip_identifier: str       # 00_0001


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_FILE_EXTENSIONS)


def is_video_file(name: str) -> bool:
    return name.lower().endswith(VIDEO_FILE_EXTENSIONS)


def scan_dataset_folders(dataset_folders: list[Path]) -> list[DiscoveredClip]:
    """All clips of one dataset spread over one or more input folders (names prefixed by the folder name)."""
    if not dataset_folders:
        raise ClipDiscoveryError("No input folder given.")
    names = [Path(folder).name for folder in dataset_folders]
    if len(set(names)) != len(names):
        raise ClipDiscoveryError(f"Two input folders have the same name: {sorted(names)}.")
    clips = [clip for folder in dataset_folders for clip in scan_dataset_folder(Path(folder))]
    if len({clip.source_kind for clip in clips}) > 1:
        raise ClipDiscoveryError("The input folders mix videos and frame folders; one dataset must use one kind.")
    return sorted(clips, key=lambda clip: clip.relative_name)


def scan_dataset_folder(dataset_folder: Path) -> list[DiscoveredClip]:
    """Video files and frame folders under one input folder. Frame folders are not descended into."""
    dataset_folder = Path(dataset_folder)
    if not dataset_folder.is_dir():
        raise ClipDiscoveryError(f"{dataset_folder} is not a folder (check the input folder setting).")
    clips: list[DiscoveredClip] = []
    pending = [dataset_folder]
    while pending:
        folder = pending.pop()
        subfolders, image_count = [], 0
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=True):
                    subfolders.append(Path(entry.path))
                elif is_video_file(entry.name):
                    relative = (Path(dataset_folder.name) / Path(entry.path).relative_to(dataset_folder).with_suffix("")).as_posix()
                    clips.append(DiscoveredClip(Path(entry.path), "video", relative))
                elif is_image_file(entry.name):
                    image_count += 1
                    if image_count >= MINIMUM_IMAGES_FOR_FRAME_FOLDER and not subfolders:
                        break   # a frame folder: stop listing its (possibly hundreds of) frames
        if image_count >= MINIMUM_IMAGES_FOR_FRAME_FOLDER:
            clips.append(DiscoveredClip(folder, "frame_folder", (Path(dataset_folder.name) / folder.relative_to(dataset_folder)).as_posix()))
            continue
        pending.extend(subfolders)
    kinds = {clip.source_kind for clip in clips}
    if not clips:
        raise ClipDiscoveryError(f"No videos or frame folders under {dataset_folder}.")
    if len(kinds) > 1:
        raise ClipDiscoveryError(f"{dataset_folder} mixes videos and frame folders; point at one of them.")
    return sorted(clips, key=lambda clip: clip.relative_name)


def try_clip_name_pattern(clips: list[DiscoveredClip], pattern: str) -> tuple[list[dict] | None, list[str]]:
    """Parse every clip with the pattern. Returns (parsed values, relative names that did not match)."""
    compiled = re.compile(pattern)
    parsed, unmatched = [], []
    for clip in clips:
        match = compiled.search(clip.relative_name)
        if match is None or match.groupdict().get("signer") is None or match.groupdict().get("sentence") is None:
            unmatched.append(clip.relative_name)
            continue
        parsed.append({"signer": match.group("signer"), "sentence": match.group("sentence"),
                       "repetition": match.groupdict().get("repetition")})
    return (parsed if not unmatched else None), unmatched


def compute_number_width(tokens: list[str], minimum_width: int = 2) -> int:
    """max(minimum_width, longest token as written, digits of the largest value)."""
    return max([minimum_width] + [len(token) for token in tokens] + [len(str(max(int(token) for token in tokens)))])


def find_folder_signer_conflicts(named: list["NamedClip"]) -> list[str]:
    """Input folders whose name says 'signer<N>' while the clip names inside parse to another signer."""
    conflicts = set()
    for clip in named:
        folder_name = clip.relative_name.split("/", 1)[0]
        match = FOLDER_SIGNER_PATTERN.search(folder_name)
        if match and int(match.group(1)) != clip.signer_number:
            conflicts.add(f"{folder_name}: folder name says signer {int(match.group(1))}, clip names say signer {clip.signer_number}")
    return sorted(conflicts)


def name_clips(clips: list[DiscoveredClip], clip_name_pattern: str | None = None,
               minimum_widths: dict | None = None) -> tuple[list[NamedClip], str]:
    """Signer and clip identifier for every clip, plus the pattern used. Raises when ambiguous.

    minimum_widths {"signer": 2, "sentence": 4} fixes the padding a dataset is known to use (Isharah:
    NN_NNNN), so a run over only some signers names clips exactly like a run over all of them.
    """
    minimum_widths = minimum_widths or {}
    patterns = [clip_name_pattern] if clip_name_pattern else list(CANDIDATE_CLIP_NAME_PATTERNS)
    reports = []
    for pattern in patterns:
        parsed, unmatched = try_clip_name_pattern(clips, pattern)
        if parsed is None:
            reports.append(f"  {pattern}: {len(unmatched)} of {len(clips)} paths do not match, e.g. {unmatched[:3]}")
            continue
        signer_width = compute_number_width([values["signer"] for values in parsed], minimum_widths.get("signer", 2))
        sentence_width = compute_number_width([values["sentence"] for values in parsed], minimum_widths.get("sentence", 2))
        repetitions = [values["repetition"] for values in parsed if values["repetition"] is not None]
        if repetitions and len(repetitions) != len(parsed):
            reports.append(f"  {pattern}: only some clips have a repetition number")
            continue
        repetition_width = compute_number_width(repetitions, minimum_widths.get("repetition", 2)) if repetitions else 0
        named = []
        for clip, values in zip(clips, parsed):
            signer, sentence = int(values["signer"]), int(values["sentence"])
            repetition = int(values["repetition"]) if values["repetition"] is not None else None
            identifier = f"{signer:0{signer_width}d}_{sentence:0{sentence_width}d}"
            if repetition is not None:
                identifier += f"_{repetition:0{repetition_width}d}"
            named.append(NamedClip(clip.source_path, clip.source_kind, clip.relative_name, signer, sentence, repetition,
                                   f"signer{signer:0{signer_width}d}", identifier))
        identifiers = [clip.clip_identifier for clip in named]
        duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
        if duplicates:
            reports.append(f"  {pattern}: {len(duplicates)} identifiers are not unique, e.g. {duplicates[:3]}")
            continue
        return named, pattern
    raise ClipDiscoveryError("No clip-name pattern reads every clip with a unique identity:\n" + "\n".join(reports)
                             + "\nPass --clip-name-pattern with named groups (?P<signer>...) and (?P<sentence>...).")
