"""Extract poses for one dataset (Isharah, ArSL or PSL) with validation gates.

Run in the pose-extraction environment (Python 3.12 venv, requirements.txt). Two commands:

  discover   list the clips, the naming pattern chosen, examples of the output names, and the frame
             rate and size of a few clips. Nothing is extracted. Run it first and check the names.
  extract    the full run (src/raabta_pose_extraction/extraction_pipeline.py): primary pass, hand
             recovery, per-clip checks, enhanced re-extraction of failing clips, reference
             comparison, gates per signer, review images; passing signers in <dataset>/, failing
             signers in <dataset>_FAILED_VALIDATION/ (exit code 0 only when every signer passed).

Examples:
  python scripts/extract_poses.py discover --dataset-name psl --input-folder /kaggle/input/.../psl-videos
  python scripts/extract_poses.py discover --dataset-name isharah --input-folder .../signer00-isharah1000 --input-folder .../signer01-isharah1000
  python scripts/extract_poses.py extract --dataset-name psl --input-folder ... --output-root /kaggle/working/pose_extraction
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT_FOLDER / "src"))

from raabta_pose_extraction.clip_discovery import find_folder_signer_conflicts, name_clips, scan_dataset_folders  # noqa: E402
from raabta_pose_extraction.configuration import load_extraction_configuration  # noqa: E402
from raabta_pose_extraction.extraction_pipeline import DATASET_NAMES, run_dataset_extraction  # noqa: E402

DEFAULT_CONFIGURATION_FILE = REPOSITORY_ROOT_FOLDER / "configs" / "pose_extraction.yaml"


def run_discover(arguments) -> int:
    from raabta_pose_extraction.frame_sources import ClipFrameInformation, iterate_clip_frames, read_frames_per_second
    configuration = load_extraction_configuration(arguments.config)
    named, pattern = name_clips(scan_dataset_folders([Path(folder) for folder in arguments.input_folder]),
                                arguments.clip_name_pattern, configuration["naming"][arguments.dataset_name])
    signers = Counter(clip.signer_folder_name for clip in named)
    print(f"{len(named)} clips ({named[0].source_kind}), naming pattern: {pattern}")
    print("clips per signer:", dict(sorted(signers.items())))
    print("examples (source -> output):")
    for clip in named[:: max(1, len(named) // 12)][:12]:
        print(f"  {clip.relative_name}  ->  {arguments.dataset_name}/{clip.signer_folder_name}/{clip.clip_identifier}/{clip.clip_identifier}.npy")
    conflicts = find_folder_signer_conflicts(named)
    if conflicts:
        print("WARNING, folder names and clip names disagree on the signer (clip names are used):")
        for conflict in conflicts:
            print("  " + conflict)
    if arguments.signers:
        missing = sorted(set(arguments.signers) - {clip.signer_number for clip in named})
        if missing:
            print(f"ERROR: selected signers {missing} have no clips in the input folders.")
            return 1
    print("frame rate and size of the first clip of each signer:")
    seen = set()
    for clip in named:
        if clip.signer_folder_name in seen or (arguments.signers and clip.signer_number not in arguments.signers):
            continue
        seen.add(clip.signer_folder_name)
        try:
            information = ClipFrameInformation(read_frames_per_second(clip.source_path, clip.source_kind))
            frames = sum(1 for _ in iterate_clip_frames(clip.source_path, clip.source_kind, information))
        except (OSError, ValueError) as error:   # one unreadable clip is reported here and skipped by extract
            print(f"  {clip.signer_folder_name} {clip.clip_identifier}: UNREADABLE ({error})")
            continue
        print(f"  {clip.signer_folder_name} {clip.clip_identifier}: {frames} frames, "
              f"{information.frame_width_in_pixels}x{information.frame_height_in_pixels}, fps {information.frames_per_second}")
    return 0


def run_extract(arguments) -> int:
    configuration = load_extraction_configuration(arguments.config)
    summary = run_dataset_extraction(
        dataset_name=arguments.dataset_name, input_folders=[Path(folder) for folder in arguments.input_folder],
        output_root=Path(arguments.output_root),
        work_folder=Path(arguments.work_folder), configuration=configuration, clip_name_pattern=arguments.clip_name_pattern,
        signers=arguments.signers, clips_per_signer=arguments.clips_per_signer,
        reference_pose_folders=[Path(folder) for folder in arguments.reference_pose_folder] or None,
        calibration_file=Path(arguments.calibration_file) if arguments.calibration_file else None,
        worker_processes=arguments.worker_processes)
    printable = {key: summary[key] for key in ("dataset_name", "clips_selected", "clips_written",
                                               "clips_re_extracted_with_enhanced_settings", "clips_kept_from_enhanced_settings",
                                               "status_counts", "passed_signers", "failed_signers", "passed_folder",
                                               "failed_folder")}
    printable["failed_gates"] = [gate for gate in summary["gates"]["results"] if not gate["passed"] and gate["blocking"]]
    printable["informational_gates"] = [gate for gate in summary["gates"]["results"] if not gate["blocking"]]
    printable["face_covered_signers"] = summary["gates"]["face_covered_signers"]
    printable["all_signers"] = summary["signers"].get("all")
    print(json.dumps(printable, indent=1))
    return 0 if summary["passed"] else 2


def parse_arguments(arguments=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("discover", "extract"))
    parser.add_argument("--dataset-name", required=True, choices=DATASET_NAMES)
    parser.add_argument("--input-folder", required=True, action="append",
                        help="A folder holding the videos or frame folders; repeat it when a dataset is split over several folders.")
    parser.add_argument("--clip-name-pattern", help="Regex with (?P<signer>...) and (?P<sentence>...) [and (?P<repetition>...)]; "
                                                     "only when discover cannot find one.")
    parser.add_argument("--output-root", default="/kaggle/working/pose_extraction")
    parser.add_argument("--work-folder", default="/tmp/pose_extraction_work", help="Intermediate files; not part of the output.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIGURATION_FILE))
    parser.add_argument("--signers", type=int, nargs="+", help="Only these signer numbers (to split a large dataset over sessions).")
    parser.add_argument("--clips-per-signer", type=int, help="Pilot: a seeded random sample of this many clips per signer.")
    parser.add_argument("--reference-pose-folder", action="append", default=[],
                        help="Folder(s) with an earlier extraction, files named <clip_identifier>.npy (Isharah originals).")
    parser.add_argument("--calibration-file", help="Reuse the hand-recovery calibration of an earlier session of this dataset.")
    parser.add_argument("--worker-processes", type=int)
    return parser.parse_args(arguments)


def main(arguments=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    parsed = parse_arguments(arguments)
    return {"discover": run_discover, "extract": run_extract}[parsed.command](parsed)


if __name__ == "__main__":
    sys.exit(main())
