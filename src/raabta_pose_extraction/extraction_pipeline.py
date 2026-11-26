"""The whole extraction run for one dataset: discover, extract, recover hands, validate, retry, write.

Steps (each clip is read from disk once per settings):
  1. discover clips and name them (clip_discovery.py); optionally keep only some signers, or a
     seeded sample of clips per signer (pilot);
  2. PRIMARY pass, in worker processes: Holistic on every frame; MediaPipe Hands on frames with a
     pose and an empty hand slot; a few raw frames kept for review images. Work files go to
     --work-folder (resumable: finished clips are skipped);
  3. calibrate the hand-recovery thresholds on the Holistic hands of this run (or load a given
     calibration file, so several sessions of one dataset share one calibration);
  4. recover missed hands and measure every clip; clips failing the per-clip checks are
     re-extracted with the ENHANCED settings and the better extraction is kept;
  5. compare with a reference extraction when one is given (Isharah original poses);
  6. decide per signer and write: a signer whose clips were all written and whose gates all pass
     goes to <output>/<dataset>/signer<SS>; any other signer goes to
     <output>/<dataset>_FAILED_VALIDATION/signer<SS> with signer_failure.json. <output>/<dataset> is
     therefore publishable as is, with no manual step: a missing signer there is a failed signer.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import random
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from raabta_pose_extraction.clip_discovery import NamedClip, find_folder_signer_conflicts, name_clips, scan_dataset_folders
from raabta_pose_extraction.missing_hand_recovery import (
    HandRecoverySettings,
    calibrate_recovery_settings,
    find_detected_hand_slots,
    recover_missing_hands_in_clip,
)
from raabta_pose_extraction.quality_validation import (
    aggregate_by_signer,
    check_clip_integrity,
    compare_with_reference,
    evaluate_dataset_gates,
    list_retry_reasons,
    measure_clip_quality,
    score_extraction,
)

LOGGER = logging.getLogger("raabta_pose_extraction")
DATASET_NAMES = ("isharah", "arsl", "psl")
SETTINGS_NAMES = ("primary", "enhanced")
CLIP_IDENTIFIER_FILE_PATTERN = re.compile(r"^\d+_\d+(?:_\d+)?$")

MAXIMUM_WORKER_POOLS_PER_PASS = 50   # a crashing clip costs about 1 + (workers) pools; 50 covers about 10 crashes

_worker_hand_detectors: dict = {}


# ---- one clip, in a worker process ---------------------------------------------------------------

def extract_clip_in_worker(task: dict) -> dict:
    """Holistic rows + hand candidates + review frames for one clip, written to the work folder."""
    from raabta_pose_extraction.frame_sources import ClipFrameInformation, iterate_clip_frames, read_frames_per_second
    from raabta_pose_extraction.mediapipe_models import HandDetector, HolisticExtractor
    started = time.time()
    work = Path(task["work_folder"])
    clip = task["clip_identifier"]
    settings = task["holistic_settings"]
    upscale = int(settings.get("upscale_short_side_below_pixels", 0))
    detector = None
    if task["hand_recovery_enabled"]:
        key = (json.dumps(task["hand_detector_settings"], sort_keys=True), upscale)
        if key not in _worker_hand_detectors:
            _worker_hand_detectors[key] = HandDetector(task["hand_detector_settings"], upscale).__enter__()
        detector = _worker_hand_detectors[key]
    information = ClipFrameInformation(frames_per_second=read_frames_per_second(Path(task["source_path"]), task["source_kind"]))
    rows, candidates = [], {}
    review_count = task["review_frames"]
    reservoir: list[tuple[int, np.ndarray]] = []
    rng = random.Random(f"review-{clip}")
    with HolisticExtractor(settings) as extractor:
        for index, frame in enumerate(iterate_clip_frames(Path(task["source_path"]), task["source_kind"], information)):
            row = extractor.extract(frame)
            rows.append(row)
            if detector is not None and np.any(row[:132]) and not (np.any(row[132:195]) and np.any(row[195:258])):
                candidates[str(index)] = detector.detect(frame)
            if review_count:
                if len(reservoir) < review_count:
                    reservoir.append((index, frame.copy()))
                else:
                    position = rng.randint(0, index)
                    if position < review_count:
                        reservoir[position] = (index, frame.copy())
    clip_folder = work / task["settings_name"]
    clip_folder.mkdir(parents=True, exist_ok=True)
    holistic_rows = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 1662), dtype=np.float32)
    np.save(clip_folder / f"{clip}.npy", holistic_rows)
    np.savez_compressed(clip_folder / f"{clip}_hand_candidates.npz", **candidates)
    if reservoir:
        import cv2
        review_folder = work / "review_frames" / clip
        review_folder.mkdir(parents=True, exist_ok=True)
        for index, frame in reservoir:
            cv2.imwrite(str(review_folder / f"{index:06d}.png"), frame)
    metadata = {"clip_identifier": clip, "settings_name": task["settings_name"], "frames_read": information.frames_read,
                "frames_per_second": information.frames_per_second, "frame_width_in_pixels": information.frame_width_in_pixels,
                "frame_height_in_pixels": information.frame_height_in_pixels, "frames_run_through_hand_detector": len(candidates),
                "seconds": round(time.time() - started, 2)}
    (clip_folder / f"{clip}.json").write_text(json.dumps(metadata), encoding="utf-8")   # written last = clip finished
    return metadata


# ---- run helpers ---------------------------------------------------------------------------------

def select_clips(clips: list[NamedClip], signers: list[int] | None, clips_per_signer: int | None, seed: int) -> list[NamedClip]:
    chosen = [clip for clip in clips if signers is None or clip.signer_number in signers]
    if clips_per_signer is None:
        return chosen
    by_signer: dict = {}
    for clip in chosen:
        by_signer.setdefault(clip.signer_number, []).append(clip)
    sampled = []
    for signer, signer_clips in sorted(by_signer.items()):
        count = min(clips_per_signer, len(signer_clips))
        sampled.extend(random.Random(f"{seed}-{signer}").sample(signer_clips, count))
    return sorted(sampled, key=lambda clip: clip.clip_identifier)


def select_review_clips(clips: list[NamedClip], clips_per_signer: int, seed: int) -> set[str]:
    by_signer: dict = {}
    for clip in clips:
        by_signer.setdefault(clip.signer_number, []).append(clip.clip_identifier)
    chosen = set()
    for signer, identifiers in by_signer.items():
        chosen.update(random.Random(f"review-{seed}-{signer}").sample(sorted(identifiers), min(clips_per_signer, len(identifiers))))
    return chosen


def index_reference_poses(reference_pose_folders: list[Path]) -> dict[str, Path]:
    """clip identifier -> reference .npy (files named <clip_identifier>.npy anywhere under the folders)."""
    index: dict[str, Path] = {}
    for folder in reference_pose_folders:
        for path in Path(folder).rglob("*.npy"):
            if CLIP_IDENTIFIER_FILE_PATTERN.match(path.stem) is None:   # skips hand_source.npy and other files
                continue
            if path.stem in index and index[path.stem] != path:
                raise RuntimeError(f"Reference clip {path.stem} found twice: {index[path.stem]} and {path}.")
            index[path.stem] = path
    return index


def run_extraction_pass(clips: list[NamedClip], settings_name: str, configuration: dict, work_folder: Path,
                        review_clips: set[str], worker_processes: int, worker_function=None) -> dict[str, dict]:
    """Extract every clip not yet finished in the work folder; returns metadata per clip (finished or failed).
    worker_function replaces extract_clip_in_worker in tests only."""
    worker_function = worker_function or extract_clip_in_worker
    pass_folder = work_folder / settings_name
    tasks, results = [], {}
    for clip in clips:
        metadata_file = pass_folder / f"{clip.clip_identifier}.json"
        if metadata_file.exists():
            results[clip.clip_identifier] = json.loads(metadata_file.read_text(encoding="utf-8"))
            continue
        tasks.append({"clip_identifier": clip.clip_identifier, "source_path": str(clip.source_path), "source_kind": clip.source_kind,
                      "settings_name": settings_name, "holistic_settings": configuration["holistic"][settings_name],
                      "hand_recovery_enabled": configuration["hand_recovery"]["enabled"],
                      "hand_detector_settings": configuration["hand_recovery"]["detector"],
                      "review_frames": configuration["review"]["frames_per_clip"] if clip.clip_identifier in review_clips and settings_name == "primary" else 0,
                      "work_folder": str(work_folder)})
    LOGGER.info("%s pass: %d clips, %d already done, %d to extract with %d workers.", settings_name, len(clips),
                len(clips) - len(tasks), len(tasks), worker_processes)
    if not tasks:
        return results
    timeout = configuration["runtime"]["clip_timeout_in_seconds"]
    started, cached = time.time(), len(results)   # cached: clips finished in an earlier run of this work folder
    suspects: list[str] = []        # clips running when a pool broke: each is re-run alone to find the culprit
    started_folder = work_folder / "started_clips"
    pools = 0
    while True:
        pending = [task for task in tasks if task["clip_identifier"] not in results]
        if not pending:
            break
        pools += 1
        if pools > MAXIMUM_WORKER_POOLS_PER_PASS:
            LOGGER.error("%s pass: %d worker pools used; %d clips left unextracted.", settings_name, pools - 1, len(pending))
            break
        isolating = [task for task in pending if task["clip_identifier"] in suspects][:1]
        batch, workers = (isolating, 1) if isolating else (pending, worker_processes)
        if pools > 1:
            LOGGER.warning("%s pass: worker pool %d for %d clips%s.", settings_name, pools, len(batch),
                           f" (re-running {isolating[0]['clip_identifier']} alone)" if isolating else "")
        shutil.rmtree(started_folder, ignore_errors=True)
        broken, timed_out = run_worker_pool(batch, workers, worker_function, timeout, results, settings_name,
                                            (started, cached, len(tasks)))
        unfinished_started = sorted(path.name for path in started_folder.glob("*")
                                    if path.name in {task["clip_identifier"] for task in batch} and path.name not in results)
        for clip in unfinished_started if timed_out else []:
            results[clip] = {"clip_identifier": clip, "error": f"timed out: no clip finished for {timeout} s"}
        if broken and isolating and not timed_out:
            clip = isolating[0]["clip_identifier"]   # it ran alone, so it is the clip that killed the worker
            results[clip] = {"clip_identifier": clip, "error": "killed its worker process (system out of memory or a crash in MediaPipe)"}
            LOGGER.error("%s (%s) killed its worker process when run alone; recorded as failed.", clip, settings_name)
        elif broken:
            suspects.extend(clip for clip in unfinished_started if clip not in suspects)
        if isolating:
            suspects.remove(isolating[0]["clip_identifier"])
    shutil.rmtree(started_folder, ignore_errors=True)
    for task in tasks:
        results.setdefault(task["clip_identifier"], {"clip_identifier": task["clip_identifier"],
                                                     "error": "not extracted: too many worker crashes in this pass"})
    return results


def run_clip_task(worker_function, task: dict) -> dict:
    """Runs in the worker: marks the clip as started (to tell which clips a broken pool was running), then extracts it."""
    marker = Path(task["work_folder"]) / "started_clips" / task["clip_identifier"]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return worker_function(task)


def run_worker_pool(batch: list[dict], workers: int, worker_function, timeout: float, results: dict,
                    settings_name: str, progress: tuple[float, int, int]) -> tuple[bool, bool]:
    """One process pool over the batch; fills results. Returns (pool broke, no clip finished within timeout).
    progress = (pass start time, results before this pass, clips to extract in this pass), for the progress log."""
    pass_started, results_before_pass, total = progress
    executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    futures = {executor.submit(run_clip_task, worker_function, task): task["clip_identifier"] for task in batch}
    waiting, broken, timed_out = set(futures), False, False
    while waiting and not broken:
        finished, waiting = wait(waiting, timeout=timeout, return_when=FIRST_COMPLETED)
        if not finished:
            LOGGER.error("%s pass: no clip finished for %d s; stopping this worker pool.", settings_name, timeout)
            broken = timed_out = True
            break
        for future in finished:
            clip = futures[future]
            try:
                results[clip] = future.result()
            except BrokenProcessPool:
                broken = True
                continue
            except Exception as error:   # noqa: BLE001 - one broken clip must not stop the run
                results[clip] = {"clip_identifier": clip, "error": repr(error)}
                LOGGER.error("%s (%s) failed: %r", clip, settings_name, error)
            done = len(results) - results_before_pass
            if done % 25 == 0 or done == total:
                elapsed = time.time() - pass_started
                LOGGER.info("%s pass: %d/%d clips, %.0f s elapsed, about %.0f s left.", settings_name, done, total,
                            elapsed, elapsed / max(done, 1) * (total - done))
    if broken:
        LOGGER.error("%s pass: worker pool stopped (a worker died or hung).", settings_name)
        for process in list((getattr(executor, "_processes", None) or {}).values()):   # stuck or orphaned workers
            process.terminate()
    executor.shutdown(wait=not broken, cancel_futures=True)
    return broken, timed_out


def load_extraction(work_folder: Path, settings_name: str, clip_identifier: str) -> tuple[np.ndarray, dict]:
    rows = np.load(work_folder / settings_name / f"{clip_identifier}.npy")
    with np.load(work_folder / settings_name / f"{clip_identifier}_hand_candidates.npz") as stored:
        candidates = {int(key): stored[key] for key in stored.files}
    return rows, candidates


def recover_and_measure(rows: np.ndarray, candidates: dict, recovery: HandRecoverySettings | None, configuration: dict) -> dict:
    if recovery is not None and rows.shape[0]:
        recovered, hand_source, counts = recover_missing_hands_in_clip(rows, candidates, recovery)
    else:
        recovered, hand_source, counts = rows, find_detected_hand_slots(rows).astype(np.uint8) if rows.shape[0] else None, {}
    problems = check_clip_integrity(recovered)
    quality = measure_clip_quality(recovered, hand_source, configuration["quality"]["pose_outlier"]) if not problems else None
    return {"rows": recovered, "hand_source": hand_source, "recovery_counts": counts, "integrity_problems": problems, "quality": quality}


def calibrate_or_load(clips: list[NamedClip], work_folder: Path, configuration: dict, calibration_file: Path | None,
                      seed: int) -> HandRecoverySettings | None:
    if not configuration["hand_recovery"]["enabled"]:
        return None
    if calibration_file is not None:
        if not Path(calibration_file).is_file():
            raise FileNotFoundError(f"Calibration file {calibration_file} does not exist (check CALIBRATION_FILE).")
        LOGGER.info("Using the given hand-recovery calibration %s.", calibration_file)
        return HandRecoverySettings.from_dictionary(json.loads(Path(calibration_file).read_text(encoding="utf-8")))
    per_signer = configuration["hand_recovery"]["calibration_clips_per_signer"]
    sample = select_clips(clips, None, per_signer, seed)
    raw_clips = []
    for clip in sample:
        path = work_folder / "primary" / f"{clip.clip_identifier}.npy"
        if path.exists():
            rows = np.load(path)
            if rows.ndim == 2 and rows.shape[0]:
                raw_clips.append(rows)
    try:
        settings = calibrate_recovery_settings(raw_clips, configuration["hand_recovery"]["neighbour_window_in_frames"])
    except ValueError as error:
        # Too few Holistic hands to measure trustworthy thresholds: recover nothing rather than guess.
        LOGGER.warning("Hand recovery switched OFF for this run, calibration impossible: %s", error)
        return None
    LOGGER.info("Calibrated hand recovery on %d clips: %s", len(raw_clips), json.dumps(settings.to_dictionary()))
    return settings


# ---- the run -------------------------------------------------------------------------------------

def run_dataset_extraction(dataset_name: str, input_folders: list[Path], output_root: Path, work_folder: Path, configuration: dict,
                           clip_name_pattern: str | None = None, signers: list[int] | None = None,
                           clips_per_signer: int | None = None, reference_pose_folders: list[Path] | None = None,
                           calibration_file: Path | None = None, worker_processes: int | None = None) -> dict:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"dataset_name must be one of {DATASET_NAMES}.")
    seed = configuration["runtime"]["seed"]
    workers = worker_processes or configuration["runtime"]["worker_processes"]
    work_folder, output_root = Path(work_folder), Path(output_root)
    named, pattern = name_clips(scan_dataset_folders([Path(folder) for folder in input_folders]), clip_name_pattern,
                                configuration["naming"][dataset_name])
    signer_conflicts = find_folder_signer_conflicts(named)
    for conflict in signer_conflicts:
        LOGGER.warning("Signer number conflict: %s", conflict)
    if signers:
        missing = sorted(set(signers) - {clip.signer_number for clip in named})
        if missing:
            raise RuntimeError(f"Selected signers {missing} have no clips in the input folders.")
    clips = select_clips(named, signers, clips_per_signer, seed)
    if not clips:
        raise RuntimeError("No clips selected.")
    review_clips = select_review_clips(clips, configuration["review"]["clips_per_signer"], seed)
    reference_index = index_reference_poses(reference_pose_folders) if reference_pose_folders else {}

    from raabta_pose_extraction.mediapipe_models import download_models_once
    download_models_once([configuration["holistic"]["primary"], configuration["holistic"]["enhanced"]],
                         configuration["hand_recovery"]["detector"] if configuration["hand_recovery"]["enabled"] else None)
    primary_metadata = run_extraction_pass(clips, "primary", configuration, work_folder, review_clips, workers)
    finished = [clip for clip in clips if "error" not in primary_metadata.get(clip.clip_identifier, {"error": "missing"})]
    recovery = calibrate_or_load(finished, work_folder, configuration, calibration_file, seed)

    outcomes, retry_clips = {}, []
    for clip in finished:
        rows, candidates = load_extraction(work_folder, "primary", clip.clip_identifier)
        outcome = recover_and_measure(rows, candidates, recovery, configuration)
        outcome["settings_name"] = "primary"
        outcome["retry_reasons"] = (["integrity"] if outcome["integrity_problems"]
                                    else list_retry_reasons(outcome["quality"], configuration["quality"]["clip_retry"]))
        outcomes[clip.clip_identifier] = outcome
        if outcome["retry_reasons"]:
            retry_clips.append(clip)
    LOGGER.info("%d of %d clips fail the per-clip checks and are re-extracted with the enhanced settings.",
                len(retry_clips), len(finished))
    enhanced_metadata = run_extraction_pass(retry_clips, "enhanced", configuration, work_folder, set(), workers) if retry_clips else {}
    for clip in retry_clips:
        if "error" in enhanced_metadata.get(clip.clip_identifier, {"error": "missing"}):
            continue
        rows, candidates = load_extraction(work_folder, "enhanced", clip.clip_identifier)
        enhanced = recover_and_measure(rows, candidates, recovery, configuration)
        primary = outcomes[clip.clip_identifier]
        primary_score = score_extraction(primary["quality"]) if primary["quality"] else float("-inf")
        enhanced_score = score_extraction(enhanced["quality"]) if enhanced["quality"] else float("-inf")
        enhanced["settings_name"] = "enhanced"
        enhanced["retry_reasons"] = primary["retry_reasons"]
        enhanced["primary_quality"] = primary["quality"]
        if enhanced_score > primary_score:
            outcomes[clip.clip_identifier] = enhanced
        else:
            primary["enhanced_quality"] = enhanced["quality"]

    staging = output_root / f"{dataset_name}_in_progress"
    if staging.exists():
        shutil.rmtree(staging)
    reports = []
    for clip in clips:
        identifier = clip.clip_identifier
        report = {"clip_identifier": identifier, "signer_folder_name": clip.signer_folder_name, "signer_number": clip.signer_number,
                  "sentence_number": clip.sentence_number, "repetition_number": clip.repetition_number,
                  "source": clip.relative_name, "source_kind": clip.source_kind}
        metadata = primary_metadata.get(identifier, {})
        if identifier not in outcomes:
            report.update(status="extraction_error", error=metadata.get("error", "missing"))
            reports.append(report)
            continue
        outcome = outcomes[identifier]
        chosen_metadata = enhanced_metadata.get(identifier, metadata) if outcome["settings_name"] == "enhanced" else metadata
        report.update({key: chosen_metadata.get(key) for key in ("frames_per_second", "frame_width_in_pixels", "frame_height_in_pixels",
                                                                 "frames_read", "seconds")})
        report.update(settings_used=outcome["settings_name"], retry_reasons=outcome["retry_reasons"],
                      integrity_problems=outcome["integrity_problems"], quality=outcome["quality"],
                      recovery_counts=outcome["recovery_counts"], primary_quality=outcome.get("primary_quality"),
                      enhanced_quality=outcome.get("enhanced_quality"))
        if outcome["integrity_problems"]:
            report["status"] = "integrity_failed"
            reports.append(report)
            continue
        reference_rows = None
        if identifier in reference_index:
            reference_rows = np.load(reference_index[identifier])
            report["reference_comparison"] = compare_with_reference(outcome["rows"], reference_rows, configuration["quality"]["pose_outlier"])
        clip_folder = staging / dataset_name / clip.signer_folder_name / identifier
        clip_folder.mkdir(parents=True, exist_ok=True)
        np.save(clip_folder / f"{identifier}.npy", outcome["rows"].astype(np.float32))
        np.save(clip_folder / "hand_source.npy", outcome["hand_source"])
        report["review_images"] = write_review_images(clip, outcome, reference_rows, work_folder, clip_folder)
        report["status"] = "written"
        (clip_folder / "extraction_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        reports.append(report)

    written = [report for report in reports if report["status"] == "written"]
    signer_summary = aggregate_by_signer(written)
    gates = evaluate_dataset_gates(signer_summary, configuration["quality"]["dataset_gates"], bool(reference_index))
    signer_outcomes = decide_signer_outcomes(clips, reports, gates,
                                             configuration["quality"]["dataset_gates"]["maximum_unwritten_clip_percent"])
    passed_signers = sorted(signer for signer, outcome in signer_outcomes.items() if outcome["passed"])
    failed_signers = sorted(signer for signer, outcome in signer_outcomes.items() if not outcome["passed"])
    from raabta_pose_extraction.mediapipe_models import describe_mediapipe_version
    summary = {
        "dataset_name": dataset_name, "input_folders": [str(folder) for folder in input_folders], "clip_name_pattern": pattern,
        "folder_signer_conflicts": signer_conflicts,
        "clips_selected": len(clips), "clips_written": len(written),
        "clips_re_extracted_with_enhanced_settings": len(retry_clips),
        "clips_kept_from_enhanced_settings": sum(1 for report in written if report["settings_used"] == "enhanced"),
        "status_counts": {status: sum(report["status"] == status for report in reports) for status in sorted({r["status"] for r in reports})},
        "passed_signers": passed_signers, "failed_signers": failed_signers, "signer_outcomes": signer_outcomes,
        "passed": not failed_signers, "signers": signer_summary, "gates": gates, "configuration": configuration,
        "hand_recovery_calibration": recovery.to_dictionary() if recovery else None,
        "hand_recovery_applied": recovery is not None,
        "mediapipe_version": describe_mediapipe_version(), "reference_clips_found": sum(1 for r in written if "reference_comparison" in r),
    }
    # Passing signers go to <dataset>/, failing signers to <dataset>_FAILED_VALIDATION/: the publishable
    # folder holds only signers that passed every gate, so a missing signer folder means that signer failed.
    # Only this run's signers are replaced, so several runs (signer groups) can share one output root.
    passed_folder, failed_folder = output_root / dataset_name, output_root / f"{dataset_name}_FAILED_VALIDATION"
    for signer in signer_outcomes:
        for folder in (passed_folder, failed_folder):
            if (folder / signer).exists():
                shutil.rmtree(folder / signer)
    for signer, outcome in signer_outcomes.items():
        source = staging / dataset_name / signer
        target = (passed_folder if outcome["passed"] else failed_folder) / signer
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            source.rename(target)
        else:
            target.mkdir(parents=True)
        if not outcome["passed"]:
            (target / "signer_failure.json").write_text(json.dumps(outcome, indent=1), encoding="utf-8")
    shutil.rmtree(staging, ignore_errors=True)   # absent when no clip of this run could be written
    run_name = "signers_" + "_".join(sorted(signer.removeprefix("signer") for signer in signer_outcomes))
    for folder in (passed_folder, failed_folder):
        if not folder.exists():
            continue
        (folder / "extraction_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")   # latest run
        (folder / "run_summaries").mkdir(exist_ok=True)
        (folder / "run_summaries" / f"{run_name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        merged = {}
        outcomes_file = folder / "signer_outcomes.json"
        if outcomes_file.exists():
            merged = json.loads(outcomes_file.read_text(encoding="utf-8")).get("signer_outcomes", {})
        merged.update(signer_outcomes)
        outcomes_file.write_text(json.dumps({"passed_signers": sorted(s for s, o in merged.items() if o["passed"]),
                                             "failed_signers": sorted(s for s, o in merged.items() if not o["passed"]),
                                             "signer_outcomes": merged}, indent=1), encoding="utf-8")
        with (folder / "clip_reports.jsonl").open("a", encoding="utf-8") as handle:
            for report in reports:
                handle.write(json.dumps(report) + "\n")
        if recovery is not None:
            (folder / "hand_recovery_calibration.json").write_text(json.dumps(recovery.to_dictionary(), indent=1), encoding="utf-8")
    summary["passed_folder"] = str(passed_folder) if passed_folder.exists() else None
    summary["failed_folder"] = str(failed_folder) if failed_folder.exists() else None
    LOGGER.info("Dataset %s: %d signers passed %s, %d failed %s.", dataset_name, len(passed_signers), passed_signers,
                len(failed_signers), failed_signers)
    return summary


def decide_signer_outcomes(clips: list[NamedClip], reports: list[dict], gates: dict, maximum_unwritten_clip_percent: float) -> dict:
    """Per signer: passed only if every gate passed and at most maximum_unwritten_clip_percent of its clips
    could not be extracted (those clips are listed; they are simply absent from the output)."""
    outcomes = {}
    for signer in sorted({clip.signer_folder_name for clip in clips}):
        signer_reports = [report for report in reports if report["signer_folder_name"] == signer]
        unwritten = [{"clip_identifier": report["clip_identifier"], "status": report["status"], "error": report.get("error"),
                      "integrity_problems": report.get("integrity_problems")}
                     for report in signer_reports if report["status"] != "written"]
        failed_gates = [result for result in gates["results"] if result["group"] == signer and not result["passed"]]
        unwritten_percent = round(100 * len(unwritten) / max(len(signer_reports), 1), 2)
        outcomes[signer] = {"passed": gates["signer_passed"].get(signer, False) and unwritten_percent <= maximum_unwritten_clip_percent,
                            "unwritten_clip_percent": unwritten_percent,
                            "clips": len(signer_reports), "clips_not_written": unwritten, "failed_gates": failed_gates}
    return outcomes


def write_review_images(clip: NamedClip, outcome: dict, reference_rows: np.ndarray | None, work_folder: Path,
                        clip_folder: Path) -> list[str]:
    """pose<frame>.jpg side-by-side images for the frames kept during extraction (review clips only)."""
    import cv2
    from raabta_pose_extraction.overlay_rendering import write_side_by_side_review_image
    review_folder = work_folder / "review_frames" / clip.clip_identifier
    if not review_folder.exists():
        return []
    rows, hand_source = outcome["rows"], outcome["hand_source"]
    width = max(4, len(str(rows.shape[0] - 1)))
    written = []
    for frame_file in sorted(review_folder.glob("*.png")):
        index = int(frame_file.stem)
        if index >= rows.shape[0]:
            continue
        frame = cv2.imread(str(frame_file))
        reference_row = reference_rows[index] if reference_rows is not None and index < reference_rows.shape[0] else None
        caption = f"{clip.clip_identifier} frame {index} ({outcome['settings_name']} settings)"
        output_file = clip_folder / f"pose{index:0{width}d}.jpg"
        write_side_by_side_review_image(frame, rows[index], hand_source[index], reference_row, caption, output_file)
        written.append(output_file.name)
    return written
