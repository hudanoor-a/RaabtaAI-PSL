"""Run the extraction on Kaggle as a self-continuing chain: one signer at a time, each passing signer
published as its own Kaggle dataset, with no manual step after the first Save & Run All.

Runs in the notebook's own Python (standard library only); the extraction itself runs in the
pose-extraction environment through scripts/extract_poses.py.

One run of the notebook:
  1. lists the attached signers (scripts/extract_poses.py discover; any number of input folders);
  2. skips signers already published (<user>/<prefix>-signerNN exists) or already failed in this chain;
  3. extracts the remaining signers ONE AT A TIME (so hand recovery is calibrated on that signer's own
     clips), each into its own output root, and publishes each passing signer;
  4. before each signer, estimates its time from its frame count and the measured speed; when it no
     longer fits in the run, or the run ends with signers left, pushes a new version of this notebook
     (same attached datasets, CHAIN_STATE updated), which starts by itself and continues.

Published dataset layout (one dataset per signer, nothing copied from the input folders' layout):
    <dataset>/signer<SS>/<SS>_<CCCC>/<SS>_<CCCC>.npy, hand_source.npy, extraction_report.json, pose<FFFF>.jpg
    <dataset>/extraction_summary.json, signer_outcomes.json, clip_reports.jsonl, hand_recovery_calibration.json

Facts not verifiable outside Kaggle, checked in the run and reported loudly when they fail:
  - a version pushed through the API keeps the notebook's secrets (needed by the next run);
  - `kaggle datasets create --dir-mode zip` keeps the folder structure (the file listing is checked);
  - pushing the next version while this run is still going does not cut this run short.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

KAGGLE_RUN_LIMIT_IN_SECONDS = 12 * 3600
SAFETY_MARGIN_IN_SECONDS = 50 * 60        # publishing the last signer and pushing the next version
ESTIMATE_SAFETY_FACTOR = 1.3              # measured speed varies between signers (frame content, retries)
MAXIMUM_CHAIN_RUNS = 40                   # stops a chain that would otherwise never end
DATASET_READY_TIMEOUT_IN_SECONDS = 60 * 60
CHAIN_STATE_LINE_PATTERN = re.compile(r"^CHAIN_STATE = .*$", re.MULTILINE)
IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
NOISE_PREFIXES = ("W0000", "I0000", "INFO: Created", "WARNING: All log")
INITIAL_CHAIN_STATE = {"run": 1, "failed_signers": {}, "seconds_per_frame": None}


class ChainError(RuntimeError):
    """A problem that must stop the chain with a clear message."""


def run_command(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """A short command (the kaggle CLI); output captured as text."""
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def dataset_slug(prefix: str, signer_folder_name: str) -> str:
    return f"{prefix}-{signer_folder_name}"


def is_dataset_published(username: str, slug: str, runner=run_command) -> bool:
    """True when <username>/<slug> already exists (published by an earlier run of this chain or another group)."""
    result = runner(["kaggle", "datasets", "status", f"{username}/{slug}"])
    return result.returncode == 0 and "ready" in (result.stdout or "").lower()


def count_signer_frames(clips: list[dict]) -> int | None:
    """Frames of one signer: image files in its frame folders. None for videos (time then is not estimated)."""
    total = 0
    for clip in clips:
        if clip["source_kind"] != "frame_folder":
            return None
        total += sum(1 for name in os.listdir(clip["source_path"]) if name.lower().endswith(IMAGE_FILE_EXTENSIONS))
    return total


def stream_process(command: list[str], working_folder: str, environment: dict, timeout: float) -> int | None:
    """Run a long command, print its output without MediaPipe's log noise. Returns the exit code, or None
    when it was stopped at the timeout (the whole process group, so the extraction's workers stop too)."""
    process = subprocess.Popen(command, cwd=working_folder, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def print_lines():
        for line in process.stdout:
            if not line.startswith(NOISE_PREFIXES):
                print(line, end="", flush=True)

    reader = threading.Thread(target=print_lines, daemon=True)
    reader.start()
    try:
        exit_code = process.wait(timeout=max(timeout, 1))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        exit_code = None
    reader.join(timeout=10)
    return exit_code


def read_signer_failure(signer_output_root: Path, dataset_name: str, signer_folder_name: str) -> str:
    failure_file = signer_output_root / f"{dataset_name}_FAILED_VALIDATION" / signer_folder_name / "signer_failure.json"
    if not failure_file.is_file():
        return "no signer_failure.json (the extraction did not finish)"
    failure = json.loads(failure_file.read_text(encoding="utf-8"))
    gates = [f"{gate['gate']}={gate['value']} (needs {gate['threshold']})" for gate in failure.get("failed_gates", [])]
    if failure.get("clips_not_written"):
        gates.append(f"{len(failure['clips_not_written'])} clips not written ({failure.get('unwritten_clip_percent')}%)")
    return "; ".join(gates) or "failed"


def publish_signer(signer_output_root: Path, dataset_name: str, username: str, slug: str, runner=run_command,
                   poll_seconds: float = 60, ready_timeout: float = DATASET_READY_TIMEOUT_IN_SECONDS) -> dict:
    """Publish <signer_output_root>/<dataset>/ as the private dataset <username>/<slug>. Called only for a passing
    signer, so the output root holds only <dataset>/. Waits until Kaggle has processed it and checks that the
    file listing keeps the signer folders."""
    unexpected = sorted(path.name for path in signer_output_root.iterdir() if path.name != dataset_name)
    if unexpected:
        return {"published": False, "error": f"{signer_output_root} holds more than {dataset_name}/: {unexpected}"}
    metadata_file = signer_output_root / "dataset-metadata.json"
    metadata = {"title": slug, "id": f"{username}/{slug}", "licenses": [{"name": "unknown"}]}
    metadata_file.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    created = runner(["kaggle", "datasets", "create", "-p", str(signer_output_root), "--dir-mode", "zip", "-q"])
    metadata_file.unlink()
    if created.returncode != 0:
        return {"published": False, "error": f"kaggle datasets create failed: {(created.stderr or created.stdout).strip()[-500:]}"}
    deadline = time.time() + ready_timeout
    while True:
        status = runner(["kaggle", "datasets", "status", f"{username}/{slug}"])
        text = (status.stdout or "").strip().lower()
        if status.returncode == 0 and "ready" in text:
            break
        if "error" in text or time.time() > deadline:
            return {"published": False, "error": f"dataset not ready: {text or status.stderr}"}
        time.sleep(poll_seconds)
    listing = runner(["kaggle", "datasets", "files", f"{username}/{slug}", "--page-size", "200"])
    listed = listing.stdout or ""
    # None = unknown (the listing command failed); False = Kaggle shows a zip or no signer folders
    structure_kept = None if listing.returncode != 0 else ("signer" in listed and ".zip" not in listed)
    return {"published": True, "dataset": f"{username}/{slug}", "structure_kept": structure_kept,
            "file_listing_start": listed[:1500]}


def push_next_version(username: str, notebook_slug: str, new_state: dict, runner=run_command) -> None:
    """Pull this notebook (its source and metadata, which lists the attached datasets), write the new
    CHAIN_STATE into the settings cell, and push it: Kaggle then runs the new version by itself."""
    folder = Path(tempfile.mkdtemp(prefix="next_version_"))
    pulled = runner(["kaggle", "kernels", "pull", f"{username}/{notebook_slug}", "-p", str(folder), "-m"])
    if pulled.returncode != 0:
        raise ChainError(f"Could not pull {username}/{notebook_slug}: {(pulled.stderr or pulled.stdout).strip()}")
    notebooks = list(folder.glob("*.ipynb"))
    if len(notebooks) != 1 or not (folder / "kernel-metadata.json").is_file():
        raise ChainError(f"Pulled notebook folder is not as expected: {sorted(p.name for p in folder.iterdir())}")
    notebook = json.loads(notebooks[0].read_text(encoding="utf-8"))
    replaced = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        new_source, count = CHAIN_STATE_LINE_PATTERN.subn("CHAIN_STATE = " + json.dumps(new_state), source)
        if count:
            cell["source"], replaced = new_source, replaced + count
        cell["outputs"], cell["execution_count"] = [], None
    if replaced != 1:
        raise ChainError(f"The notebook must have exactly one 'CHAIN_STATE = ...' line in a code cell; found {replaced}.")
    notebooks[0].write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    pushed = runner(["kaggle", "kernels", "push", "-p", str(folder)])
    shutil.rmtree(folder, ignore_errors=True)
    if pushed.returncode != 0 or "error" in (pushed.stdout or "").lower():
        raise ChainError(f"Could not push the next version: {(pushed.stderr or pushed.stdout).strip()}")
    print((pushed.stdout or "").strip())


def run_chain(settings: dict, runner=run_command, process_runner=stream_process, clock=time.time) -> dict:
    """One run of the chain (see module docstring). settings: dataset_name, input_folders, signers (None = all),
    kaggle_username, notebook_slug, dataset_prefix, output_root, work_root, repository_folder, pose_python,
    environment, run_started_at, chain_state, can_push_next_version; optional clip_name_pattern, configuration_file."""
    state = json.loads(json.dumps(settings["chain_state"]))
    if state["run"] > MAXIMUM_CHAIN_RUNS:
        raise ChainError(f"Chain run {state['run']} exceeds {MAXIMUM_CHAIN_RUNS} runs; stopping.")
    deadline = settings["run_started_at"] + KAGGLE_RUN_LIMIT_IN_SECONDS - SAFETY_MARGIN_IN_SECONDS
    output_root, work_root = Path(settings["output_root"]), Path(settings["work_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_name = settings["dataset_name"]

    base = [settings["pose_python"], "scripts/extract_poses.py"]
    folder_arguments = [argument for folder in settings["input_folders"] for argument in ("--input-folder", folder)]
    if settings.get("clip_name_pattern"):
        folder_arguments += ["--clip-name-pattern", settings["clip_name_pattern"]]
    if settings.get("configuration_file"):   # default: configs/pose_extraction.yaml
        folder_arguments += ["--config", settings["configuration_file"]]
    discovery_file = output_root / "discovered_clips.json"
    discover = base + ["discover", "--dataset-name", dataset_name, *folder_arguments, "--summary-file", str(discovery_file)]
    if settings.get("signers"):
        discover += ["--signers", *map(str, settings["signers"])]
    if process_runner(discover, settings["repository_folder"], settings["environment"], 3600) != 0:
        raise ChainError("discover failed: read the message above (clip names or input folders).")
    discovered = json.loads(discovery_file.read_text(encoding="utf-8"))["signers"]
    selected = [signer for signer, entry in sorted(discovered.items())
                if not settings.get("signers") or entry["signer_number"] in settings["signers"]]

    report = {"run": state["run"], "published": {}, "failed": {}, "skipped_already_published": [],
              "skipped_failed_earlier": [], "left_for_next_run": [], "structure_problems": []}
    todo = []
    for signer in selected:
        if signer in state["failed_signers"]:
            report["skipped_failed_earlier"].append(signer)
        elif is_dataset_published(settings["kaggle_username"], dataset_slug(settings["dataset_prefix"], signer), runner):
            report["skipped_already_published"].append(signer)
        else:
            todo.append(signer)
    print(f"chain run {state['run']}: {len(selected)} signers attached and selected, to do now: {todo}")
    print(f"  already published: {report['skipped_already_published']} | failed earlier: {report['skipped_failed_earlier']}")

    processed_in_this_run = 0
    for position, signer in enumerate(todo):
        remaining = deadline - clock()
        frames = count_signer_frames(discovered[signer]["clips"])
        estimate = frames * state["seconds_per_frame"] * ESTIMATE_SAFETY_FACTOR if frames and state["seconds_per_frame"] else None
        print(f"\n=== {signer}: {len(discovered[signer]['clips'])} clips, {frames} frames, "
              f"estimate {'unknown' if estimate is None else f'{estimate / 3600:.1f} h'}, {remaining / 3600:.1f} h left in this run")
        if processed_in_this_run and (estimate is None or estimate > remaining):
            report["left_for_next_run"] = todo[position:]
            break
        signer_root, signer_work = output_root / signer, work_root / signer
        shutil.rmtree(signer_root, ignore_errors=True)
        shutil.rmtree(signer_work, ignore_errors=True)
        extract = base + ["extract", "--dataset-name", dataset_name, *folder_arguments,
                          "--signers", str(discovered[signer]["signer_number"]),
                          "--output-root", str(signer_root), "--work-folder", str(signer_work)]
        started = clock()
        exit_code = process_runner(extract, settings["repository_folder"], settings["environment"], remaining)
        elapsed = clock() - started
        shutil.rmtree(signer_work, ignore_errors=True)
        if exit_code is None:
            if processed_in_this_run:   # it had less than a full run: try again first thing in the next run
                report["left_for_next_run"] = todo[position:]
                print(f"{signer}: stopped at the time limit; it restarts in the next run")
                break
            reason = f"did not finish within one run ({elapsed / 3600:.1f} h)"
            state["failed_signers"][signer] = report["failed"][signer] = reason
            processed_in_this_run += 1
            continue
        processed_in_this_run += 1
        if frames:
            state["seconds_per_frame"] = elapsed / frames
        if exit_code == 0:
            slug = dataset_slug(settings["dataset_prefix"], signer)
            result = publish_signer(signer_root, dataset_name, settings["kaggle_username"], slug, runner)
            if result["published"]:
                report["published"][signer] = result["dataset"]
                if result["structure_kept"] is not True:
                    report["structure_problems"].append({"signer": signer, "structure_kept": result["structure_kept"],
                                                         "listing": result["file_listing_start"]})
                print(f"{signer}: PASSED and published as {result['dataset']}")
            else:
                state["failed_signers"][signer] = report["failed"][signer] = "publishing failed: " + result["error"]
        elif exit_code == 2:
            state["failed_signers"][signer] = report["failed"][signer] = read_signer_failure(signer_root, dataset_name, signer)
            print(f"{signer}: FAILED validation, not published: {report['failed'][signer]}")
        else:
            state["failed_signers"][signer] = report["failed"][signer] = f"extraction crashed (exit code {exit_code})"
            print(f"{signer}: extraction crashed (exit code {exit_code}), not published")

    report["state_after_run"] = state
    report["next_version_pushed"] = False
    push_needed = bool(report["left_for_next_run"]) and settings["can_push_next_version"]
    # Written before the push: if Kaggle stops this run when the new version starts, the report is already saved.
    with (output_root / "chain_report.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(report, next_version_pushed="pushing" if push_needed else False)) + "\n")
    if report["left_for_next_run"] and not push_needed:
        print("Signers left, but this is an interactive session: no next version is pushed. Use Save & Run All.")
    if push_needed:
        push_next_version(settings["kaggle_username"], settings["notebook_slug"], dict(state, run=state["run"] + 1), runner)
        report["next_version_pushed"] = True
    return report
