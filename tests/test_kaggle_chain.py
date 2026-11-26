"""The Kaggle chain with a stand-in kaggle CLI and stand-in extraction: skipping, publishing, time budget, relay."""

import json
import subprocess
from pathlib import Path

import pytest

from raabta_pose_extraction import kaggle_chain
from raabta_pose_extraction.kaggle_chain import ChainError, push_next_version, run_chain

class FakeKaggle:
    """Records kaggle CLI calls; datasets in `published` exist; created datasets become ready at once."""

    def __init__(self, published=(), notebook_cells=None):
        self.published, self.calls, self.pushed_notebook = set(published), [], None
        self.notebook_cells = notebook_cells or [{"cell_type": "code", "source": "X = 1\nCHAIN_STATE = {}\n", "outputs": [],
                                                  "execution_count": 3, "metadata": {}}]

    def __call__(self, command, timeout=None):
        self.calls.append(command)
        verb = command[1:3]
        if verb == ["datasets", "status"]:
            ready = command[3].split("/")[1] in self.published
            return subprocess.CompletedProcess(command, 0 if ready else 1, "ready" if ready else "", "" if ready else "404")
        if verb == ["datasets", "create"]:
            folder = Path(command[command.index("-p") + 1])
            self.published.add(json.loads((folder / "dataset-metadata.json").read_text())["id"].split("/")[1])
            return subprocess.CompletedProcess(command, 0, "created", "")
        if verb == ["datasets", "files"]:
            return subprocess.CompletedProcess(command, 0, "isharah/signer00/00_0001/00_0001.npy 1MB", "")
        if verb == ["kernels", "pull"]:
            folder = Path(command[command.index("-p") + 1])
            (folder / "kernel-metadata.json").write_text("{}")
            (folder / "notebook.ipynb").write_text(json.dumps({"cells": self.notebook_cells, "metadata": {}, "nbformat": 4,
                                                               "nbformat_minor": 5}))
            return subprocess.CompletedProcess(command, 0, "", "")
        if verb == ["kernels", "push"]:
            folder = Path(command[command.index("-p") + 1])
            self.pushed_notebook = json.loads(next(folder.glob("*.ipynb")).read_text())
            return subprocess.CompletedProcess(command, 0, "Kernel version 2 successfully pushed", "")
        raise AssertionError(command)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeExtraction:
    """discover writes the clips of the attached signers; extract writes a passing or failing signer and
    advances the clock by frames x seconds_per_frame."""

    def __init__(self, tmp_path, clock, signers, failing=(), seconds_per_frame=0.05, crashing=()):
        self.clock, self.failing, self.crashing, self.seconds_per_frame = clock, set(failing), set(crashing), seconds_per_frame
        self.signers, self.extracted = {}, []
        for number, frames in signers.items():
            folder = tmp_path / "input" / f"signer{number}-isharah1000" / f"{number:02d}_0001"
            folder.mkdir(parents=True)
            for index in range(frames):
                (folder / f"frame{index:04d}.jpg").write_bytes(b"")
            self.signers[f"signer{number:02d}"] = {"signer_number": number, "frames": frames, "clips": [
                {"clip_identifier": f"{number:02d}_0001", "source_path": str(folder), "source_kind": "frame_folder"}]}

    def __call__(self, command, working_folder, environment, timeout):
        if command[2] == "discover":
            summary = {"signers": {name: {"signer_number": entry["signer_number"], "clips": entry["clips"]}
                                   for name, entry in self.signers.items()}}
            Path(command[command.index("--summary-file") + 1]).write_text(json.dumps(summary))
            return 0
        number = int(command[command.index("--signers") + 1])
        name = f"signer{number:02d}"
        self.extracted.append(name)
        duration = self.signers[name]["frames"] * self.seconds_per_frame
        if duration > timeout:
            self.clock.now += timeout
            return None
        self.clock.now += duration
        if name in self.crashing:
            return 1
        root = Path(command[command.index("--output-root") + 1])
        if name in self.failing:
            folder = root / "isharah_FAILED_VALIDATION" / name
            folder.mkdir(parents=True)
            (folder / "signer_failure.json").write_text(json.dumps({"failed_gates": [
                {"gate": "any_hand_frame_percent", "value": 51.0, "threshold": ">= 80.0"}], "clips_not_written": []}))
            return 2
        (root / "isharah" / name / f"{number:02d}_0001").mkdir(parents=True)
        return 0


def make_settings(tmp_path, clock, **overrides):
    settings = {"dataset_name": "isharah", "input_folders": [str(tmp_path / "input")], "signers": None,
                "kaggle_username": "huda", "notebook_slug": "isharah-pose-group1", "dataset_prefix": "isharahpose",
                "output_root": str(tmp_path / "out"), "work_root": str(tmp_path / "work"), "repository_folder": str(tmp_path),
                "pose_python": "python", "environment": {}, "run_started_at": clock.now,
                "chain_state": json.loads(json.dumps(kaggle_chain.INITIAL_CHAIN_STATE)), "can_push_next_version": True}
    settings.update(overrides)
    return settings


def test_every_passing_signer_is_published_and_failing_ones_are_not(tmp_path):
    clock = FakeClock()
    extraction = FakeExtraction(tmp_path, clock, {0: 100, 1: 100, 2: 100}, failing={"signer01"})
    kaggle = FakeKaggle()
    report = run_chain(make_settings(tmp_path, clock), kaggle, extraction, clock)
    assert report["published"] == {"signer00": "huda/isharahpose-signer00", "signer02": "huda/isharahpose-signer02"}
    assert "any_hand_frame_percent=51.0" in report["failed"]["signer01"]
    assert not report["next_version_pushed"] and report["structure_problems"] == []
    assert kaggle.published == {"isharahpose-signer00", "isharahpose-signer02"}
    assert not list((tmp_path / "out").rglob("dataset-metadata.json"))       # removed after upload


def test_published_and_earlier_failed_signers_are_skipped(tmp_path):
    clock = FakeClock()
    extraction = FakeExtraction(tmp_path, clock, {0: 10, 1: 10, 2: 10})
    state = {"run": 2, "failed_signers": {"signer01": "any hand"}, "seconds_per_frame": 0.05}
    report = run_chain(make_settings(tmp_path, clock, chain_state=state), FakeKaggle(published={"isharahpose-signer00"}),
                       extraction, clock)
    assert extraction.extracted == ["signer02"]
    assert report["skipped_already_published"] == ["signer00"] and report["skipped_failed_earlier"] == ["signer01"]


def test_a_signer_that_does_not_fit_goes_to_the_next_version_with_the_measured_speed(tmp_path):
    clock = FakeClock()
    # 600 frames at 36 s per frame = 6 hours each: the second does not fit after the first
    extraction = FakeExtraction(tmp_path, clock, {0: 600, 1: 600}, seconds_per_frame=36)
    kaggle = FakeKaggle()
    report = run_chain(make_settings(tmp_path, clock), kaggle, extraction, clock)
    assert extraction.extracted == ["signer00"] and report["left_for_next_run"] == ["signer01"]
    assert report["next_version_pushed"]
    source = kaggle.pushed_notebook["cells"][0]["source"]
    pushed_state = json.loads(source.split("CHAIN_STATE = ")[1])
    assert pushed_state["run"] == 2 and pushed_state["seconds_per_frame"] == pytest.approx(36)
    assert source.startswith("X = 1\n") and kaggle.pushed_notebook["cells"][0]["outputs"] == []


def test_a_signer_too_long_for_a_whole_run_fails_instead_of_looping(tmp_path):
    clock = FakeClock()
    extraction = FakeExtraction(tmp_path, clock, {0: 1300, 1: 10}, seconds_per_frame=36)   # 13 hours
    report = run_chain(make_settings(tmp_path, clock), FakeKaggle(), extraction, clock)
    assert report["failed"]["signer00"].startswith("did not finish within one run")
    assert report["state_after_run"]["failed_signers"]["signer00"]


def test_a_crash_is_recorded_and_the_next_signer_still_runs(tmp_path):
    clock = FakeClock()
    extraction = FakeExtraction(tmp_path, clock, {0: 10, 1: 10}, crashing={"signer00"})
    report = run_chain(make_settings(tmp_path, clock), FakeKaggle(), extraction, clock)
    assert report["failed"]["signer00"] == "extraction crashed (exit code 1)" and "signer01" in report["published"]


def test_an_interactive_session_never_pushes_a_version(tmp_path):
    clock = FakeClock()
    extraction = FakeExtraction(tmp_path, clock, {0: 600, 1: 600}, seconds_per_frame=36)
    kaggle = FakeKaggle()
    report = run_chain(make_settings(tmp_path, clock, can_push_next_version=False), kaggle, extraction, clock)
    assert report["left_for_next_run"] == ["signer01"] and not report["next_version_pushed"] and kaggle.pushed_notebook is None


def test_selected_signers_limit_the_run(tmp_path):
    clock = FakeClock()
    extraction = FakeExtraction(tmp_path, clock, {0: 10, 1: 10, 2: 10})
    run_chain(make_settings(tmp_path, clock, signers=[2]), FakeKaggle(), extraction, clock)
    assert extraction.extracted == ["signer02"]


def test_pushing_needs_exactly_one_chain_state_line(tmp_path):
    kaggle = FakeKaggle(notebook_cells=[{"cell_type": "code", "source": "X = 1\n", "outputs": [], "execution_count": None,
                                         "metadata": {}}])
    with pytest.raises(ChainError, match="exactly one"):
        push_next_version("huda", "slug", {"run": 2}, kaggle)


def test_a_chain_that_never_ends_is_stopped(tmp_path):
    clock = FakeClock()
    with pytest.raises(ChainError, match="exceeds"):
        run_chain(make_settings(tmp_path, clock, chain_state={"run": 41, "failed_signers": {}, "seconds_per_frame": None}),
                  FakeKaggle(), FakeExtraction(tmp_path, clock, {0: 1}), clock)
