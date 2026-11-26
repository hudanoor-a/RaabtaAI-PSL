"""Worker-pool behaviour of the extraction passes: a hung or killed worker must not stop the run."""

import time

from raabta_pose_extraction.extraction_pipeline import run_extraction_pass


def make_pass_inputs(tmp_path, timeout_in_seconds):
    from raabta_pose_extraction.clip_discovery import NamedClip
    clips = [NamedClip(tmp_path / f"00_{n:04d}", "frame_folder", f"x/{n:04d}", 0, n, None, "signer00", f"00_{n:04d}")
             for n in range(1, 5)]
    configuration = {"holistic": {"primary": {}}, "hand_recovery": {"enabled": False, "detector": {}},
                     "review": {"frames_per_clip": 0}, "runtime": {"clip_timeout_in_seconds": timeout_in_seconds}}
    return clips, configuration


def test_a_hung_worker_times_out_and_the_other_clips_still_finish(tmp_path):
    from stand_in_clip_workers import finish_quickly_or_hang
    clips, configuration = make_pass_inputs(tmp_path, 5)
    started = time.time()
    results = run_extraction_pass(clips, "primary", configuration, tmp_path, set(), 2, worker_function=finish_quickly_or_hang)
    assert time.time() - started < 60
    assert results["00_0002"]["error"].startswith("timed out")
    assert all("error" not in results[f"00_{n:04d}"] for n in (1, 3, 4))


def test_a_clip_that_kills_its_worker_is_isolated_and_the_other_clips_still_finish(tmp_path):
    from stand_in_clip_workers import finish_quickly_or_crash
    clips, configuration = make_pass_inputs(tmp_path, 60)
    results = run_extraction_pass(clips, "primary", configuration, tmp_path, set(), 2, worker_function=finish_quickly_or_crash)
    assert results["00_0002"]["error"].startswith("killed its worker process")   # found by re-running suspects alone
    assert all("error" not in results[f"00_{n:04d}"] for n in (1, 3, 4))
