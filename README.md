# RaabtaAI-PSL: pose extraction

One MediaPipe pose extraction for **Isharah, ArSL and PSL**, so the recognition model is trained, fine-tuned and served on the same kind of poses. It checks quality gates per signer and writes passing signers to a publishable folder automatically, with no manual step.

Run it with `notebooks/pose_extraction.ipynb` on Kaggle (CPU, Internet on, Save & Run All), or with `scripts/extract_poses.py`.

## Output

```text
<output_root>/<dataset>/                       dataset = isharah | arsl | psl
    signer<SS>/<SS>_<CCCC>/<SS>_<CCCC>.npy     (T, 1662) float32 poses, one row per frame
    signer<SS>/<SS>_<CCCC>/hand_source.npy     (T, 2) uint8 per hand slot: 0 empty, 1 Holistic, 2 recovered
    signer<SS>/<SS>_<CCCC>/extraction_report.json   source, fps, frame size, settings used, quality, reference comparison
    signer<SS>/<SS>_<CCCC>/pose<FFFF>.jpg      side-by-side review images (a sample of clips only)
    extraction_summary.json                    configuration, MediaPipe version, calibration, per-signer quality, gates
    clip_reports.jsonl                         one line per clip
    hand_recovery_calibration.json
    signer_outcomes.json                       passed and failed signers, with the failed gates and clips

<output_root>/<dataset>_FAILED_VALIDATION/     same layout, only for signers that failed
    signer<SS>/signer_failure.json             the gates it failed and any clip that could not be extracted
```

- **Row layout:** pose 33 × (x, y, z, visibility), left hand 21 × 3, right hand 21 × 3, face 468 × 3. A part not detected is all zeros. This is the layout of the original Isharah files, so the RaabtaAI data pipeline reads it unchanged.
- **Input:** one or more input folders per dataset (`--input-folder` repeated; on Kaggle Isharah is split into one dataset per signer, `signerNN-isharah1000`). Each clip's name starts with its input folder's name, so the signer number can come from the folder name (`signer02-isharah1000/0001`) or from the path inside it. `discover` warns when the two disagree; with a reference, the frame-count gate then fails that signer.
- **Naming:** numbers keep the padding the source uses, with a minimum per dataset (`naming` in the config). Isharah is always `signer00/00_0001` (2 + 4 digits, like the original pose files), even when only some signers run; PSL sentence 7 of signer 3 becomes `signer03/03_07`; a repetition gives `03_07_01`.
- **Pose file name:** each clip's pose file is named after the clip (`00_0001.npy`), not `pose.npy`. RaabtaAI's Isharah reader finds clips by that name, so Isharah output works there with only a new folder glob.
- **Pass or fail per signer:** `<output_root>/<dataset>/` holds only signers whose clips were all extracted and whose gates all pass, so it can be published as it is. A signer missing there failed; its folder and `signer_failure.json` are in `<dataset>_FAILED_VALIDATION/`. The extraction exits with code 0 only when every signer passed.

## Extraction settings (`configs/pose_extraction.yaml`)

| Setting | Value | Why |
| --- | --- | --- |
| Holistic `model_complexity` | 2 (heavy) | best upper-body and wrist landmarks |
| `static_image_mode` | false | tracking: steadier landmarks between frames |
| `smooth_landmarks` | true | MediaPipe's temporal filter on the pose |
| `refine_face_landmarks` | true | more precise lips and eyes; iris points 468-477 dropped to keep 1,662 values |
| detection / tracking confidence | 0.5 / 0.5 | 0.3 locked onto a poster behind Isharah signer 8 in an earlier trial |
| hand recovery | MediaPipe Hands (static, 2 hands, complexity 1, confidence 0.3) on frames with a pose and an empty hand slot | Holistic only looks near the pose wrist; recovered hands must pass 4 checks calibrated on Holistic's own hands |
| enhanced settings (retry) | same, plus frames with a short side under 512 px enlarged ×2 | used only for clips that fail the per-clip checks; the better extraction is kept |

**Not verified here:** the heavy pose model (complexity 2) is downloaded by MediaPipe at first use. The test suite therefore runs with complexity 1 offline; the shipped configuration runs on Kaggle with Internet on.

## Checks and gates

**Per-clip checks.** A clip that fails any of these is re-extracted with the enhanced settings:

- pose found in fewer than 90% of frames;
- any hand found in fewer than 60% of frames with a pose;
- more than 2% pose outlier frames. An outlier frame is one where the shoulder midpoint is more than 1 shoulder width from the clip median, or the shoulder width is outside 0.6-1.6 × the median: a jump to another person or a poster.

**Gates**, per signer, set before any result; a signer passes only if all of its gates pass:

| Gate | Pass if |
| --- | --- |
| pose frames | ≥ 95% |
| pose outlier frames | ≤ 1% |
| face frames | ≥ 90%; skipped (and reported) for signers whose face is found in fewer than 20% of frames, for example niqabi signers |
| any-hand frames, without a reference | ≥ 80% |
| with a reference extraction (Isharah originals) | ≥ 99% of the signer's clips have an original pose file (a reference dataset not attached fails loudly); frame count equal to the original for ≥ 99% of those clips (catches a clip matched to the wrong frames); pose and any-hand coverage not below the original; median wrist offset from the original ≤ 0.10 shoulder widths. The overall any-hand gain (target ≥ 5 points) is reported but blocks no signer |
| clips written | at most 1% of the signer's clips not extracted (unreadable source, crash, timeout); those clips are listed in `signer_failure.json` / `clip_reports.jsonl` and absent from the output |

**Robustness.** Clips run in 4 worker processes. A clip that raises an error is recorded and the run goes on. A worker killed by the system breaks the pool: the clips it was running are re-run one at a time to find the one that caused it, which is recorded as failed, and every other clip continues. A worker that hangs is stopped after `clip_timeout_in_seconds` (3600) without any clip finishing.

The side-by-side images `pose<FFFF>.jpg` are kept in a sample of clip folders for optional spot checks; nothing waits for a person.

## Runtime

The notebook's pilot step measures the real speed and projects the full run. As a guide, complexity 1 took about 140 ms per frame per worker on 512 × 512 frames here; complexity 2 is slower.

- **PSL** (about 750 videos) fits one session.
- **Isharah** (3.5 million frames) needs several sessions. Split it with `SIGNERS` (a group per session, e.g. `[0, 1]`). From the second session on, attach the first session's output and set `CALIBRATION_FILE` to its `hand_recovery_calibration.json`, so every group uses the same hand-recovery thresholds. Several groups can also share one output root: a run replaces only its own signer folders, merges `signer_outcomes.json`, and keeps one summary per group in `run_summaries/`. `/kaggle/working` holds at most 20 GB, and Isharah poses are about 23 GB in total, so plan at most about 9 signers per session (time, not disk, is the real limit).

## Using the output in RaabtaAI

- **Isharah:** publish the folder as a Kaggle dataset whose name does NOT match `isharah*-pose` (for example `reextracted-isharah-poses`), so it is never read together with the original poses. Then add a dataset config with that folder glob, as done for the hand-recovered data.
- **PSL and ArSL:** the frame rate and size differ per clip. They are recorded in each `extraction_report.json` and the RaabtaAI reader for this format must use them (open decision 1 in RaabtaAI).
- **Consistency rule:** live inference must use this same configuration.

## Layout of the repository

| Path | Job |
| --- | --- |
| `src/raabta_pose_extraction/clip_discovery.py` | find videos or frame folders, read signer and sentence numbers |
| `src/raabta_pose_extraction/frame_sources.py` | frames in order, frame rate and size |
| `src/raabta_pose_extraction/mediapipe_models.py` | Holistic rows and Hands candidates |
| `src/raabta_pose_extraction/missing_hand_recovery.py` | calibrated hand recovery (same code as RaabtaAI) |
| `src/raabta_pose_extraction/quality_validation.py` | clip quality, outliers, retry rule, reference comparison, gates |
| `src/raabta_pose_extraction/overlay_rendering.py` | side-by-side review images |
| `src/raabta_pose_extraction/extraction_pipeline.py` | the whole run |
| `scripts/extract_poses.py` | `discover` and `extract` commands |
| `notebooks/pose_extraction.ipynb` | the Kaggle notebook |

Tests: `python -m pytest -q -p no:cacheprovider`. The test photo `tests/data/person_upper_body.png` is scikit-image's public-domain astronaut image.
