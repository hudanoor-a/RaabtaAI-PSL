# RaabtaAI-PSL: pose extraction

One MediaPipe pose extraction for **Isharah, ArSL and PSL**, so the recognition model is trained, fine-tuned and served on the same kind of poses. It checks quality gates per signer and writes passing signers to a publishable folder automatically, with no manual step.

Run it with `notebooks/pose_extraction.ipynb` on Kaggle (CPU, Internet on, Save & Run All): it extracts one signer at a time and publishes every passing signer as its own dataset, continuing by itself across 12-hour runs. `scripts/extract_poses.py` runs the same extraction from the command line.

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
- **Input:** one or more input folders per dataset (`--input-folder` repeated; on Kaggle Isharah is split into one dataset per signer, `signerNN-isharah1000`). Each clip's name starts with its input folder's name, so the signer number can come from the folder name (`signer02-isharah1000/0001`) or from the path inside it. `discover` warns when the two disagree. Signer and sentence come from the clip folder name (`00_0001`); intermediate folders such as `SarahAlyami/isharah500/00/` are ignored.
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
| any-hand frames | ≥ 80% |
| only with `--reference-pose-folder` (not used by the notebook) | ≥ 99% of the signer's clips have an original pose file (a reference dataset not attached fails loudly); frame count equal to the original for ≥ 99% of those clips (catches a clip matched to the wrong frames); pose and any-hand coverage not below the original; median wrist offset from the original ≤ 0.10 shoulder widths. The overall any-hand gain (target ≥ 5 points) is reported but blocks no signer |
| clips written | at most 1% of the signer's clips not extracted (unreadable source, crash, timeout); those clips are listed in `signer_failure.json` / `clip_reports.jsonl` and absent from the output |

**Robustness.** Clips run in 4 worker processes. A clip that raises an error is recorded and the run goes on. A worker killed by the system breaks the pool: the clips it was running are re-run one at a time to find the one that caused it, which is recorded as failed, and every other clip continues. A worker that hangs is stopped after `clip_timeout_in_seconds` (3600) without any clip finishing.

The side-by-side images `pose<FFFF>.jpg` are kept in a sample of clip folders for optional spot checks; nothing waits for a person.

## Running on Kaggle: one notebook, no manual step

`notebooks/pose_extraction.ipynb` runs as a self-continuing chain (`src/raabta_pose_extraction/kaggle_chain.py`):

1. It lists the attached signers. Any number of datasets can be attached; nothing depends on how many.
2. It skips signers already published (`<user>/isharahpose-signerNN` exists) or failed earlier in the chain.
3. It extracts the remaining signers **one at a time**, so hand recovery is calibrated on each signer's own Holistic hands. A passing signer is published at once as a private dataset `isharahpose-signerNN`; a failing one is not (its poses and `signer_failure.json` stay in that version's output).
4. Before each signer, it compares the signer's frame count × the measured speed with the time left in the 12-hour run. When the signer no longer fits, it pushes a new version of itself (same attached datasets, `CHAIN_STATE` updated), which starts by itself.

**Setup, once per notebook:** attach the frame datasets (one group per notebook if you split them; groups must not share a signer); accelerator None, Internet on; secrets `github_token1` and `KAGGLE_KEY`; in step 1 set `KAGGLE_USERNAME` and `NOTEBOOK_SLUG`; then Save Version → Save & Run All.

**Not verifiable outside Kaggle** (each is checked in the run and fails loudly):
- a version pushed through the API keeps the notebook's secrets: if not, the next version stops at step 2, and the signers already published stay published;
- `kaggle datasets create --dir-mode zip` keeps the folder structure: the run checks the dataset's file listing and raises an error at the end if it shows a zip or no signer folders;
- pushing the next version does not cut the current run short: the run report is written before the push.

**Speed:** measured by the run itself (`seconds_per_frame` in `CHAIN_STATE`). As a guide, complexity 1 took about 140 ms per frame per worker on 512 × 512 frames here; complexity 2 is slower. A signer that does not finish within a whole run is recorded as failed instead of being retried forever; the chain stops after 40 versions.

`scripts/extract_poses.py` also runs alone (`discover`, `extract`); the optional `--reference-pose-folder` (comparison with an earlier extraction) and `--calibration-file` are not used by the notebook.

## Using the output in RaabtaAI

- **Isharah:** the published datasets `isharahpose-signerNN` each hold `isharah/signerNN/...`; their names do not match `isharah*-pose`, so they are never read together with the original poses. Add a RaabtaAI dataset config with the folder glob `*/isharahpose-signer*/isharah`.
- **Checking against the original poses** is done separately (not in this extraction): compare per clip, including the frame count, which also catches a dataset labelled with the wrong signer.
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
| `src/raabta_pose_extraction/extraction_pipeline.py` | the whole run for a set of signers |
| `src/raabta_pose_extraction/kaggle_chain.py` | Kaggle: one signer at a time, publishing, next version |
| `scripts/extract_poses.py` | `discover` and `extract` commands |
| `notebooks/pose_extraction.ipynb` | the Kaggle notebook |

Tests: `python -m pytest -q -p no:cacheprovider`. The test photo `tests/data/person_upper_body.png` is scikit-image's public-domain astronaut image.
