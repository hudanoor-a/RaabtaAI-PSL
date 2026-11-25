# RaabtaAI-PSL: pose extraction

One MediaPipe pose extraction for **Isharah, ArSL and PSL**, so the recognition model is trained, fine-tuned and served on the same kind of poses. It writes validated per-clip pose files and keeps a dataset publishable only when every quality gate passes.

Run it with `notebooks/pose_extraction.ipynb` on Kaggle (CPU, Internet on), or with `scripts/extract_poses.py`.

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
    VALIDATION_PASSED.json | VALIDATION_FAILED.json
```

- **Row layout:** pose 33 × (x, y, z, visibility), left hand 21 × 3, right hand 21 × 3, face 468 × 3. A part not detected is all zeros. This is the layout of the original Isharah files, so the RaabtaAI data pipeline reads it unchanged.
- **Naming:** numbers keep the padding the source uses, at least 2 digits. Isharah becomes `signer00/00_0001`; PSL sentence 7 of signer 3 becomes `signer03/03_07`; a repetition gives `03_07_01`.
- **Pose file name:** each clip's pose file is named after the clip (`00_0001.npy`), not `pose.npy`. RaabtaAI's Isharah reader finds clips by that name, so Isharah output works there with only a new folder glob.
- **Failed validation:** when any gate fails, the dataset folder is named `<dataset>_FAILED_VALIDATION`.

## Extraction settings (`configs/pose_extraction.yaml`)

| Setting | Value | Why |
| --- | --- | --- |
| Holistic `model_complexity` | 2 (heavy) | best upper-body and wrist landmarks |
| `static_image_mode` | false | tracking: steadier landmarks between frames |
| `smooth_landmarks` | true | MediaPipe's temporal filter on the pose |
| `refine_face_landmarks` | true | more precise lips and eyes; iris points 468-477 dropped to keep 1,662 values |
| detection / tracking confidence | 0.5 / 0.5 | 0.3 locked onto a poster behind Isharah signer 8 in an earlier trial |
| hand recovery | MediaPipe Hands (static, 2 hands, complexity 1, confidence 0.5) on frames with a pose and an empty hand slot | Holistic only looks near the pose wrist; recovered hands must pass 4 checks calibrated on Holistic's own hands |
| enhanced settings (retry) | same, plus frames with a short side under 512 px enlarged ×2 | used only for clips that fail the per-clip checks; the better extraction is kept |

**Not verified here:** the heavy pose model (complexity 2) is downloaded by MediaPipe at first use. The test suite therefore runs with complexity 1 offline; the shipped configuration runs on Kaggle with Internet on.

## Checks and gates

**Per-clip checks.** A clip that fails any of these is re-extracted with the enhanced settings:

- pose found in fewer than 90% of frames;
- any hand found in fewer than 60% of frames with a pose;
- more than 2% pose outlier frames. An outlier frame is one where the shoulder midpoint is more than 1 shoulder width from the clip median, or the shoulder width is outside 0.6-1.6 × the median: a jump to another person or a poster.

**Dataset gates**, per signer, set before any result:

| Gate | Pass if |
| --- | --- |
| pose frames | ≥ 95% |
| pose outlier frames | ≤ 1% |
| face frames | ≥ 90%; skipped (and reported) for signers whose face is found in fewer than 20% of frames, for example niqabi signers |
| any-hand frames, without a reference | ≥ 80% |
| with a reference extraction (Isharah originals) | pose and any-hand coverage not below the reference for any signer; overall any-hand gain ≥ 5 points; median wrist offset from the reference ≤ 0.10 shoulder widths |
| all clips written | every selected clip extracted without error |
| human review (notebook step 6) | ≥ 95% of the side-by-side images correct |

## Runtime

The notebook's pilot step measures the real speed and projects the full run. As a guide, complexity 1 took about 140 ms per frame per worker on 512 × 512 frames here; complexity 2 is slower.

- **PSL** (about 750 videos) fits one session.
- **Isharah** (3.5 million frames) needs several sessions. Split it with `SIGNERS`, and reuse the first session's `hand_recovery_calibration.json` through `CALIBRATION_FILE`, so every shard uses the same thresholds. `/kaggle/working` holds at most 20 GB, and Isharah poses are about 23 GB in total, so plan at most about 9 signers per session.

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
