# Naming conventions

These rules apply to every file, folder, class, function, variable, config key and constant.

- **Full English words, no abbreviations.** Do not write `kp`, `preproc`, `proc`, `cfg`, `util`, `tmp`, `misc`, `helpers` or `data2`. Write `pose`, `configuration`, `temporary_path`. Two accepted exceptions:
  - `nn` in `from torch import nn` (PyTorch's own convention);
  - external API names such as `cv2.CAP_PROP_POS_MSEC` and `mp.solutions.holistic`.
- **Files and folders:** `snake_case` that describes the job. Write `frame_rate_resampling.py`, not `resample.py`.
- **Classes:** `PascalCase` nouns, for example `SignerFoldGenerator`.
- **Functions:** verb then object, for example `resample_pose_sequence_to_target_frame_rate`.
- **Units go in names:** `target_frames_per_second`, `frame_width_in_pixels`, `maximum_gap_in_frames`, `frame_timestamps_in_seconds`.
- **Forbidden file names:** `utils.py`, `helpers.py`, `common.py`, `misc.py`. Put shared code in the module that owns the concept.
- **Every module starts with a docstring** saying what it does and where it sits in the pipeline.
- **Config keys** follow the same rules as variables. Unknown facts get an explicit `null` or a value marked `# PLACEHOLDER`. Never invent a plausible-looking value without that mark.
