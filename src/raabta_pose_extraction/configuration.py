"""Load configs/pose_extraction.yaml and refuse it when a required key is missing or misspelled."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_KEYS = {
    "holistic": {"primary", "enhanced"},
    "hand_recovery": {"enabled", "detector", "calibration_clips_per_signer", "neighbour_window_in_frames"},
    "quality": {"clip_retry", "pose_outlier", "dataset_gates"},
    "review": {"clips_per_signer", "frames_per_clip"},
    "runtime": {"worker_processes", "clip_timeout_in_seconds", "seed"},
}
HOLISTIC_KEYS = {"static_image_mode", "model_complexity", "smooth_landmarks", "refine_face_landmarks",
                 "min_detection_confidence", "min_tracking_confidence", "upscale_short_side_below_pixels"}


class ConfigurationError(ValueError):
    """The extraction configuration is incomplete or has unknown keys."""


def load_extraction_configuration(configuration_file: str | Path) -> dict:
    with Path(configuration_file).open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    for section, keys in REQUIRED_KEYS.items():
        if section not in configuration:
            raise ConfigurationError(f"Missing section '{section}'.")
        missing = keys - set(configuration[section])
        if missing:
            raise ConfigurationError(f"Section '{section}' is missing {sorted(missing)}.")
    for name in ("primary", "enhanced"):
        keys = set(configuration["holistic"][name])
        if keys != HOLISTIC_KEYS:
            raise ConfigurationError(f"holistic.{name} keys must be exactly {sorted(HOLISTIC_KEYS)}; "
                                     f"missing {sorted(HOLISTIC_KEYS - keys)}, unknown {sorted(keys - HOLISTIC_KEYS)}.")
    return configuration
