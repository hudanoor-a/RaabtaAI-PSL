"""The shipped configuration loads; a misspelled Holistic key is refused."""

from pathlib import Path

import pytest
import yaml

from raabta_pose_extraction.configuration import ConfigurationError, load_extraction_configuration

CONFIGURATION_FILE = Path(__file__).resolve().parents[1] / "configs" / "pose_extraction.yaml"


def test_shipped_configuration_has_the_decided_values():
    configuration = load_extraction_configuration(CONFIGURATION_FILE)
    primary, enhanced = configuration["holistic"]["primary"], configuration["holistic"]["enhanced"]
    assert primary["model_complexity"] == 2 and primary["refine_face_landmarks"] and not primary["static_image_mode"]
    assert primary["min_detection_confidence"] == 0.5 and primary["upscale_short_side_below_pixels"] == 0
    assert enhanced["upscale_short_side_below_pixels"] == 512 and configuration["hand_recovery"]["enabled"]


def test_misspelled_keys_are_refused(tmp_path):
    configuration = yaml.safe_load(CONFIGURATION_FILE.read_text(encoding="utf-8"))
    configuration["holistic"]["primary"]["model_complexty"] = configuration["holistic"]["primary"].pop("model_complexity")
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="model_complexty"):
        load_extraction_configuration(broken)
