"""MediaPipe Holistic (raw 1662-value rows) and MediaPipe Hands (missed-hand candidates).

Runs only in the pose-extraction environment: Python 3.12 venv with mediapipe 0.10.21,
numpy 1.26.4, protobuf 4.25.3, opencv-contrib-python 4.9.0.80 (requirements.txt).

Raw row per frame, the layout of the original Isharah files:
    pose 33 x (x, y, z, visibility) + left hand 21 x 3 + right hand 21 x 3 + face 468 x 3 = 1662
A part that is not detected is written as zeros. With refine_face_landmarks the face mesh has
478 points; the first 468 are the same points as without refinement (only placed more
precisely), and the 10 iris points 468-477 are dropped so the layout does not change.

`upscale_short_side_below_pixels`: when the frame's short side is below this value, the frame is
enlarged x2 (cubic) before detection. Landmarks stay normalised to the frame, so the output is
directly comparable with an extraction made without upscaling. 0 switches it off.

A NEW Holistic object is created for every clip so tracking state never leaks between clips.
The two sys.modules lines are a known packaging workaround (mediapipe/__init__ imports the Tasks
API, which fails on Kaggle images); they must run before `import mediapipe`.
"""

import sys
import types

sys.modules['mediapipe.tasks'] = types.ModuleType('mediapipe.tasks')
sys.modules['mediapipe.tasks.python'] = types.ModuleType('mediapipe.tasks.python')

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
FACE_LANDMARK_COUNT = 468
RAW_VALUES_PER_FRAME = POSE_LANDMARK_COUNT * 4 + 2 * HAND_LANDMARK_COUNT * 3 + FACE_LANDMARK_COUNT * 3
HOLISTIC_OPTION_NAMES = ("static_image_mode", "model_complexity", "smooth_landmarks", "refine_face_landmarks",
                         "min_detection_confidence", "min_tracking_confidence")


def prepare_frame(frame_bgr: np.ndarray, upscale_short_side_below_pixels: int) -> np.ndarray:
    """RGB frame, enlarged x2 when its short side is below the threshold."""
    if upscale_short_side_below_pixels and min(frame_bgr.shape[:2]) < upscale_short_side_below_pixels:
        frame_bgr = cv2.resize(frame_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    return frame_rgb


def landmarks_to_array(landmark_list, count: int, with_visibility: bool) -> np.ndarray:
    if landmark_list is None:
        return np.zeros(count * (4 if with_visibility else 3), dtype=np.float32)
    points = landmark_list.landmark[:count]
    if with_visibility:
        return np.array([[point.x, point.y, point.z, point.visibility] for point in points], dtype=np.float32).reshape(-1)
    return np.array([[point.x, point.y, point.z] for point in points], dtype=np.float32).reshape(-1)


def flatten_holistic_results(results) -> np.ndarray:
    row = np.concatenate([
        landmarks_to_array(results.pose_landmarks, POSE_LANDMARK_COUNT, True),
        landmarks_to_array(results.left_hand_landmarks, HAND_LANDMARK_COUNT, False),
        landmarks_to_array(results.right_hand_landmarks, HAND_LANDMARK_COUNT, False),
        landmarks_to_array(results.face_landmarks, FACE_LANDMARK_COUNT, False),
    ])
    if row.shape[0] != RAW_VALUES_PER_FRAME:
        raise ValueError(f"Holistic row has {row.shape[0]} values, expected {RAW_VALUES_PER_FRAME}.")
    return row


class HolisticExtractor:
    """One Holistic object for one clip: `with HolisticExtractor(settings) as extractor: extractor.extract(frame)`."""

    def __init__(self, settings: dict):
        self.holistic_options = {name: settings[name] for name in HOLISTIC_OPTION_NAMES}
        self.upscale_short_side_below_pixels = int(settings.get("upscale_short_side_below_pixels", 0))
        self._holistic = None

    def __enter__(self) -> "HolisticExtractor":
        self._holistic = mp.solutions.holistic.Holistic(**self.holistic_options)
        return self

    def __exit__(self, *exception) -> None:
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        return flatten_holistic_results(self._holistic.process(prepare_frame(frame_bgr, self.upscale_short_side_below_pixels)))


class HandDetector:
    """MediaPipe Hands on single frames (static mode): `detect(frame)` -> (N, 21, 3) float32, N = 0, 1 or 2."""

    def __init__(self, settings: dict, upscale_short_side_below_pixels: int = 0):
        self.options = {"static_image_mode": True, "max_num_hands": int(settings["max_num_hands"]),
                        "model_complexity": int(settings["model_complexity"]),
                        "min_detection_confidence": float(settings["min_detection_confidence"])}
        self.upscale_short_side_below_pixels = upscale_short_side_below_pixels
        self._hands = None

    def __enter__(self) -> "HandDetector":
        self._hands = mp.solutions.hands.Hands(**self.options)
        return self

    def __exit__(self, *exception) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        results = self._hands.process(prepare_frame(frame_bgr, self.upscale_short_side_below_pixels))
        if not results.multi_hand_landmarks:
            return np.zeros((0, HAND_LANDMARK_COUNT, 3), dtype=np.float32)
        return np.stack([landmarks_to_array(hand, HAND_LANDMARK_COUNT, False).reshape(HAND_LANDMARK_COUNT, 3)
                         for hand in results.multi_hand_landmarks])


def describe_mediapipe_version() -> str:
    return getattr(mp, "__version__", "unknown")


def download_models_once(holistic_settings_list: list[dict], hand_settings: dict | None) -> None:
    """Create each needed model once in the main process, so MediaPipe downloads its heavy model files
    (model_complexity 2) here, before worker processes start, instead of in several workers at once."""
    for settings in holistic_settings_list:
        with HolisticExtractor(settings):
            pass
    if hand_settings is not None:
        with HandDetector(hand_settings):
            pass
