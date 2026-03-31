"""Shared OpenCV YuNet + SFace runtime helpers."""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


YUNET_MODEL_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
    "face_recognition_sface_2021dec.onnx"
)


@dataclass
class FaceBox:
    raw: np.ndarray
    score: float
    width: float
    height: float


class OpenCVFaceRuntime:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        input_width: int = 320,
        input_height: int = 320,
        score_threshold: float = 0.82,
        nms_threshold: float = 0.3,
        top_k: int = 10,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
            or os.getenv("VISION_MODEL_CACHE_DIR")
            or (Path.home() / ".cache" / "noble_vision_models")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.detector_model = self._ensure_model("face_detection_yunet_2023mar.onnx", YUNET_MODEL_URL)
        self.recognizer_model = self._ensure_model("face_recognition_sface_2021dec.onnx", SFACE_MODEL_URL)
        self.detector = cv2.FaceDetectorYN_create(
            self.detector_model,
            "",
            (input_width, input_height),
            score_threshold,
            nms_threshold,
            top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(self.recognizer_model, "")

    def detect_faces(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return []

        detections: list[FaceBox] = []
        for row in faces:
            width = float(max(row[2], 0.0))
            height = float(max(row[3], 0.0))
            if width <= 0.0 or height <= 0.0:
                continue
            score = float(row[-1]) if len(row) >= 15 else 0.0
            detections.append(
                FaceBox(
                    raw=np.asarray(row, dtype=np.float32),
                    score=score,
                    width=width,
                    height=height,
                )
            )
        detections.sort(key=lambda face: face.width * face.height, reverse=True)
        return detections

    def detect_largest_face(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        faces = self.detect_faces(frame_bgr)
        return faces[0] if faces else None

    def extract_feature(self, frame_bgr: np.ndarray, face: FaceBox) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame_bgr, face.raw)
        feature = self.recognizer.feature(aligned)
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    def crop_face_jpeg(self, frame_bgr: np.ndarray, face: FaceBox, pad_ratio: float = 0.15) -> bytes:
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = face.raw[:4]
        pad_w = bw * pad_ratio
        pad_h = bh * pad_ratio
        x1 = max(0, int(x - pad_w))
        y1 = max(0, int(y - pad_h))
        x2 = min(w, int(x + bw + pad_w))
        y2 = min(h, int(y + bh + pad_h))
        crop = frame_bgr[y1:y2, x1:x2]
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("failed to encode face crop")
        return encoded.tobytes()

    def frame_to_jpeg(self, frame_bgr: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("failed to encode frame")
        return encoded.tobytes()

    @staticmethod
    def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0:
            return -1.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0.0:
            return -1.0
        return float(np.dot(left, right) / denom)

    @staticmethod
    def to_list(feature: np.ndarray) -> list[float]:
        return [float(x) for x in np.asarray(feature, dtype=np.float32).reshape(-1).tolist()]

    @staticmethod
    def from_list(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(-1)

    @staticmethod
    def default_similarity_threshold() -> float:
        # OpenCV SFace cosine threshold commonly used for same-identity matching.
        return 0.363

    def _ensure_model(self, filename: str, url: str) -> str:
        path = self.cache_dir / filename
        if path.exists() and path.stat().st_size > 0:
            return str(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(path)
        return str(path)


def feature_similarity(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float32).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom <= 0.0:
        return -1.0
    return float(np.dot(left_arr, right_arr) / denom)


def quality_from_face_box(face: FaceBox, frame_shape: tuple[int, int, int] | tuple[int, int]) -> float:
    h, w = frame_shape[:2]
    frame_area = max(1.0, float(w * h))
    face_area = max(1.0, face.width * face.height)
    score = min(1.0, face_area / (frame_area * 0.18))
    return max(0.1, score)
