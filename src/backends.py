"""
Real (webcam) backends for the pipeline interfaces.

ArcFaceBackend   Wraps YOUR existing week-1/2 modules (src.recognize,
                 src.align). It uses exactly the calls from your
                 locked-face tracker, so enrolment and thresholds are the
                 same as before.
SimpleBackend    Stand-alone OpenCV Haar face detector + a weak HOG-descriptor
                 face matcher with built-in enrolment. It exists so the
                 module runs without your models; it is NOT reliable enough
                 for a serious evaluation. Use ArcFace for that.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .whole_person import FaceObs, IdMatch


# ----------------------------------------------------------------- ArcFace path
class ArcFaceBackend:
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        db_path: str = "data/db/face_db.npz",
        threshold: float = 0.70,
        face_detector: str = "yunet",
        yunet_path: str = "models/face_detection_yunet_2023mar.onnx",
        min_face: int = 70,
    ):
        # imported lazily: these live in your repo, not in this module
        from src.align import align_face_5pt
        from src.recognize import (
            ArcFaceEmbedderONNX,
            FaceDBMatcher,
            HaarFaceMesh5pt,
            YuNetFaceDetector,
            load_db_npz,
        )

        self._align = align_face_5pt
        if face_detector == "yunet" and Path(yunet_path).exists():
            print(f"[backend] Face detector: YuNet ({yunet_path})")
            self._detector = YuNetFaceDetector(model_path=yunet_path, conf_thresh=0.60, debug=False)
        else:
            if face_detector == "yunet":
                print(f"[backend] Notice: YuNet model not found at {yunet_path}, falling back to Haar")
            self._detector = HaarFaceMesh5pt(min_size=(min_face, min_face), debug=False)

        self._embedder = ArcFaceEmbedderONNX(model_path=model_path, input_size=(112, 112), debug=False)
        self._matcher = FaceDBMatcher(load_db_npz(Path(db_path)), dist_thresh=threshold)

    # face detector interface
    def detect(self, frame) -> List[FaceObs]:
        faces = self._detector.detect(frame, max_faces=8)
        return [FaceObs((float(f.x1), float(f.y1), float(f.x2), float(f.y2)), raw=f) for f in faces]

    # identifier interface
    def identify(self, frame, face: FaceObs) -> IdMatch:
        aligned, _ = self._align(frame, face.raw.kps, out_size=(112, 112))
        m = self._matcher.match(self._embedder.embed(aligned))
        return IdMatch(getattr(m, "name", None), float(m.similarity), bool(m.accepted))


# ------------------------------------------------------------------ simple path
class OpenCVYuNetFaces:
    def __init__(self, model_path: str = "models/face_detection_yunet_2023mar.onnx", conf_thresh: float = 0.60):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold=conf_thresh, nms_threshold=0.30
        )
        self._size = (320, 320)

    def detect(self, frame) -> List[FaceObs]:
        H, W = frame.shape[:2]
        if (W, H) != self._size:
            self.detector.setInputSize((W, H))
            self._size = (W, H)
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        return [FaceObs((float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3]))) for f in faces]


class OpenCVHaarFaces:
    def __init__(self, min_size: int = 70, scale_width: int = 480):
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Could not load {path}. Install opencv-python (not -headless) or point "
                "cv2.data.haarcascades at a folder with haarcascade_frontalface_default.xml."
            )
        self.min_size = min_size
        self.scale_width = scale_width

    def detect(self, frame) -> List[FaceObs]:
        h, w = frame.shape[:2]
        s = self.scale_width / float(w) if w > self.scale_width else 1.0
        small = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        rects = self.cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(int(self.min_size * s), int(self.min_size * s))
        )
        return [FaceObs((x / s, y / s, (x + rw) / s, (y + rh) / s)) for (x, y, rw, rh) in rects]


def _face_descriptor(frame, box) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(cv2.resize(gray, (64, 64)))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    d = hog.compute(gray).reshape(-1).astype(np.float32)
    return d / (np.linalg.norm(d) + 1e-9)


class SimpleIdentifier:
    """Cosine similarity to the mean enrolled descriptor. Weak but self-contained."""

    def __init__(self, target_name: str, enroll_path: str = "data/simple_enroll.npz", accept: float = 0.80):
        self.target_name = target_name
        self.enroll_path = enroll_path
        self.accept = accept
        self.samples: List[np.ndarray] = []
        self.mean: Optional[np.ndarray] = None
        if os.path.exists(enroll_path):
            self.mean = np.load(enroll_path)["mean"]

    @property
    def enrolled(self) -> bool:
        return self.mean is not None

    def add_sample(self, frame, face: FaceObs) -> int:
        d = _face_descriptor(frame, face.box)
        if d is not None:
            self.samples.append(d)
        return len(self.samples)

    def finish_enrollment(self) -> bool:
        if len(self.samples) < 5:
            return False
        m = np.mean(self.samples, axis=0)
        self.mean = m / (np.linalg.norm(m) + 1e-9)
        Path(self.enroll_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.enroll_path, mean=self.mean)
        self.samples = []
        return True

    def identify(self, frame, face: FaceObs) -> IdMatch:
        d = _face_descriptor(frame, face.box)
        if d is None or self.mean is None:
            return IdMatch(None, 0.0, False)
        sim = float(np.dot(d, self.mean))
        ok = sim >= self.accept
        return IdMatch(self.target_name if ok else None, sim, ok)
