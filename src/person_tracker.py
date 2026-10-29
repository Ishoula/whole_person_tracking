"""
Whole-person detection and multi-person body tracking (CPU only).

Detectors (all return a list of Detection):
  * HogPersonDetector      OpenCV's built-in HOG+SVM pedestrian detector.
  * UpperBodyDetector      Haar upper-body cascade (better for a laptop webcam
                           where only head and shoulders are visible).
  * YoloOnnxPersonDetector Optional YOLOv8 ONNX model run through cv2.dnn.

PersonTracker keeps one PersonTrack per body. Tracks coast on a damped
velocity when the body is not detected, and carry:
  * head      : best estimate of the head point (used for centring)
  * hist      : HSV colour histogram of the torso (appearance cue)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]


# --------------------------------------------------------------------- geometry
def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def box_center(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)


def box_diag(b: Sequence[float]) -> float:
    return max(1.0, math.hypot(b[2] - b[0], b[3] - b[1]))


def point_in_head_zone(p: Sequence[float], b: Sequence[float], frac: float = 0.45) -> bool:
    """Is point p inside the upper `frac` of box b (where a head should be)?"""
    h = b[3] - b[1]
    return b[0] <= p[0] <= b[2] and (b[1] - 0.10 * h) <= p[1] <= (b[1] + frac * h)


def face_to_body_box(face: Sequence[float], frame_shape) -> Box:
    """Estimate a head-and-torso box from a face box (webcam close-up case)."""
    H, W = frame_shape[:2]
    fw, fh = face[2] - face[0], face[3] - face[1]
    cx = (face[0] + face[2]) / 2.0
    x1, x2 = cx - 1.4 * fw, cx + 1.4 * fw
    y1, y2 = face[1] - 0.4 * fh, face[1] + 6.5 * fh
    return (max(0.0, x1), max(0.0, y1), min(float(W - 1), x2), min(float(H - 1), y2))


# -------------------------------------------------------------------- detectors
@dataclass
class Detection:
    box: Box
    score: float = 1.0
    source: str = "det"


def _nms(boxes: List[Box], scores: List[float], iou_thr: float = 0.4) -> List[int]:
    if not boxes:
        return []
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], 0.0, iou_thr)
    return [int(k) for k in np.array(keep).reshape(-1)]


class HogPersonDetector:
    """OpenCV HOG pedestrian detector, run on a downscaled frame."""

    def __init__(self, scale_width: int = 400, hit_threshold: float = 0.0):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.scale_width = scale_width
        self.hit_threshold = hit_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        s = self.scale_width / float(w) if w > self.scale_width else 1.0
        small = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame
        rects, weights = self.hog.detectMultiScale(
            small, hitThreshold=self.hit_threshold, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        boxes = [(x / s, y / s, (x + rw) / s, (y + rh) / s) for (x, y, rw, rh) in rects]
        scores = [float(wt) for wt in np.array(weights).reshape(-1)] if len(rects) else []
        return [Detection(boxes[i], scores[i], "hog") for i in _nms(boxes, scores)]


class UpperBodyDetector:
    """Haar upper-body cascade. Needs haarcascade_upperbody.xml (ships with
    standard opencv-python builds under cv2.data.haarcascades)."""

    def __init__(self, scale_width: int = 480, cascade_path: Optional[str] = None):
        import os

        path = cascade_path or os.path.join(cv2.data.haarcascades, "haarcascade_upperbody.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Could not load upper-body cascade at {path}. Pass cascade_path=... "
                "or use --person-detector hog."
            )
        self.scale_width = scale_width

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        s = self.scale_width / float(w) if w > self.scale_width else 1.0
        small = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        boxes = [(x / s, y / s, (x + rw) / s, (y + rh) / s) for (x, y, rw, rh) in rects]
        return [Detection(b, 1.0, "upper") for b in boxes]


def decode_yolov8(
    output: np.ndarray, orig_w: int, orig_h: int, input_size: int = 640, conf_thr: float = 0.35
) -> List[Detection]:
    """Decode a YOLOv8 ONNX output of shape (1, 84, N) for class 0 (person).
    Input was letterboxed-free resized to input_size x input_size."""
    pred = np.squeeze(output)
    if pred.shape[0] == 84 or pred.shape[0] < pred.shape[1]:
        pred = pred.T                      # channels-first (84, N) -> (N, 84)
    boxes, scores = [], []
    sx, sy = orig_w / float(input_size), orig_h / float(input_size)
    for row in pred:
        score = float(row[4])              # class 0 = person
        if score < conf_thr:
            continue
        cx, cy, bw, bh = row[:4]
        boxes.append(((cx - bw / 2) * sx, (cy - bh / 2) * sy, (cx + bw / 2) * sx, (cy + bh / 2) * sy))
        scores.append(score)
    return [Detection(boxes[i], scores[i], "yolo") for i in _nms(boxes, scores, 0.5)]


class YoloOnnxPersonDetector:
    """YOLOv8 (e.g. yolov8n.onnx) on CPU via cv2.dnn. Decode logic is unit
    tested; end-to-end inference needs a real model file."""

    def __init__(self, model_path: str, input_size: int = 640, conf_thr: float = 0.35):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.conf_thr = conf_thr

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()
        return decode_yolov8(out, w, h, self.input_size, self.conf_thr)


def decode_yolox(
    output: np.ndarray,
    orig_w: int,
    orig_h: int,
    input_size: Tuple[int, int] = (416, 416),
    conf_thr: float = 0.35,
    strides: Sequence[int] = (8, 16, 32),
) -> List[Detection]:
    """Decode YOLOX ONNX output of shape (1, N, 85) or (N, 85) for class 0 (person)."""
    pred = np.squeeze(output)
    in_h, in_w = input_size

    # Check if raw grid offsets need to be applied
    if np.any(pred[..., 2:4] < 0):
        grids = []
        expanded_strides = []
        for s in strides:
            hsize, wsize = in_h // s, in_w // s
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((grid.shape[0], 1), s))
        all_grids = np.concatenate(grids, axis=0)
        all_strides = np.concatenate(expanded_strides, axis=0)
        decoded = np.copy(pred)
        decoded[:, :2] = (decoded[:, :2] + all_grids) * all_strides
        decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * all_strides
        pred = decoded

    boxes, scores = [], []
    sx = orig_w / float(in_w)
    sy = orig_h / float(in_h)
    for row in pred:
        score = float(row[4]) * float(row[5])  # obj_conf * cls_0_conf (person)
        if score < conf_thr:
            continue
        cx, cy, bw, bh = row[:4]
        boxes.append(((cx - bw / 2.0) * sx, (cy - bh / 2.0) * sy, (cx + bw / 2.0) * sx, (cy + bh / 2.0) * sy))
        scores.append(score)
    return [Detection(boxes[i], scores[i], "yolox") for i in _nms(boxes, scores, 0.45)]


class YoloxPersonDetector:
    """YOLOX (e.g. yolox_nano.onnx, yolox_tiny.onnx) on CPU via cv2.dnn.
    Ultra-lightweight real-time person detector."""

    def __init__(
        self,
        model_path: str = "models/yolox_nano.onnx",
        input_size: Tuple[int, int] = (416, 416),
        conf_thr: float = 0.35,
    ):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_size = input_size
        self.conf_thr = conf_thr

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (self.input_size[1], self.input_size[0]), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()
        return decode_yolox(out, w, h, self.input_size, self.conf_thr)


# ----------------------------------------------------------------------- tracks
@dataclass
class PersonTrack:
    id: int
    box: np.ndarray                      # x1,y1,x2,y2 (float32)
    head: np.ndarray                     # estimated head point
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hist: Optional[np.ndarray] = None
    last_seen_t: float = 0.0
    born_t: float = 0.0
    source: str = "det"                  # 'det' or 'face_est'
    visible: bool = True                 # matched a detection / face this frame
    face_seen_t: Optional[float] = None
    # identity cache for this body (filled by the pipeline)
    last_id_name: Optional[str] = None
    last_id_sim: float = 0.0
    last_id_t: float = -1e9
    last_id_frame: int = -10**9

    @property
    def center(self) -> np.ndarray:
        return box_center(self.box)


def torso_hist(frame: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    # clothing region: middle of the box horizontally, upper-middle vertically
    rx1, rx2 = int(max(0, x1 + 0.2 * w)), int(min(W, x2 - 0.2 * w))
    ry1, ry2 = int(max(0, y1 + 0.25 * h)), int(min(H, y1 + 0.60 * h))
    if rx2 - rx1 < 6 or ry2 - ry1 < 6:
        return None
    hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """1 - Bhattacharyya distance, in [0, 1] (1 = identical)."""
    if a is None or b is None:
        return None
    d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, min(1.0, 1.0 - d)))


class PersonTracker:
    def __init__(
        self,
        max_coast_s: float = 5.0,
        gate: float = -0.30,
        vel_alpha: float = 0.4,
        box_alpha: float = 0.6,
        vel_damp_tau: float = 0.5,
        entangle_iou: float = 0.10,
    ):
        self.max_coast_s = max_coast_s
        self.gate = gate
        self.vel_alpha = vel_alpha
        self.box_alpha = box_alpha
        self.vel_damp_tau = vel_damp_tau
        self.entangle_iou = entangle_iou

        self.tracks: Dict[int, PersonTrack] = {}
        self._next_id = 1
        self._t_prev: Optional[float] = None

    # ----------------------------------------------------------------- predict
    def predict(self, t: float) -> None:
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        for tr in self.tracks.values():
            shift = tr.vel * dt
            tr.box = tr.box + np.array([shift[0], shift[1], shift[0], shift[1]], np.float32)
            tr.head = tr.head + shift
            tr.vel = tr.vel * math.exp(-dt / self.vel_damp_tau)   # damped coasting
            tr.visible = False

    # ------------------------------------------------------------------ update
    def update(self, frame: np.ndarray, detections: List[Detection], t: float) -> None:
        """Associate full-body detections with tracks (call after predict)."""
        pairs = []
        for tid, tr in self.tracks.items():
            tr_diag = box_diag(tr.box)
            for di, d in enumerate(detections):
                score = iou(tr.box, d.box) - 0.35 * float(
                    np.linalg.norm(box_center(d.box) - tr.center) / tr_diag
                )
                if point_in_head_zone(tr.head, d.box):
                    score += 0.5
                sim = hist_similarity(tr.hist, torso_hist(frame, d.box))
                if sim is not None:
                    score += 0.25 * (sim - 0.5)
                if score > self.gate:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for score, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            self._apply_detection(frame, self.tracks[tid], detections[di], t)

        for di, d in enumerate(detections):
            if di not in used_d:
                self._spawn(frame, d.box, t, source="det")

        self._expire(t)

    def _apply_detection(self, frame, tr: PersonTrack, d: Detection, t: float) -> None:
        a = self.box_alpha
        new_box = np.array(d.box, np.float32)
        old_c = tr.center
        tr.box = a * new_box + (1 - a) * tr.box
        dt = max(1e-3, t - tr.last_seen_t)
        inst_v = (tr.center - old_c) / dt
        tr.vel = self.vel_alpha * inst_v + (1 - self.vel_alpha) * tr.vel
        # follow the body with the head point unless a face pins it
        tr.head = tr.head + (tr.center - old_c)
        tr.last_seen_t = t
        tr.visible = True
        tr.source = "det"
        self._update_hist(frame, tr)

    def _update_hist(self, frame, tr: PersonTrack) -> None:
        if self.is_entangled(tr.id):
            return                                    # don't learn from a mixed region
        h = torso_hist(frame, tr.box)
        if h is None:
            return
        tr.hist = h if tr.hist is None else 0.9 * tr.hist + 0.1 * h

    def _spawn(self, frame, box: Sequence[float], t: float, source: str) -> PersonTrack:
        b = np.array(box, np.float32)
        h = b[3] - b[1]
        tr = PersonTrack(
            id=self._next_id,
            box=b,
            head=np.array([(b[0] + b[2]) / 2.0, b[1] + 0.15 * h], np.float32),
            last_seen_t=t,
            born_t=t,
            source=source,
            visible=True,
        )
        tr.hist = torso_hist(frame, b)
        self.tracks[tr.id] = tr
        self._next_id += 1
        return tr

    def spawn_from_face(self, frame, face_box: Sequence[float], t: float) -> PersonTrack:
        tr = self._spawn(frame, face_to_body_box(face_box, frame.shape), t, source="face_est")
        tr.head = box_center(face_box)
        tr.face_seen_t = t
        return tr

    def apply_face(self, frame, tr: PersonTrack, face_box: Sequence[float], t: float) -> None:
        """A face was linked to this body: pin the head, refresh liveness."""
        fc = box_center(face_box)
        old_c = tr.center
        if tr.source == "face_est":
            # no real body detection: re-derive the body box from the face
            nb = np.array(face_to_body_box(face_box, frame.shape), np.float32)
            tr.box = self.box_alpha * nb + (1 - self.box_alpha) * tr.box
            dt = max(1e-3, t - tr.last_seen_t)
            tr.vel = self.vel_alpha * ((tr.center - old_c) / dt) + (1 - self.vel_alpha) * tr.vel
        tr.head = 0.7 * fc + 0.3 * tr.head
        tr.last_seen_t = t
        tr.face_seen_t = t
        tr.visible = True
        self._update_hist(frame, tr)

    def expire(self, t: float) -> None:
        self._expire(t)

    def _expire(self, t: float) -> None:
        dead = [tid for tid, tr in self.tracks.items() if t - tr.last_seen_t > self.max_coast_s]
        for tid in dead:
            del self.tracks[tid]

    # -------------------------------------------------------------- crossings
    def is_entangled(self, tid: int) -> bool:
        """Is this body overlapping / swallowing another person right now?"""
        tr = self.tracks.get(tid)
        if tr is None:
            return False
        for oid, other in self.tracks.items():
            if oid == tid:
                continue
            if iou(tr.box, other.box) > self.entangle_iou:
                return True
            if point_in_head_zone(other.head, tr.box, 1.0) and point_in_head_zone(tr.head, other.box, 1.0):
                return True
        return False
