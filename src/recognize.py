# src/recognize.py

"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:

Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.

Run:
python -m src.recognize

Keys:
q : quit
r : reload DB from disk (data/db/face_db.npz)
+/- : adjust threshold (distance) live
d : toggle debug overlay

Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame). This avoids the
  “FaceMesh points not consistent with Haar box” problem and enables multi-face.
- DB is expected from enroll: data/db/face_db.npz (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
  Since embeddings are L2-normalized, cosine_similarity = dot(a,b).
"""

from __future__ import annotations

import argparse
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e


# Reuse your known-good alignment method (you said alignment is OK now)
from .haar_5pt import align_face_5pt


# -------------------------
# Data
# -------------------------


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Math helpers
# -------------------------


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)

    return float(np.dot(a, b))


def cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    W: int,
    H: int,
) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.

    kps: (5,2) in full-frame coords
    """

    k = kps.astype(np.float32)

    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w

    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


def _kps_span_ok(
    kps: np.ndarray,
    min_eye_dist: float,
) -> bool:
    """
    Minimal geometry sanity:

    - eyes not collapsed
    - mouth generally below nose
    """

    k = kps.astype(np.float32)

    le, re, no, lm, rm = k

    eye_dist = float(
        np.linalg.norm(re - le)
    )

    if eye_dist < float(min_eye_dist):
        return False

    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False

    return True


# -------------------------
# DB helpers
# -------------------------


def load_db_npz(
    db_path: Path,
) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}

    data = np.load(
        str(db_path),
        allow_pickle=True,
    )

    out: Dict[str, np.ndarray] = {}

    for k in data.files:
        out[k] = np.asarray(
            data[k],
            dtype=np.float32,
        ).reshape(-1)

    return out


# -------------------------
# Embedder (same as embed_new)
# -------------------------


class ArcFaceEmbedderONNX:
    """
    ArcFace-style ONNX embedder.

    Input: 112x112 BGR -> internally RGB + (x-127.5)/128, NCHW float32.
    Output: (1,D) or (D,)
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.model_path = model_path

        self.in_w, self.in_h = (
            int(input_size[0]),
            int(input_size[1]),
        )

        self.debug = bool(debug)

        self.sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.in_name = (
            self.sess.get_inputs()[0].name
        )

        self.out_name = (
            self.sess.get_outputs()[0].name
        )

        if self.debug:
            print(
                "[embed] model:",
                model_path,
            )

            print(
                "[embed] input:",
                self.sess.get_inputs()[0].name,
                self.sess.get_inputs()[0].shape,
                self.sess.get_inputs()[0].type,
            )

            print(
                "[embed] output:",
                self.sess.get_outputs()[0].name,
                self.sess.get_outputs()[0].shape,
                self.sess.get_outputs()[0].type,
            )

    def _preprocess(
        self,
        aligned_bgr_112: np.ndarray,
    ) -> np.ndarray:
        img = aligned_bgr_112

        if (
            img.shape[1] != self.in_w
            or img.shape[0] != self.in_h
        ):
            img = cv2.resize(
                img,
                (self.in_w, self.in_h),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

        rgb = (rgb - 127.5) / 128.0

        x = np.transpose(
            rgb,
            (2, 0, 1),
        )[None, ...]

        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(
        v: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        v = (
            v.astype(np.float32)
            .reshape(-1)
        )

        n = float(
            np.linalg.norm(v) + eps
        )

        return (v / n).astype(np.float32)

    def embed(
        self,
        aligned_bgr_112: np.ndarray,
    ) -> np.ndarray:
        x = self._preprocess(
            aligned_bgr_112
        )

        y = self.sess.run(
            [self.out_name],
            {self.in_name: x},
        )[0]

        emb = np.asarray(
            y,
            dtype=np.float32,
        ).reshape(-1)

        return self._l2_normalize(emb)


# -------------------------
# Face Detectors: YuNet (fast ONNX DNN) and Haar (legacy)
# -------------------------


class YuNetFaceDetector:
    """
    Fast, CPU-friendly YuNet face detector using OpenCV DNN (cv2.FaceDetectorYN).
    Directly predicts bounding boxes and 5 facial landmarks (right eye, left eye,
    nose tip, right mouth corner, left mouth corner) on CPU in a single pass.
    """

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        conf_thresh: float = 0.60,
        nms_thresh: float = 0.30,
        top_k: int = 5000,
        input_size: Tuple[int, int] = (320, 320),
        debug: bool = False,
    ):
        self.model_path = str(model_path)
        self.conf_thresh = float(conf_thresh)
        self.nms_thresh = float(nms_thresh)
        self.top_k = int(top_k)
        self.debug = bool(debug)
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YuNet ONNX model not found at {self.model_path}."
            )

        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (self.in_w, self.in_h),
            score_threshold=self.conf_thresh,
            nms_threshold=self.nms_thresh,
            top_k=self.top_k,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 5,
    ) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        if (W, H) != (self.in_w, self.in_h):
            self.detector.setInputSize((W, H))
            self.in_w, self.in_h = W, H

        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return []

        # Sort by area descending
        faces = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)[:max_faces]
        out: List[FaceDet] = []

        for f in faces:
            w, h = float(f[2]), float(f[3])
            score = float(f[14])

            # Landmarks from YuNet:
            # f[4:6]: right eye (subject's right = image left)
            # f[6:8]: left eye (subject's left = image right)
            # f[8:10]: nose tip
            # f[10:12]: right mouth corner
            # f[12:14]: left mouth corner
            kps = np.array([
                [f[4], f[5]],
                [f[6], f[7]],
                [f[8], f[9]],
                [f[10], f[11]],
                [f[12], f[13]],
            ], dtype=np.float32)

            # Enforce image-space left-to-right eye ordering (smaller x first)
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            if not _kps_span_ok(kps, min_eye_dist=max(8.0, 0.15 * float(w))):
                if self.debug:
                    print("[YuNet] 5pt geometry sanity failed -> skip")
                continue

            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)

            out.append(
                FaceDet(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    kps=kps.astype(np.float32),
                )
            )

        return out


class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = bool(debug)

        self.min_size = tuple(
            map(int, min_size)
        )

        if haar_xml is None:
            haar_xml = (
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )

        self.face_cascade = (
            cv2.CascadeClassifier(haar_xml)
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade: {haar_xml}"
            )

        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe==0.10.21"
            )

        # IMPORTANT: we run FaceMesh on ROI (one face per ROI), so max_num_faces=1
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 5pt indices (same as your working file)
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )

        if faces is None or len(faces) == 0:
            return np.zeros(
                (0, 4),
                dtype=np.int32,
            )

        return faces.astype(
            np.int32
        )  # (x,y,w,h)

    def _roi_facemesh_5pt(
        self,
        roi_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]

        if H < 20 or W < 20:
            return None

        rgb = cv2.cvtColor(
            roi_bgr,
            cv2.COLOR_BGR2RGB,
        )

        res = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            return None

        lm = (
            res.multi_face_landmarks[0]
            .landmark
        )

        idxs = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        pts = []

        for i in idxs:
            p = lm[i]
            pts.append(
                [p.x * W, p.y * H]
            )

        kps = np.array(
            pts,
            dtype=np.float32,
        )

        # enforce left/right ordering
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]

        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 5,
    ) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        faces = self._haar_faces(gray)

        if faces.shape[0] == 0:
            return []

        # sort by area desc, keep top max_faces
        areas = faces[:, 2] * faces[:, 3]

        order = np.argsort(areas)[::-1]

        faces = faces[order][:max_faces]

        out: List[FaceDet] = []

        for (x, y, w, h) in faces:
            # expand ROI a bit for FaceMesh stability
            mx, my = (
                0.25 * w,
                0.35 * h,
            )

            rx1, ry1, rx2, ry2 = _clip_xyxy(
                x - mx,
                y - my,
                x + w + mx,
                y + h + my,
                W,
                H,
            )

            roi = frame_bgr[
                ry1:ry2,
                rx1:rx2,
            ]

            kps_roi = self._roi_facemesh_5pt(
                roi
            )

            if kps_roi is None:
                if self.debug:
                    print(
                        "[recognize] FaceMesh none for ROI -> skip"
                    )

                continue

            # map ROI kps back to full-frame coords
            kps = kps_roi.copy()

            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)

            # sanity: eye distance relative to Haar width
            if not _kps_span_ok(
                kps,
                min_eye_dist=max(
                    10.0,
                    0.18 * float(w),
                ),
            ):
                if self.debug:
                    print(
                        "[recognize] 5pt geometry failed -> skip"
                    )

                continue

            # build bbox from kps (centered)
            bb = _bbox_from_5pt(
                kps,
                pad_x=0.55,
                pad_y_top=0.85,
                pad_y_bot=1.15,
            )

            x1, y1, x2, y2 = _clip_xyxy(
                bb[0],
                bb[1],
                bb[2],
                bb[3],
                W,
                H,
            )

            out.append(
                FaceDet(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=1.0,
                    kps=kps.astype(
                        np.float32
                    ),
                )
            )

        return out


# -------------------------
# Matcher
# -------------------------


class FaceDBMatcher:
    def __init__(
        self,
        db: Dict[str, np.ndarray],
        dist_thresh: float = 0.30,
    ):
        self.db = db
        # Auto-handle threshold as either similarity (e.g. 0.70 or 70%) or distance (0.30)
        t = float(dist_thresh)
        if t > 1.0:
            t = t / 100.0
        if t > 0.5:
            self.sim_thresh = t
            self.dist_thresh = 1.0 - t
        else:
            self.dist_thresh = t
            self.sim_thresh = 1.0 - t

        # pre-stack for speed
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None

        self._rebuild()

    def _rebuild(self):
        self._names = sorted(
            self.db.keys()
        )

        if self._names:
            self._mat = np.stack(
                [
                    self.db[n]
                    .reshape(-1)
                    .astype(np.float32)
                    for n in self._names
                ],
                axis=0,
            )
            # (K,D)

        else:
            self._mat = None

    def reload_from(
        self,
        path: Path,
    ):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(
        self,
        emb: np.ndarray,
    ) -> MatchResult:
        if (
            self._mat is None
            or len(self._names) == 0
        ):
            return MatchResult(
                name=None,
                distance=1.0,
                similarity=0.0,
                accepted=False,
            )

        e = (
            emb.reshape(1, -1)
            .astype(np.float32)
        )
        # (1,D)

        # cosine similarity since both sides are normalized: sim = dot
        sims = (
            self._mat @ e.T
        ).reshape(-1)
        # (K,)

        best_i = int(
            np.argmax(sims)
        )

        best_sim = float(
            sims[best_i]
        )

        best_dist = 1.0 - best_sim

        # Accepted if similarity >= threshold (or distance <= threshold)
        ok = (
            best_sim >= self.sim_thresh
        )

        return MatchResult(
            name=(
                self._names[best_i]
                if ok
                else None
            ),
            distance=float(
                best_dist
            ),
            similarity=float(
                best_sim
            ),
            accepted=bool(ok),
        )


# -------------------------
# Demo
# -------------------------


def main():
    parser = argparse.ArgumentParser(description="Multi-face recognition using YuNet/Haar + ArcFace ONNX")
    parser.add_argument("--detector", choices=["yunet", "haar"], default="yunet", help="Face detector (default: yunet)")
    parser.add_argument("--yunet-model", default="models/face_detection_yunet_2023mar.onnx", help="YuNet ONNX model path")
    parser.add_argument("--model", default="models/embedder_arcface.onnx", help="ArcFace ONNX model path")
    parser.add_argument("--db", default="data/db/face_db.npz", help="Enrolled face DB path")
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity threshold e.g. 0.70 (70%) or distance 0.30")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.detector == "yunet" and Path(args.yunet_model).exists():
        print(f"[recognize] Using YuNet detector ({args.yunet_model})")
        det = YuNetFaceDetector(model_path=args.yunet_model, debug=False)
    else:
        if args.detector == "yunet":
            print(f"[recognize] Warning: YuNet model not found at {args.yunet_model}, falling back to HaarFaceMesh5pt")
        det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)

    embedder = ArcFaceEmbedderONNX(
        model_path=args.model,
        input_size=(112, 112),
        debug=False,
    )

    db = load_db_npz(db_path)

    matcher = FaceDBMatcher(
        db=db,
        dist_thresh=args.threshold,
    )

    # Open camera with fallback (try requested index, then try 0, 1)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        fallback_idx = 1 if args.camera == 0 else 0
        print(f"[recognize] Camera {args.camera} failed, trying camera {fallback_idx}...")
        cap = cv2.VideoCapture(fallback_idx)
    if not cap.isOpened():
        raise RuntimeError("No working camera found. Check webcam connection.")

    print(
        f"Recognize (multi-face). Detector={args.detector.upper()} | "
        f"Thresh={matcher.sim_thresh*100:.0f}% (dist<={matcher.dist_thresh:.2f})\n"
        "Keys: q=quit, r=reload DB, +/- threshold, d=debug overlay"
    )

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None

    show_debug = False

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        faces = det.detect(
            frame,
            max_faces=5,
        )

        vis = frame.copy()

        # compute fps
        frames += 1

        dt = time.time() - t0

        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        # draw + recognize each face
        # show aligned thumbnails stacked on the RIGHT,
        # but lower to avoid overlay with green text
        h, w = vis.shape[:2]

        thumb = 112
        pad = 8

        x0 = w - thumb - pad
        y0 = 80  # moved down to avoid your text overlay area

        shown = 0

        for i, f in enumerate(faces):
            # draw bbox + kps
            cv2.rectangle(
                vis,
                (f.x1, f.y1),
                (f.x2, f.y2),
                (0, 255, 0),
                2,
            )

            for (x, y) in f.kps.astype(int):
                cv2.circle(
                    vis,
                    (int(x), int(y)),
                    2,
                    (0, 255, 0),
                    -1,
                )

            # align -> embed -> match
            aligned, _ = align_face_5pt(
                frame,
                f.kps,
                out_size=(112, 112),
            )

            emb = embedder.embed(
                aligned
            )

            mr = matcher.match(
                emb
            )

            # label
            label = (
                mr.name
                if mr.name is not None
                else "Unknown"
            )

            line1 = f"{label}"

            line2 = (
                f"dist={mr.distance:.3f} "
                f"sim={mr.similarity:.3f}"
            )

            # color: known green, unknown red
            color = (
                (0, 255, 0)
                if mr.accepted
                else (0, 0, 255)
            )

            cv2.putText(
                vis,
                line1,
                (
                    f.x1,
                    max(0, f.y1 - 28),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.putText(
                vis,
                line2,
                (
                    f.x1,
                    max(0, f.y1 - 6),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # aligned preview thumbnails (stack)
            if (
                y0 + thumb <= h
                and shown < 4
            ):
                vis[
                    y0:y0 + thumb,
                    x0:x0 + thumb,
                ] = aligned

                cv2.putText(
                    vis,
                    f"{i + 1}:{label}",
                    (
                        x0,
                        y0 - 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

                y0 += thumb + pad
                shown += 1

            if show_debug:
                # show kps coords quickly
                dbg = (
                    f"kpsLeye="
                    f"({f.kps[0, 0]:.0f},"
                    f"{f.kps[0, 1]:.0f})"
                )

                cv2.putText(
                    vis,
                    dbg,
                    (
                        10,
                        h - 20,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # overlay header
        header = (
            f"IDs={len(matcher._names)} "
            f"thr(dist)={matcher.dist_thresh:.2f}"
        )

        if fps is not None:
            header += f" fps={fps:.1f}"

        cv2.putText(
            vis,
            header,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "recognize_new",
            vis,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            matcher.reload_from(
                db_path
            )

            print(
                f"[recognize] reloaded DB: "
                f"{len(matcher._names)} identities"
            )

        elif key in (
            ord("+"),
            ord("="),
        ):
            matcher.dist_thresh = float(
                min(
                    1.20,
                    matcher.dist_thresh + 0.01,
                )
            )

            print(
                f"[recognize] thr(dist)="
                f"{matcher.dist_thresh:.2f} "
                f"(sim~{1.0 - matcher.dist_thresh:.2f})"
            )

        elif key == ord("-"):
            matcher.dist_thresh = float(
                max(
                    0.05,
                    matcher.dist_thresh - 0.01,
                )
            )

            print(
                f"[recognize] thr(dist)="
                f"{matcher.dist_thresh:.2f} "
                f"(sim~{1.0 - matcher.dist_thresh:.2f})"
            )

        elif key == ord("d"):
            show_debug = not show_debug

            print(
                f"[recognize] debug overlay: "
                f"{'ON' if show_debug else 'OFF'}"
            )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()