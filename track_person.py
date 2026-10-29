"""
Whole-person tracking + face association, live on your webcam (CPU only).

    python track_person.py --target Christella               # Full tracking (YuNet + YOLOX + ArcFace)
    python track_person.py --target Christella --fullscreen   # Start directly in fullscreen!

Keys:
    q  quit                      r  reset identity
    f  toggle FULLSCREEN mode    v  toggle Virtual Camera PIP
    e  enrol target (simple backend only; look at the camera for ~2 s)
    Ground-truth labels for the evaluation log (press while the event happens):
    1  normal    2  head turn    3  occlusion    4  crossing
    w  toggle "tracker is on the WRONG person"
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.centering import CenteringController, VirtualCamera
from src.identity_state import IdState
from src.person_tracker import HogPersonDetector, UpperBodyDetector, YoloOnnxPersonDetector
from src.runlog import RunLogger
from src.whole_person import PipelineConfig, WholePersonPipeline

SCENARIOS = {ord("1"): "normal", ord("2"): "head_turn", ord("3"): "occlusion", ord("4"): "crossing"}
STATE_COLOR = {
    IdState.CONFIRMED: (0, 220, 0),
    IdState.TEMPORARY: (50, 230, 100),
    IdState.UNKNOWN: (60, 60, 255),
}


def text(img, s, xy, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_backends(args):
    if args.backend == "arcface":
        from src.backends import ArcFaceBackend

        b = ArcFaceBackend(
            model_path=args.model,
            db_path=args.db,
            threshold=args.threshold,
            face_detector=args.face_detector,
            yunet_path=args.yunet_model,
        )
        return b, b, None
    from src.backends import OpenCVHaarFaces, OpenCVYuNetFaces, SimpleIdentifier

    if args.face_detector == "yunet" and Path(args.yunet_model).exists():
        fdet = OpenCVYuNetFaces(args.yunet_model)
    else:
        fdet = OpenCVHaarFaces()
    ident = SimpleIdentifier(args.target)
    return fdet, ident, ident


def build_person_detector(args):
    if args.person_detector == "yolox":
        from src.person_tracker import YoloxPersonDetector

        return YoloxPersonDetector(args.yolox_model)
    if args.person_detector == "yolo":
        return YoloOnnxPersonDetector(args.yolo_model)
    if args.person_detector == "hog":
        return HogPersonDetector()
    if args.person_detector == "upper":
        return UpperBodyDetector()
    return None


def draw(frame, res, cmd, cam, scenario, wrong, target, info_text="", show_pip=True):
    view = frame.copy()
    H, W = view.shape[:2]

    for tv in res.tracks:
        x1, y1, x2, y2 = (int(v) for v in tv.box)
        if tv.is_target:
            if res.state == IdState.CONFIRMED:
                col = (0, 220, 0)
                status_label = "FACE VERIFIED"
            elif res.state == IdState.TEMPORARY:
                col = (50, 230, 100)  # Bright cyan-green (stays locked even with back turned)
                status_label = "BODY TRACKED (FACE AWAY)"
            else:
                col = (60, 60, 255)
                status_label = "UNKNOWN"
            thick = 3
            tag = f"{target} #{tv.id} [{status_label}]"
        else:
            col, thick = (160, 160, 160), 1
            tag = f"#{tv.id}" + (" (coasting)" if not tv.visible else "")
        if tv.entangled:
            tag += " [OVERLAP]"

        cv2.rectangle(view, (x1, y1), (x2, y2), col, thick)
        text(view, tag, (x1 + 4, max(18, y1 - 8)), col, 0.6)
        cv2.circle(view, (int(tv.head[0]), int(tv.head[1])), 4, col, -1)

    for fb in res.face_boxes:
        cv2.rectangle(view, (int(fb[0]), int(fb[1])), (int(fb[2]), int(fb[3])), (255, 200, 0), 2)

    # State banner
    if res.state == IdState.CONFIRMED:
        state_col = (0, 220, 0)
        status_str = "CONFIRMED"
    elif res.state == IdState.TEMPORARY:
        state_col = (50, 230, 100)
        status_str = "BODY TRACKED"
    else:
        state_col = (60, 60, 255)
        status_str = "UNKNOWN"

    header = f"{status_str}: {target}   conf={res.conf:.2f}"
    if info_text:
        header += f" | {info_text}"
    text(view, header, (12, 28), state_col, 0.75)

    # Confidence bar
    cv2.rectangle(view, (12, 38), (212, 50), (90, 90, 90), 1)
    cv2.rectangle(view, (13, 39), (13 + int(198 * res.conf), 49), state_col, -1)

    # Motor centering command info
    text(view, f"H={cmd.horizontal} V={cmd.vertical} err=({cmd.error_x:+.2f},{cmd.error_y:+.2f}) gain={cmd.gain:.2f}",
         (12, 72), (255, 170, 0), 0.55)

    # Footer controls
    text(view, f"label: {scenario}  ['f'=Fullscreen | 'v'=PIP Cam | 'r'=Reset | 'q'=Quit]" + ("   [WRONG TARGET FLAGGED]" if wrong else ""),
         (12, H - 14), (0, 0, 255) if wrong else (230, 230, 230), 0.52)

    # Virtual camera tracking box outline
    x1, y1, x2, y2 = cam.rect()
    cv2.rectangle(view, (x1, y1), (x2, y2), (200, 120, 255), 2)

    # Picture-in-Picture virtual camera in top-right corner
    if show_pip and cam is not None:
        vcam = cam.render(frame)
        vh, vw = vcam.shape[:2]
        pip_h = int(H * 0.28)
        pip_w = int(pip_h * (vw / vh))
        pip_img = cv2.resize(vcam, (pip_w, pip_h))
        px = W - pip_w - 14
        py = 14
        cv2.rectangle(view, (px - 2, py - 2), (px + pip_w + 2, py + pip_h + 2), (200, 120, 255), 2)
        view[py:py + pip_h, px:px + pip_w] = pip_img
        text(view, "Virtual Cam (Motors)", (px + 6, py + 18), (200, 120, 255), 0.45)

    return view


def main():
    ap = argparse.ArgumentParser(description="Whole-person tracking with YuNet/Haar and YoloX")
    ap.add_argument("--target", required=True, help="Enrolled name to track (e.g. Christella)")
    ap.add_argument("--backend", choices=["arcface", "simple"], default="arcface")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--source", default=None, help="video file instead of the webcam")
    ap.add_argument("--person-detector", choices=["yolox", "yolo", "hog", "upper", "none"], default="yolox",
                    help="Person detector (default: yolox)")
    ap.add_argument("--yolox-model", default="models/yolox_nano.onnx", help="YOLOX model path (default: models/yolox_nano.onnx)")
    ap.add_argument("--yolo-model", default="models/yolov8n.onnx")
    ap.add_argument("--face-detector", choices=["yunet", "haar"], default="yunet",
                    help="Face detector (default: yunet)")
    ap.add_argument("--yunet-model", default="models/face_detection_yunet_2023mar.onnx",
                    help="YuNet ONNX model path")
    ap.add_argument("--model", default="models/embedder_arcface.onnx")
    ap.add_argument("--db", default="data/db/face_db.npz")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="Verification threshold (e.g. 0.70 for 70%% similarity, or 0.30 distance)")
    ap.add_argument("--det-every", type=int, default=3)
    ap.add_argument("--verify-every", type=int, default=8)
    ap.add_argument("--mismatch-below", type=float, default=0.20,
                    help="similarity below this (and not accepted) counts as a clear non-match")
    ap.add_argument("--tau", type=float, default=30.0,
                    help="confidence decay time constant in seconds (default: 30.0s for generous body-turn persistence)")
    ap.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode")
    ap.add_argument("--log", default=None, help="CSV run log (default logs/run_<time>.csv)")
    ap.add_argument("--mqtt", action="store_true", help="also publish pan/tilt over MQTT")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    face_det, identifier, simple = build_backends(args)
    person_det = build_person_detector(args)

    cfg = PipelineConfig(det_every=args.det_every, verify_every=args.verify_every,
                         mismatch_sim_ceiling=args.mismatch_below)
    cfg.identity.tau = args.tau
    pipe = WholePersonPipeline(args.target, face_det, identifier, person_detector=person_det, cfg=cfg)

    # Open camera with fallback (try requested camera, then alternative 0/1)
    cam_src = args.source if args.source else args.camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened() and not args.source:
        fallback_idx = 1 if args.camera == 0 else 0
        print(f"[camera] Index {args.camera} failed, trying index {fallback_idx}...")
        cap = cv2.VideoCapture(fallback_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Video source '{cam_src}' not available. Check camera index or connection.")

    # Request high resolution from webcam if available
    if not args.source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read a frame from video source")
    H, W = frame.shape[:2]
    video_fps = cap.get(cv2.CAP_PROP_FPS) if args.source else 0.0

    ctrl = CenteringController()
    cam = VirtualCamera(W, H)
    log_path = args.log or f"logs/run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log = RunLogger(log_path, ctrl.dead_zone)
    print(f"[log] {log_path}")

    mqtt = None
    if args.mqtt:
        from src.mqtt_controller import MQTTController

        mqtt = MQTTController()
        mqtt.connect()

    WIN_NAME = "Whole-Person Tracking - YuNet + YOLOX"
    if not args.no_gui:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WIN_NAME, 1280, 720)

    is_fullscreen = args.fullscreen
    show_pip = True
    scenario, wrong = "normal", 0
    enrolling = 0
    t0 = time.monotonic()
    t_prev = 0.0
    n = 0
    try:
        while ok:
            n += 1
            t = (n - 1) / video_fps if (args.source and video_fps > 0) else time.monotonic() - t0
            res = pipe.process(frame, t)

            err = cam.error_to(res.target_point) if res.target_point is not None else None
            cmd = ctrl.step(err, res.state, res.conf)
            cam.apply(cmd, max(1e-3, t - t_prev))
            t_prev = t
            if mqtt is not None:
                mqtt.publish_tracking(cmd.horizontal, cmd.vertical) if cmd.active else mqtt.stop()

            log.write(res, cmd, scenario, wrong)

            if simple is not None and enrolling > 0:
                faces = face_det.detect(frame)
                if faces:
                    big = max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
                    simple.add_sample(frame, big)
                enrolling -= 1
                if enrolling == 0:
                    print("[enrol] saved" if simple.finish_enrollment() else "[enrol] not enough faces seen")

            if not args.no_gui:
                view = draw(frame, res, cmd, cam, scenario, wrong, args.target, show_pip=show_pip)
                if enrolling > 0:
                    text(view, f"ENROLLING {enrolling}", (12, 100), (0, 255, 255), 0.7)
                cv2.imshow(WIN_NAME, view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)
                    if not is_fullscreen:
                        cv2.resizeWindow(WIN_NAME, 1280, 720)
                elif key in (ord("v"), ord("V")):
                    show_pip = not show_pip
                elif key in SCENARIOS:
                    scenario = SCENARIOS[key]
                elif key == ord("w"):
                    wrong = 1 - wrong
                elif key == ord("r"):
                    pipe.identity.reset()
                    ctrl.reset()
                elif key == ord("e") and simple is not None:
                    enrolling = 40
                    print("[enrol] look at the camera...")

            if args.max_frames and n >= args.max_frames:
                break
            ok, frame = cap.read()
    finally:
        log.close()
        if mqtt is not None:
            mqtt.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[done] {n} frames -> {log_path}\n       python evaluate_run.py {log_path}")


if __name__ == "__main__":
    main()
