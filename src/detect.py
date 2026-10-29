# src/detect.py

import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Face detection demo using YuNet (or Haar)")
    parser.add_argument("--detector", choices=["yunet", "haar"], default="yunet", help="Face detector (default: yunet)")
    parser.add_argument("--yunet-model", default="models/face_detection_yunet_2023mar.onnx", help="YuNet ONNX model path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    use_yunet = args.detector == "yunet" and Path(args.yunet_model).exists()

    if use_yunet:
        print(f"[detect] Using YuNet detector ({args.yunet_model})")
        yunet = cv2.FaceDetectorYN.create(
            args.yunet_model,
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        current_size = (320, 320)
    else:
        print("[detect] Using Haar cascade detector")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        fallback_idx = 1 if args.camera == 0 else 0
        print(f"[detect] Camera {args.camera} failed, trying camera {fallback_idx}...")
        cap = cv2.VideoCapture(fallback_idx)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Check webcam connection.")

    print(f"Face detect running ({'YuNet' if use_yunet else 'Haar'}). Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        if use_yunet:
            if (W, H) != current_size:
                yunet.setInputSize((W, H))
                current_size = (W, H)
            _, faces = yunet.detect(frame)
            if faces is not None:
                for f in faces:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    score = float(f[14])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"YuNet: {score:.2f}",
                        (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    # 5 Landmarks: right eye, left eye, nose tip, right mouth, left mouth
                    kps = [
                        (int(f[4]), int(f[5])),   # eye R
                        (int(f[6]), int(f[7])),   # eye L
                        (int(f[8]), int(f[9])),   # nose
                        (int(f[10]), int(f[11])), # mouth R
                        (int(f[12]), int(f[13])), # mouth L
                    ]
                    for pt in kps:
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Haar",
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Face Detection (YuNet/Haar)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()