import argparse
import os
import time
from typing import Tuple

import cv2
import numpy as np
from joblib import load


def preprocess_face(gray: np.ndarray, bbox: Tuple[int, int, int, int], img_size: Tuple[int, int]) -> np.ndarray:
    x, y, w, h = bbox
    face = gray[y : y + h, x : x + w]
    if face.size == 0:
        return None
    resized = cv2.resize(face, img_size, interpolation=cv2.INTER_AREA)
    return resized.flatten().astype(np.float32)[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam face recognition with PCA+ANN")
    parser.add_argument("--model", required=True, help="Path to .joblib model artifact")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--min_neighbors", type=int, default=5, help="Haar minNeighbors")
    parser.add_argument("--scale_factor", type=float, default=1.1, help="Haar scaleFactor")
    parser.add_argument("--min_size", type=int, default=60, help="Minimum detected face size in pixels")
    args = parser.parse_args()

    artifact = load(args.model)
    model = artifact["model"]
    class_names = artifact["class_names"]
    img_size = artifact["img_size"]

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    # Prefer DirectShow on Windows for better compatibility; fall back to MSMF
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    # Optional: set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "PCA+ANN Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Show message if no frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "No frames from camera...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, blank)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_size, args.min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for (x, y, w, h) in faces:
                X = preprocess_face(gray, (x, y, w, h), img_size)
                if X is None:
                    continue
                y_pred = model.predict(X)[0]
                label = class_names[y_pred] if 0 <= y_pred < len(class_names) else str(y_pred)
                conf_text = ""
                if hasattr(model.named_steps["mlp"], "predict_proba"):
                    try:
                        proba = model.predict_proba(X)[0]
                        conf = float(np.max(proba))
                        conf_text = f" {conf:.2f}"
                    except Exception:
                        conf_text = ""

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label}{conf_text}",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # If no faces, show hint
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # FPS overlay
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                last_fps_time = now
                frame_count = 0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

