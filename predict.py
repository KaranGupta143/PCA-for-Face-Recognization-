import argparse
import os
import cv2
import numpy as np
from joblib import load


def load_and_preprocess(image_path: str, img_size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    return img.flatten().astype(np.float32)[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict face identity using PCA+ANN model")
    parser.add_argument("--model", required=True, help="Path to .joblib model artifact")
    parser.add_argument("--image", required=True, help="Path to input face image")
    args = parser.parse_args()

    artifact = load(args.model)
    model = artifact["model"]
    class_names = artifact["class_names"]
    img_size = artifact["img_size"]

    X = load_and_preprocess(args.image, img_size)
    y_pred = model.predict(X)[0]
    proba = None
    if hasattr(model.named_steps["mlp"], "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None

    label = class_names[y_pred] if 0 <= y_pred < len(class_names) else str(y_pred)
    print(f"Predicted label: {label}")
    if proba is not None:
        topk = min(5, len(class_names))
        top_idx = np.argsort(proba)[::-1][:topk]
        print("Top probabilities:")
        for idx in top_idx:
            print(f"  {class_names[idx]}: {proba[idx]:.4f}")


if __name__ == "__main__":
    main()

