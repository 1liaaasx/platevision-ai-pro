"""Record robustness outcomes on generated examples without claiming accuracy."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.cnn_inference import load_classes
from src.pipeline import analyze_image

ROOT = Path(__file__).resolve().parents[1]


def plate(text: str = "456DEF") -> np.ndarray:
    image = np.full((90, 400, 3), 235, np.uint8)
    cv2.putText(image, text, (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (20, 20, 20), 3, cv2.LINE_AA)
    return image


def main() -> None:
    base = plate()
    rotated = cv2.warpAffine(base, cv2.getRotationMatrix2D((200, 45), 12, 1),
                             (400, 90), borderValue=(235, 235, 235))
    glare = base.copy()
    cv2.rectangle(glare, (155, 0), (200, 90), (255, 255, 255), -1)
    vehicle = np.full((400, 640, 3), 90, np.uint8)
    vehicle[260:350, 110:510] = base
    multiple = np.full((480, 640, 3), 90, np.uint8)
    multiple[90:180, 110:510] = base
    multiple[300:390, 110:510] = plate("123ABC")
    cases = {
        "base_crop": (base, "cropped"),
        "tiny_crop": (cv2.resize(base, (80, 18)), "cropped"),
        "large_crop": (cv2.resize(base, (2400, 540)), "cropped"),
        "dark_crop": ((base * 0.22).astype(np.uint8), "cropped"),
        "blurred_crop": (cv2.GaussianBlur(base, (15, 15), 4), "cropped"),
        "glare_crop": (glare, "cropped"),
        "inclined_crop": (rotated, "cropped"),
        "partial_crop": (base[:, :220], "cropped"),
        "vehicle": (vehicle, "photo"),
        "multiple_regions": (multiple, "photo"),
        "no_plate": (np.full((400, 640, 3), 100, np.uint8), "photo"),
        "too_small_photo": (np.full((20, 35, 3), 100, np.uint8), "photo"),
    }
    classes = load_classes(ROOT / "data/classes.json")
    rows = []
    for name, (image, mode) in cases.items():
        analysis = analyze_image(image, mode, "auto", classes, run_cnn=False)
        rows.append({"case": name, "size": [image.shape[1], image.shape[0]],
                     "mode": mode, "status": analysis.status, "message": analysis.message,
                     "segmented": len(analysis.segmentation.characters) if analysis.segmentation else 0,
                     "candidate": analysis.candidate is not None})
    target = ROOT / "validation/release_robustness.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
