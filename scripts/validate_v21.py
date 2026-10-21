"""Record observed outputs on fabricated plates without claiming real-world accuracy."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.cnn_inference import load_classes, load_model
from src.pipeline import analyze_image

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ("AB1234", "123ABC", "XY7890", "456DEF")


def make_plate(text: str) -> np.ndarray:
    image = np.full((90, 400, 3), 235, dtype=np.uint8)
    cv2.putText(image, text, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (20, 20, 20), 3, cv2.LINE_AA)
    return image


def make_vehicle(text: str) -> np.ndarray:
    image = np.full((400, 800, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (220, 170), (620, 260), (235, 235, 235), -1)
    cv2.rectangle(image, (220, 170), (620, 260), (20, 20, 20), 3)
    cv2.putText(image, text, (238, 238), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (20, 20, 20), 3, cv2.LINE_AA)
    return image


def record(reference: str, scenario: str, image: np.ndarray, mode: str,
           model, classes: list[str]) -> dict:
    result = analyze_image(image, mode, "auto", classes, model=model)
    predicted = "".join(item["character"] for item in result.predictions or [])
    errors = [{"position": i + 1,
               "reference": reference[i] if i < len(reference) else None,
               "observed": predicted[i] if i < len(predicted) else None}
              for i in range(max(len(reference), len(predicted)))
              if i >= len(reference) or i >= len(predicted) or reference[i] != predicted[i]]
    return {"scenario": scenario, "reference": reference, "status": result.status,
            "segmented": len(result.segmentation.characters) if result.segmentation else 0,
            "observed_cnn": predicted or None, "differences": errors,
            "candidate_score": round(result.candidate.score, 4) if result.candidate else None}


def main():
    classes = load_classes(ROOT / "data" / "classes.json")
    model = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    rows = []
    for reference in REFERENCES:
        rows.append(record(reference, "cropped", make_plate(reference), "cropped", model, classes))
        rows.append(record(reference, "vehicle", make_vehicle(reference), "photo", model, classes))
    chart = np.asarray(Image.open(ROOT / "assets" / "cnn_v2_training.png").convert("RGB"))
    for mode in ("photo", "cropped"):
        result = analyze_image(chart, mode, "auto", classes, model=model)
        rows.append({"scenario": f"training_chart_{mode}", "reference": None,
                     "status": result.status, "segmented": len(result.segmentation.characters)
                     if result.segmentation else 0, "observed_cnn": None, "differences": [],
                     "candidate_score": None})
    output = ROOT / "validation" / "synthetic_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    for row in rows:
        print(f"{row['scenario']:22} {str(row['reference']):8} {row['status']:13} "
              f"{str(row['observed_cnn']):9} differences={len(row['differences'])}")


if __name__ == "__main__":
    main()
