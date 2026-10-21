"""Measure V2, V3.1 and EasyOCR on fixed local demonstrations, without tuning."""

import hashlib
import json
from pathlib import Path

from src.cnn_inference import load_classes, load_model, predict_characters
from src.engines import (character_diagnostics, load_v31_model, predict_v31,
                         prediction_text, reference_check)
from src.ocr import availability, create_reader, read_plate
from src.pipeline import analyze_image
from src.preprocessing import decode_upload

from .validate_v21 import REFERENCES, make_plate, make_vehicle

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "images.jpg"


def measure(image, mode, reference, classes, v2_model, v31_model,
            reader=None, crop_coords=None):
    analysis = analyze_image(image, mode, "auto", classes,
                             crop_coords=crop_coords, run_cnn=False)
    row = {"mode": mode, "reference_comparison_only": reference,
           "status": analysis.status,
           "candidate_polygon": (analysis.candidate.polygon.tolist()
                                 if analysis.candidate is not None else None),
           "segmented": (len(analysis.segmentation.characters)
                         if analysis.segmentation is not None else 0)}
    if analysis.status != "ready":
        row["reason"] = analysis.message
        return row
    chars = analysis.segmentation.characters
    v2 = predict_characters(v2_model, classes, chars)
    v31 = predict_v31(v31_model, classes, chars)
    for name, predictions in (("cnn_v2", v2), ("cnn_v31", v31)):
        text = prediction_text(predictions)
        row[name] = {"text": text, "predictions": predictions,
                     "reference_check": reference_check(text, reference)}
    row["diagnostics"] = character_diagnostics(v2, v31, reference)
    if reader is not None:
        ocr = read_plate(reader, analysis.raw_plate)
        row["easyocr"] = {"text": ocr["text"], "score_single_detection": ocr["score"],
                          "scores": ocr["scores"],
                          "reference_check": reference_check(ocr["text"], reference)}
    return row


def main():
    classes = load_classes(ROOT / "data" / "classes.json")
    v2 = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    v31 = load_v31_model(ROOT / "models" / "platevision_cnn_v31.keras",
                           ROOT / "data" / "manifest_v31.json", classes)
    synthetic = []
    for text in REFERENCES:
        for mode, image in (("cropped", make_plate(text)),
                            ("photo", make_vehicle(text))):
            synthetic.append(measure(image, mode, text, classes, v2, v31))
    payload = {"protocol": "Eight fabricated plate paths; smoke comparison only, not an accuracy benchmark.",
               "synthetic": synthetic, "real_photo": []}
    if PHOTO.is_file():
        photo_bytes = PHOTO.read_bytes()
        image = decode_upload(photo_bytes)
        if availability() != "ready":
            raise RuntimeError("EasyOCR non disponible pour le test réel V3.2.")
        reader = create_reader()
        payload["photo_sha256"] = hashlib.sha256(photo_bytes).hexdigest()
        for mode, coords in (("photo", None), ("manual", (124, 124, 180, 150))):
            payload["real_photo"].append(measure(
                image, mode, "J389NLT", classes, v2, v31, reader, coords))
    target = ROOT / "validation" / "v32_results.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(target)
    for row in payload["real_photo"]:
        print(row["mode"], "V2", row["cnn_v2"]["text"],
              "V3.1", row["cnn_v31"]["text"], "OCR", row["easyocr"]["text"])
    for row in synthetic:
        print("synthetic", row["mode"], row["reference_comparison_only"],
              "V2", row["cnn_v2"]["text"], "V3.1", row["cnn_v31"]["text"])


if __name__ == "__main__":
    main()
