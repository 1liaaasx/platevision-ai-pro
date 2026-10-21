"""Evaluate an explicitly annotated, locally authorized real-photo set."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.cnn_inference import load_classes, load_model, predict_characters
from src.engines import load_v31_model, predict_v31, prediction_text
from src.ocr import availability, create_reader, read_plate
from src.pipeline import analyze_image
from src.preprocessing import decode_upload

ROOT = Path(__file__).resolve().parents[1]


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def bbox_iou(a: list[int], b: list[int]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def validate_annotations(data: dict, images_dir: Path) -> list[dict]:
    rows = data.get("images", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Annotations vides : ajoutez des photos autorisées avant le benchmark.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not {"id", "file", "plate_bbox", "text", "rights"} <= row.keys():
            raise ValueError("Annotation incomplète : id, file, plate_bbox, text et rights requis.")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Identifiants d'images non uniques ou invalides.")
        seen.add(row["id"])
        if not isinstance(row["file"], str) or not row["file"]:
            raise ValueError(f"Nom de fichier invalide : {row['id']}")
        path = (images_dir / row["file"]).resolve()
        if not path.is_relative_to(images_dir.resolve()) or not path.is_file():
            raise ValueError(f"Image manquante ou hors dossier : {row['id']}")
        box = row["plate_bbox"]
        if (not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(x, int) or x < 0 for x in box) or
                box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError(f"Boîte invalide : {row['id']}")
        if not isinstance(row["text"], str) or not row["text"].isalnum() or row["text"] != row["text"].upper():
            raise ValueError(f"Référence de plaque invalide : {row['id']}")
        if not isinstance(row["rights"], str) or not row["rights"].strip():
            raise ValueError(f"Droits non documentés : {row['id']}")
    return rows


def evaluate(annotations: Path, images_dir: Path, engines: list[str]) -> dict:
    rows = validate_annotations(json.loads(annotations.read_text(encoding="utf-8")), images_dir)
    classes = load_classes(ROOT / "data/classes.json")
    models = {}
    if "CNN V2" in engines:
        models["CNN V2"] = load_model(ROOT / "models/platevision_cnn_v2.keras")
    if "CNN V3.1" in engines:
        models["CNN V3.1"] = load_v31_model(ROOT / "models/platevision_cnn_v31.keras",
                                            ROOT / "data/manifest_v31.json", classes)
    if "EasyOCR" in engines:
        if availability() != "ready":
            raise RuntimeError("EasyOCR et ses poids vérifiés sont nécessaires pour ce moteur.")
        models["EasyOCR"] = create_reader()
    observations = []
    for row in rows:
        image = decode_upload((images_dir / row["file"]).read_bytes())
        if row["plate_bbox"][2] > image.shape[1] or row["plate_bbox"][3] > image.shape[0]:
            raise ValueError(f"Boîte hors image : {row['id']}")
        started = perf_counter()
        analysis = analyze_image(image, "photo", "auto", classes, run_cnn=False)
        detection_ms = round((perf_counter()-started)*1000, 2)
        observed_box = None
        iou = 0.0
        if analysis.candidate is not None:
            polygon = analysis.candidate.polygon
            observed_box = [int(polygon[:, 0].min()), int(polygon[:, 1].min()),
                            int(polygon[:, 0].max()) + 1, int(polygon[:, 1].max()) + 1]
            iou = bbox_iou(observed_box, row["plate_bbox"])
        outputs = {}
        for name, model in models.items():
            begin = perf_counter()
            text = ""
            error = None
            if analysis.status == "ready":
                try:
                    if name == "CNN V2":
                        text = prediction_text(predict_characters(model, classes, analysis.segmentation.characters))
                    elif name == "CNN V3.1":
                        text = prediction_text(predict_v31(model, classes, analysis.segmentation.characters))
                    else:
                        text = read_plate(model, analysis.raw_plate)["text"]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            outputs[name] = {"text": text, "exact": text == row["text"],
                             "edit_distance": edit_distance(text, row["text"]),
                             "inference_ms": round((perf_counter()-begin)*1000, 2), "error": error}
        observations.append({"id": row["id"], "reference": row["text"],
                             "status": analysis.status, "message": analysis.message,
                             "annotated_bbox": row["plate_bbox"], "detected_bbox": observed_box,
                             "iou": round(iou, 4), "detection_success_iou_0_5": iou >= 0.5,
                             "pipeline_ms": detection_ms, "engines": outputs})
    summary = {"images": len(rows),
               "detection_recall_iou_0_5": sum(r["detection_success_iou_0_5"] for r in observations)/len(rows),
               "engines": {}}
    for name in engines:
        summary["engines"][name] = {
            "plate_exact_accuracy": sum(r["engines"][name]["exact"] for r in observations)/len(rows),
            "character_accuracy_edit": max(0.0, 1 - sum(r["engines"][name]["edit_distance"] for r in observations) /
                                           sum(len(r["reference"]) for r in observations)),
            "mean_inference_ms": float(np.mean([r["engines"][name]["inference_ms"] for r in observations])),
        }
    return {"protocol": "real-photo automatic detection; IoU >= 0.5", "summary": summary,
            "observations": observations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", choices=["CNN V2", "CNN V3.1", "EasyOCR"],
                        default=["CNN V2", "CNN V3.1"])
    args = parser.parse_args()
    report = evaluate(args.annotations, args.images_dir, args.engines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
