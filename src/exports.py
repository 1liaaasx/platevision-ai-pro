"""Portable exports of one observed analysis; never infer correctness from a score."""

import csv
import io
import json

import cv2
import numpy as np

from .detection import draw_candidate
from .engines import reference_check


def result_record(result: dict, reference: str = "") -> dict:
    analysis = result["analysis"]
    engines = {}
    for name, item in result.get("engines", {}).items():
        predictions = item.get("predictions") or []
        ocr = item.get("ocr") or {}
        engines[name] = {
            "text": item.get("text") or "",
            "error": item.get("error"),
            "elapsed_ms": item.get("elapsed_ms"),
            "reference_check": reference_check(item.get("text") or "", reference),
            "characters": [{"character": p["character"], "raw_softmax": float(p["score"])}
                           for p in predictions],
            "ocr_raw_score": ocr.get("score"),
        }
    candidate = analysis.candidate
    return {
        "schema_version": "1.0",
        "status": analysis.status,
        "message": analysis.message,
        "reference": reference or None,
        "source_size": [int(result["source"].shape[1]), int(result["source"].shape[0])],
        "candidate_polygon": candidate.polygon.tolist() if candidate is not None else None,
        "candidate_heuristic_score": float(candidate.score) if candidate is not None else None,
        "segmented_characters": len(analysis.segmentation.characters) if analysis.segmentation else 0,
        "pipeline_elapsed_ms": result.get("pipeline_elapsed_ms"),
        "engines": engines,
    }


def result_json(result: dict, reference: str = "") -> bytes:
    return (json.dumps(result_record(result, reference), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def result_csv(result: dict, reference: str = "") -> bytes:
    record = result_record(result, reference)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["engine", "status", "text", "reference", "exact",
                                                "elapsed_ms", "raw_score", "error"])
    writer.writeheader()
    for name, item in record["engines"].items():
        check = item["reference_check"]
        writer.writerow({"engine": name, "status": record["status"], "text": item["text"],
                         "reference": reference, "exact": check["exact"] if check else "",
                         "elapsed_ms": item["elapsed_ms"], "raw_score": item["ocr_raw_score"],
                         "error": item["error"] or ""})
    return stream.getvalue().encode("utf-8-sig")


def annotated_png(result: dict) -> bytes | None:
    analysis = result["analysis"]
    if analysis.candidate is None or analysis.status != "ready":
        return None
    rgb = draw_candidate(result["source"], analysis.candidate)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    return encoded.tobytes() if success else None
