"""Measure local cold loads and inference without claiming benchmark accuracy."""

import json
import os
import sys
from pathlib import Path
from time import perf_counter, process_time

import cv2
import numpy as np

from src.cnn_inference import load_classes, load_model, predict_characters
from src.engines import load_v31_model, predict_v31
from src.ocr import availability, create_reader, read_plate
from src.pipeline import analyze_image

ROOT = Path(__file__).resolve().parents[1]


def memory_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 1048576, 1)
    except ImportError:
        if sys.platform != "win32":
            return None
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        process = kernel.GetCurrentProcess()
        buffer = ctypes.create_string_buffer(72)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not get_info(process, buffer, len(buffer)):
            return None
        return round(ctypes.c_size_t.from_buffer(buffer, 16).value / 1048576, 1)


def measure(fn):
    wall, cpu, before = perf_counter(), process_time(), memory_mb()
    value = fn()
    return value, {"wall_ms": round((perf_counter()-wall)*1000, 2),
                   "cpu_ms": round((process_time()-cpu)*1000, 2),
                   "rss_before_mb": before, "rss_after_mb": memory_mb()}


def main():
    classes = load_classes(ROOT / "data/classes.json")
    image = np.full((90, 400, 3), 235, dtype=np.uint8)
    cv2.putText(image, "456DEF", (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (20, 20, 20), 3, cv2.LINE_AA)
    analysis, pipeline = measure(lambda: analyze_image(image, "cropped", "auto", classes, run_cnn=False))
    if analysis.status != "ready":
        raise RuntimeError(analysis.message)
    report = {"environment": "local CPU; single process; one synthetic crop",
              "pipeline": pipeline, "engines": {}, "easyocr_availability": availability()}
    v2, load = measure(lambda: load_model(ROOT / "models/platevision_cnn_v2.keras"))
    _, inference = measure(lambda: predict_characters(v2, classes, analysis.segmentation.characters))
    report["engines"]["CNN V2"] = {"cold_load": load, "inference": inference}
    v31, load = measure(lambda: load_v31_model(ROOT / "models/platevision_cnn_v31.keras",
                                                ROOT / "data/manifest_v31.json", classes))
    _, inference = measure(lambda: predict_v31(v31, classes, analysis.segmentation.characters))
    report["engines"]["CNN V3.1"] = {"cold_load": load, "inference": inference}
    if report["easyocr_availability"] == "ready":
        reader, load = measure(create_reader)
        _, inference = measure(lambda: read_plate(reader, analysis.raw_plate))
        report["engines"]["EasyOCR"] = {"cold_load": load, "inference": inference}
    output = ROOT / "validation/release_profile.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
