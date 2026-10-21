"""Independent EasyOCR preparation and lazy CPU inference."""

import hashlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Checksums from EasyOCR 1.7.2 config for its official CRAFT and English G2 files.
MODEL_MD5 = {
    "craft_mlt_25k.pth": "2f8227d2def4037cdb3b34389dcf9ec1",
    "english_g2.pth": "5864788e1821be9e454ec108d61b887d",
}


def model_directory() -> Path:
    base = os.environ.get("EASYOCR_MODULE_PATH") or os.environ.get("MODULE_PATH")
    return (Path(base).expanduser() if base else Path.home() / ".EasyOCR") / "model"


@lru_cache(maxsize=8)
def _valid_weight(path: str, size: int, mtime_ns: int, expected_md5: str) -> bool:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_md5


def availability() -> str:
    """Report package and verified-weight readiness without importing Torch."""
    if importlib.util.find_spec("easyocr") is None:
        return "package_missing"
    paths = [(model_directory() / name, checksum) for name, checksum in MODEL_MD5.items()]
    if not all(path.is_file() for path, _ in paths):
        return "weights_missing"
    try:
        if not all(_valid_weight(str(path), path.stat().st_size, path.stat().st_mtime_ns, checksum)
                   for path, checksum in paths):
            return "weights_invalid"
    except OSError:
        return "weights_invalid"
    return "ready"


def create_reader():
    """Initialize only after explicit readiness; Streamlit caches the returned reader."""
    if availability() != "ready":
        raise RuntimeError("Poids EasyOCR absents ou invalides ; lancer python -m scripts.download_easyocr.")
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False, download_enabled=False)


def prepare_ocr_image(plate: np.ndarray) -> np.ndarray:
    """RGB/raw plate → grayscale → cubic ×4 → binary Otsu; CNN input is untouched."""
    if not isinstance(plate, np.ndarray) or plate.dtype != np.uint8 or plate.size == 0:
        raise ValueError("Région OCR vide ou non uint8.")
    if plate.ndim == 2:
        gray = plate
    elif plate.ndim == 3 and plate.shape[2] == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Région OCR attendue en gris ou RGB.")
    if min(gray.shape) < 2:
        raise ValueError("Recadrage OCR trop petit.")
    enlarged = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(enlarged, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def read_plate(reader, plate: np.ndarray) -> dict:
    prepared = prepare_ocr_image(plate)
    detections = reader.readtext(prepared, detail=1, paragraph=False, allowlist=ALLOWLIST)
    detections = sorted(detections, key=lambda item: min(point[0] for point in item[0]))
    text = "".join("".join(c for c in item[1].upper() if c in ALLOWLIST)
                   for item in detections)
    scores = [float(item[2]) for item in detections]
    return {"text": text, "detections": detections, "scores": scores,
            "score": scores[0] if len(scores) == 1 else None,
            "prepared": prepared}
