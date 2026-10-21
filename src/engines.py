"""Independent recognition engines and reference-only diagnostics."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .cnn_inference import load_model, predict_characters
from .preprocessing import v31_character_tensor

V31_SHA256 = "851887cd2ecc5603a359fd0f753c3629514306fc2fcb852ee69d528a3e9913a9"
CONFUSION_PAIRS = {frozenset(pair): pair for pair in ("0O", "1I", "JU", "8B")}


@lru_cache(maxsize=4)
def _hash_file(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v31_availability(model_path: str | Path, manifest_path: str | Path,
                     classes: list[str]) -> str:
    model_path, manifest_path = Path(model_path), Path(manifest_path)
    if not model_path.is_file() or not manifest_path.is_file():
        return "missing"
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict)
                or metadata.get("classes") != "".join(classes)
                or metadata.get("image_shape") != [32, 32, 1]
                or "1 - gray/255" not in metadata.get("input_contract", "")):
            return "incompatible"
        stat = model_path.stat()
        if _hash_file(str(model_path), stat.st_size, stat.st_mtime_ns) != V31_SHA256:
            return "corrupt"
    except (OSError, ValueError, TypeError):
        return "corrupt"
    return "ready"


def load_v31_model(model_path: str | Path, manifest_path: str | Path,
                   classes: list[str]):
    state = v31_availability(model_path, manifest_path, classes)
    if state != "ready":
        raise RuntimeError(f"CNN V3.1 indisponible : {state}.")
    return load_model(model_path)


def predict_v31(model, classes: list[str], characters) -> list[dict]:
    return predict_characters(model, classes, characters,
                              tensorizer=v31_character_tensor)


def prediction_text(predictions: list[dict] | None) -> str:
    return "".join(item["character"] for item in predictions or [])


def reference_check(text: str, reference: str) -> dict | None:
    """Evaluate after inference; never feed this result into engine selection."""
    if not reference:
        return None
    correct = sum(a == b for a, b in zip(text, reference))
    return {"exact": text == reference, "correct_positions": correct,
            "reference_length": len(reference)}


def character_diagnostics(v2: list[dict] | None,
                          v31: list[dict] | None,
                          reference: str = "") -> list[dict]:
    if v2 is None or v31 is None:
        return []
    rows = []
    for index in range(max(len(v2), len(v31))):
        left = v2[index] if index < len(v2) else None
        right = v31[index] if index < len(v31) else None
        a, b = (left or {}).get("character"), (right or {}).get("character")
        alerts = []
        pair = CONFUSION_PAIRS.get(frozenset((a, b))) if a and b and a != b else None
        if pair:
            alerts.append(f"{pair[0]}/{pair[1]} · V2/V3.1")
        expected = reference[index] if index < len(reference) else None
        for name, observed in (("V2", a), ("V3.1", b)):
            pair = (CONFUSION_PAIRS.get(frozenset((expected, observed)))
                    if expected and observed and expected != observed else None)
            if pair:
                alerts.append(f"{pair[0]}/{pair[1]} · {name}/référence")
        rows.append({"Position": index + 1, "CNN V2": a or "—",
                     "Score V2": None if left is None else round(left["score"], 4),
                     "CNN V3.1": b or "—",
                     "Score V3.1": None if right is None else round(right["score"], 4),
                     "Accord": "Oui" if a == b else "Non",
                     "Confusion à examiner": "⚠ " + ", ".join(alerts) if alerts else ""})
    return rows
