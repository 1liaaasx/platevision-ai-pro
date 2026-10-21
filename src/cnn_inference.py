"""Load the supplied CNN and map outputs to the supplied class order."""

import json
from pathlib import Path

import numpy as np

from .preprocessing import character_tensor


def load_classes(path: str | Path) -> list[str]:
    classes = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(classes, list) or len(classes) != 36
            or len(set(classes)) != 36 or any(not isinstance(c, str) or len(c) != 1 for c in classes)):
        raise ValueError("classes.json doit contenir 36 caractères uniques.")
    return classes


def load_model(path: str | Path):
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    if tuple(model.input_shape[1:]) != (32, 32, 1) or model.output_shape[-1] != 36:
        raise ValueError("Dimensions du CNN incompatibles avec les artefacts fournis.")
    return model


def predict_characters(model, classes: list[str], characters: list[np.ndarray],
                       tensorizer=character_tensor) -> list[dict]:
    if not characters:
        return []
    batch = np.concatenate([tensorizer(char) for char in characters], axis=0)
    outputs = np.asarray(model.predict(batch, verbose=0))
    if outputs.shape != (len(characters), len(classes)) or not np.all(np.isfinite(outputs)):
        raise ValueError("Sortie du CNN invalide.")
    predictions = []
    for row in outputs:
        idx = int(np.argmax(row))
        top = np.argsort(row)[::-1][:3]
        predictions.append({
            "character": classes[idx],
            "score": float(row[idx]),
            "top3": [(classes[int(i)], float(row[i])) for i in top],
        })
    return predictions
