"""V3.2 artifact, preprocessing, reference and UI regression checks."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import ocr as ocr_module
from src.cnn_inference import load_classes
from src.engines import (V31_SHA256, character_diagnostics, load_v31_model,
                         predict_v31, reference_check, v31_availability)
from src.preprocessing import character_tensor, v31_character_tensor

ROOT = Path(__file__).resolve().parents[1]


def glyph(text="8"):
    image = np.zeros((42, 28), dtype=np.uint8)
    cv2.putText(image, text, (2, 34), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, 255, 2, cv2.LINE_AA)
    return image


def test_v31_artifact_loads_and_predicts_without_custom_layers():
    classes = load_classes(ROOT / "data" / "classes.json")
    model_path = ROOT / "models" / "platevision_cnn_v31.keras"
    manifest_path = ROOT / "data" / "manifest_v31.json"
    assert v31_availability(model_path, manifest_path, classes) == "ready"
    assert len(V31_SHA256) == 64
    model = load_v31_model(model_path, manifest_path, classes)
    assert model.input_shape == (None, 32, 32, 1)
    assert model.output_shape == (None, 36)
    predictions = predict_v31(model, classes, [glyph()])
    assert len(predictions) == 1
    assert predictions[0]["character"] in classes
    assert 0 <= predictions[0]["score"] <= 1


def test_v31_preprocessing_is_separate_and_inverted():
    image = glyph()
    v2, v31 = character_tensor(image), v31_character_tensor(image)
    assert v31.shape == (1, 32, 32, 1)
    assert v31.dtype == np.float32
    assert v31[0, 0, 0, 0] == 0
    assert 0 <= float(v31.min()) < float(v31.max()) <= 1
    assert not np.array_equal(v2, v31)
    with pytest.raises(ValueError):
        v31_character_tensor(np.zeros((8, 8), dtype=np.uint8))


def test_v31_missing_corrupt_and_incompatible_artifacts(tmp_path):
    classes = load_classes(ROOT / "data" / "classes.json")
    model, manifest = tmp_path / "model.keras", tmp_path / "manifest.json"
    assert v31_availability(model, manifest, classes) == "missing"
    model.write_bytes(b"corrupt")
    manifest.write_text("[]")
    assert v31_availability(model, manifest, classes) == "incompatible"
    manifest.write_text('{"classes":"wrong","image_shape":[32,32,1]}')
    assert v31_availability(model, manifest, classes) == "incompatible"
    manifest.write_bytes((ROOT / "data" / "manifest_v31.json").read_bytes())
    assert v31_availability(model, manifest, classes) == "corrupt"
    with pytest.raises(RuntimeError, match="corrupt"):
        load_v31_model(model, manifest, classes)


@pytest.mark.parametrize("expected,observed", [("0", "O"), ("1", "I"),
                                                ("J", "U"), ("8", "B")])
def test_confusion_diagnostics_only_label_never_replace(expected, observed):
    a = [{"character": expected, "score": 0.6}]
    b = [{"character": observed, "score": 0.7}]
    before_a, before_b = a[0]["character"], b[0]["character"]
    rows = character_diagnostics(a, b, expected)
    assert rows[0]["Accord"] == "Non"
    assert f"{expected}/{observed}" in rows[0]["Confusion à examiner"]
    assert (a[0]["character"], b[0]["character"]) == (before_a, before_b)
    assert reference_check(observed, expected)["exact"] is False
    assert reference_check(expected, expected)["correct_positions"] == 1
    assert reference_check(observed, "") is None


class Upload:
    def __init__(self, content):
        self.content = content

    def getvalue(self):
        return self.content


def test_streamlit_v31_simple_and_three_engine_comparison(monkeypatch):
    image = np.full((90, 400, 3), 235, dtype=np.uint8)
    cv2.putText(image, "456DEF", (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (20, 20, 20), 3, cv2.LINE_AA)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(buffer.getvalue()))
    monkeypatch.setattr(ocr_module, "availability", lambda: "ready")
    monkeypatch.setattr(ocr_module, "create_reader", lambda: object())
    monkeypatch.setattr(ocr_module, "read_plate", lambda reader, raw: {
        "text": "456DEF", "score": 0.7, "scores": [0.7], "detections": [],
        "prepared": ocr_module.prepare_ocr_image(raw),
    })
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    next(box for box in app.selectbox if box.label == "Moteur de reconnaissance").set_value(
        "CNN V3.1").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("CNN V3.1" in caption.value for caption in app.caption)
    assert any(header.value for header in app.header)
    app.segmented_control(key="space").set_value("Comparaison IA").run()
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    app.text_input[0].set_value("456DEF").run()
    app.button[0].click().run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert {"CNN V2", "CNN V3.1", "EasyOCR", "Score brut EasyOCR"} <= set(metrics)
    assert any("Diagnostic des caractères" in item.value for item in app.subheader)
    assert any("lecture exacte" in item.value for item in app.success)


def test_streamlit_missing_v31_keeps_v2_available(monkeypatch):
    image = np.full((90, 400, 3), 235, dtype=np.uint8)
    cv2.putText(image, "456DEF", (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (20, 20, 20), 3, cv2.LINE_AA)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(buffer.getvalue()))
    from src import engines as engine_module
    monkeypatch.setattr(engine_module, "v31_availability", lambda *args: "missing")
    monkeypatch.setattr(engine_module, "load_v31_model",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(engine_module, "predict_v31",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("missing")))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    app.segmented_control(key="space").set_value("Comparaison IA").run()
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    app.multiselect[0].set_value(["CNN V2", "CNN V3.1"]).run()
    app.button[0].click().run()
    assert not app.exception
    assert any(item.label == "CNN V2" and item.value == "456DEF" for item in app.metric)
    assert any("missing" in item.value for item in app.warning)
