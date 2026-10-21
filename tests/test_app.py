from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from streamlit.testing.v1 import AppTest
from src import ocr as ocr_module


def test_app_starts_and_analysis_page_renders():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert any("Choisissez une image" in item.value for item in app.info)
    app.radio(key="space").set_value("À propos du projet").run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["CNN V1"] == "57.36%"
    assert metrics["CNN V2"] == "80.65%"
    assert metrics["Amélioration"] == "+23.29 points"
    assert metrics["Classes"] == "36"
    assert len(app.tabs) == 6
    app.radio(key="space").set_value("Comparaison IA").run()
    assert not app.exception
    app.radio(key="space").set_value("Analyse de plaque").run()
    assert not app.exception


class Upload:
    def __init__(self, content):
        self.content = content

    def getvalue(self):
        return self.content


def test_app_chart_shows_fallback_without_prediction(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(st, "file_uploader",
                        lambda *args, **kwargs: Upload((root / "assets" / "cnn_v2_training.png").read_bytes()))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any("Aucune plaque détectée" in item.value for item in app.warning)
    assert not any("RÉSULTAT PRINCIPAL" in item.value for item in app.caption)
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Aucun caractère exploitable" in item.value for item in app.warning)
    assert not any("RÉSULTAT PRINCIPAL" in item.value for item in app.caption)


def test_app_corrupt_upload_shows_error(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(b"broken PNG"))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert any("Fichier image invalide" in item.value for item in app.error)
    assert not app.button


def test_app_cropped_plate_renders_cnn_result(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    image = np.full((90, 400, 3), 235, dtype=np.uint8)
    cv2.putText(image, "456DEF", (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (20, 20, 20), 3, cv2.LINE_AA)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(buffer.getvalue()))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=120).run()
    assert not app.exception
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("RÉSULTAT PRINCIPAL" in item.value for item in app.caption)
    assert any(item.value == "456DEF" for item in app.header)


def test_ocr_comparison_renders_prepared_image_score_and_reference(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    image = np.full((90, 400, 3), 235, dtype=np.uint8)
    cv2.putText(image, "456DEF", (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                1.7, (20, 20, 20), 3, cv2.LINE_AA)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: Upload(buffer.getvalue()))
    monkeypatch.setattr(ocr_module, "availability", lambda: "ready")
    monkeypatch.setattr(ocr_module, "create_reader", lambda: object())

    def fake_read_plate(reader, plate):
        prepared = ocr_module.prepare_ocr_image(plate)
        return {"text": "456DEF", "score": 0.9823, "scores": [0.9823],
                "detections": [([[0, 0], [10, 0], [10, 10], [0, 10]],
                                "456DEF", 0.9823)], "prepared": prepared}

    monkeypatch.setattr(ocr_module, "read_plate", fake_read_plate)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not app.exception
    next(radio for radio in app.radio if radio.label == "Type d'image").set_value(
        "Plaque déjà recadrée").run()
    app.checkbox[0].check().run()
    app.text_input[0].set_value("456DEF").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Comparaison des méthodes de reconnaissance" in item.value
               for item in app.subheader)
    assert any(metric.label == "Score brut EasyOCR" and metric.value == "0.9823"
               for metric in app.metric)
    assert any("EasyOCR : lecture exacte" in item.value for item in app.success)
