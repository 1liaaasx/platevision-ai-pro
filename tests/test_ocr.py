import hashlib
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src import ocr
from src.ocr import prepare_ocr_image, read_plate


class StubReader:
    def readtext(self, image, **kwargs):
        assert image.shape == (128, 384)
        assert image.dtype == np.uint8
        assert set(np.unique(image)).issubset({0, 255})
        assert kwargs == {"detail": 1, "paragraph": False,
                          "allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
        return [
            ([[50, 0], [90, 0], [90, 20], [50, 20]], "b-2", 0.8),
            ([[0, 0], [40, 0], [40, 20], [0, 20]], "a 1", 0.7),
        ]


def test_otsu_x4_matches_explicit_colab_sequence():
    plate = np.full((32, 96, 3), 225, dtype=np.uint8)
    cv2.putText(plate, "A1", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (15, 25, 30), 2, cv2.LINE_AA)
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    large = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, expected = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert np.array_equal(prepare_ocr_image(plate), expected)
    assert np.array_equal(prepare_ocr_image(gray), expected)


def test_ocr_joins_detections_left_to_right_and_keeps_raw_scores():
    output = read_plate(StubReader(), np.zeros((32, 96, 3), dtype=np.uint8))
    assert output["text"] == "A1B2"
    assert output["scores"] == [0.7, 0.8]
    assert output["score"] is None  # Multiple fragments: no invented aggregate score.
    assert len(output["detections"]) == 2


@pytest.mark.parametrize("invalid", [
    np.zeros((0, 4), dtype=np.uint8),
    np.zeros((1, 4), dtype=np.uint8),
    np.zeros((4, 4), dtype=np.float32),
    np.zeros((4, 4, 2), dtype=np.uint8),
])
def test_ocr_rejects_invalid_or_empty_crop(invalid):
    with pytest.raises(ValueError):
        prepare_ocr_image(invalid)


def test_ocr_readiness_requires_validated_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("EASYOCR_MODULE_PATH", str(tmp_path))
    payloads = {"craft_mlt_25k.pth": b"craft", "english_g2.pth": b"english"}
    monkeypatch.setattr(ocr, "MODEL_MD5", {name: hashlib.md5(data).hexdigest()
                                           for name, data in payloads.items()})
    model = tmp_path / "model"
    model.mkdir()
    assert ocr.availability() == "weights_missing"
    (model / "craft_mlt_25k.pth").write_bytes(payloads["craft_mlt_25k.pth"])
    assert ocr.availability() == "weights_missing"
    english = model / "english_g2.pth"
    english.write_bytes(b"partial")
    assert ocr.availability() == "weights_invalid"
    english.write_bytes(payloads["english_g2.pth"])
    assert ocr.availability() == "ready"


def test_ocr_readiness_reports_missing_package(monkeypatch):
    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda name: None)
    assert ocr.availability() == "package_missing"


def test_reader_is_cpu_only_and_download_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, "availability", lambda: "ready")
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(
        Reader=lambda *args, **kwargs: calls.append((args, kwargs)) or "reader"))
    assert ocr.create_reader() == "reader"
    assert calls == [((["en"],), {"gpu": False, "verbose": False,
                                  "download_enabled": False})]
