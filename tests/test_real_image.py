"""Regression checks for the supplied low-resolution real photograph."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.cnn_inference import load_classes, load_model
from src.detection import detect_plate
from src.pipeline import analyze_image

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "images.jpg"
REFERENCE = "J389NLT"
PHOTO_SHA256 = "96fc6f54ee823bcd758a2245effab12c4dbc25a1bdcaa8a9eaf5625cbd6c6f6b"
pytestmark = pytest.mark.skipif(not PHOTO.is_file(), reason="Photo réelle privée non fournie")


def real_image():
    content = PHOTO.read_bytes()
    assert hashlib.sha256(content).hexdigest() == PHOTO_SHA256
    return np.asarray(Image.open(PHOTO).convert("RGB"))


def test_small_real_plate_is_localized():
    candidate = detect_plate(real_image())
    assert candidate is not None
    x0, y0 = candidate.polygon.min(axis=0)
    x1, y1 = candidate.polygon.max(axis=0)
    assert 120 <= x0 <= 130 and 120 <= y0 <= 130
    assert 175 <= x1 <= 190 and 145 <= y1 <= 155
    assert candidate.crop.shape[1] >= 50


def test_real_photo_pipeline_records_partial_reading():
    image = real_image()
    classes = load_classes(ROOT / "data" / "classes.json")
    model = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    result = analyze_image(image, "photo", "auto", classes, model=model)
    assert result.status == "ready"
    assert "non vérifiée" in result.message
    assert len(result.segmentation.characters) == len(REFERENCE)
    assert result.raw_plate.shape[:2] == (26, 56)
    assert result.plate.shape[:2] == (52, 112)
    observed = "".join(item["character"] for item in result.predictions)
    assert len(observed) == len(REFERENCE)
    # Regression floor for this single fixture, not a model-accuracy claim.
    assert sum(a == b for a, b in zip(observed, REFERENCE)) >= 5


def test_manual_region_on_same_photo_recovers_seven_characters():
    image = real_image()
    classes = load_classes(ROOT / "data" / "classes.json")
    model = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    for coords in ((119, 124, 184, 151), (122, 126, 182, 148)):
        result = analyze_image(image, "manual", "auto", classes,
                               crop_coords=coords, model=model)
        assert result.status == "ready"
        assert len(result.segmentation.characters) == 7


def test_precropped_mode_on_manually_defined_region():
    image = real_image()[124:151, 119:184]
    classes = load_classes(ROOT / "data" / "classes.json")
    model = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    result = analyze_image(image, "cropped", "auto", classes, model=model)
    assert result.status == "ready"
    assert len(result.segmentation.characters) == 7
