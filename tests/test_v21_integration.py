"""End-to-end tests use fabricated images; texts are references, not expected CNN outputs."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.cnn_inference import load_classes, load_model
from src.pipeline import analyze_image, region_is_plausible
from src.preprocessing import decode_upload
from src.segmentation import segment_characters

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ("AB1234", "123ABC", "XY7890", "456DEF")


def plate(text: str, inverted: bool = False) -> np.ndarray:
    background, ink = ((20, 235) if inverted else (235, 20))
    image = np.full((90, 400, 3), background, dtype=np.uint8)
    cv2.putText(image, text, (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (ink, ink, ink), 3, cv2.LINE_AA)
    return image


def vehicle(text: str) -> np.ndarray:
    image = np.full((400, 800, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (220, 170), (620, 260), (235, 235, 235), -1)
    cv2.rectangle(image, (220, 170), (620, 260), (20, 20, 20), 3)
    cv2.putText(image, text, (238, 238), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                (20, 20, 20), 3, cv2.LINE_AA)
    return image


@pytest.fixture(scope="module")
def cnn():
    return load_model(ROOT / "models" / "platevision_cnn_v2.keras")


@pytest.fixture(scope="module")
def classes():
    return load_classes(ROOT / "data" / "classes.json")


@pytest.mark.parametrize("reference", TEXTS)
def test_cropped_plate_through_real_cnn(reference, cnn, classes):
    image = plate(reference)
    result = analyze_image(image, "cropped", "auto", classes, model=cnn)
    assert result.status == "ready"
    assert len(result.predictions) == len(reference)
    assert all(item["character"] in classes for item in result.predictions)
    assert all(0 <= item["score"] <= 1 for item in result.predictions)


def test_vehicle_detection_segmentation_real_cnn(cnn, classes):
    result = analyze_image(vehicle("AB1234"), "photo", "auto", classes, model=cnn)
    assert result.status == "ready"
    assert result.candidate is not None
    assert len(result.predictions) == 6


def test_inverted_plate_and_character_order(cnn, classes):
    image = plate("123ABC", inverted=True)
    seg = segment_characters(image, "auto")
    assert seg.polarity == "light-on-dark"
    assert [box[0] for box in seg.boxes] == sorted(box[0] for box in seg.boxes)
    assert region_is_plausible(image, seg)
    assert analyze_image(image, "cropped", "auto", classes, model=cnn).status == "ready"


def test_multiple_candidate_regions(cnn, classes):
    image = np.full((500, 1000, 3), 80, dtype=np.uint8)
    for text, x, y in (("AB1234", 30, 100), ("456DEF", 550, 280)):
        cv2.rectangle(image, (x, y), (x+400, y+90), (235, 235, 235), -1)
        cv2.rectangle(image, (x, y), (x+400, y+90), (20, 20, 20), 3)
        cv2.putText(image, text, (x+18, y+68), cv2.FONT_HERSHEY_SIMPLEX,
                    1.7, (20, 20, 20), 3, cv2.LINE_AA)
    result = analyze_image(image, "photo", "auto", classes, model=cnn)
    assert result.status == "ready"
    assert result.candidate is not None
    assert len(result.predictions) == 6


def test_irrelevant_chart_and_blank_skip_cnn(classes):
    chart = np.asarray(Image.open(ROOT / "assets" / "cnn_v2_training.png").convert("RGB"))
    def forbidden():
        raise AssertionError("CNN must not load on irrelevant image")
    assert analyze_image(chart, "photo", "auto", classes, model_loader=forbidden).status == "no_candidate"
    assert analyze_image(chart, "cropped", "auto", classes, model_loader=forbidden).status == "unusable"
    blank = np.full((200, 400, 3), 180, dtype=np.uint8)
    assert analyze_image(blank, "photo", "auto", classes, model_loader=forbidden).status == "no_candidate"
    assert analyze_image(blank, "cropped", "auto", classes, model_loader=forbidden).status == "unusable"


def test_manual_crop_and_invalid_bounds(cnn, classes):
    image = vehicle("XY7890")
    result = analyze_image(image, "manual", "auto", classes,
                           crop_coords=(221, 171, 619, 259), model=cnn)
    assert result.status == "ready"
    with pytest.raises(ValueError, match="Coordonnées"):
        analyze_image(image, "manual", "auto", classes, crop_coords=(20, 20, 10, 30),
                      model=cnn)


def test_large_plate_is_bounded_before_segmentation(cnn, classes):
    large = cv2.resize(plate("456DEF"), (2400, 540), interpolation=cv2.INTER_LINEAR)
    result = analyze_image(large, "cropped", "auto", classes, model=cnn)
    assert result.status == "ready"
    assert result.plate.shape[1] == 1600
    assert len(result.predictions) == 6


def test_upload_png_and_corruption():
    image = Image.new("RGB", (64, 32), "white")
    ImageDraw.Draw(image).text((4, 8), "AB", fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    decoded = decode_upload(stream.getvalue())
    assert decoded.shape == (32, 64, 3)
    with pytest.raises(ValueError, match="invalide"):
        decode_upload(stream.getvalue()[:35])
    with pytest.raises(ValueError, match="15 Mo"):
        decode_upload(b"x" * (15 * 1024 * 1024 + 1))
