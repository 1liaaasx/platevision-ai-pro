from pathlib import Path

import cv2
import numpy as np
import pytest

from src.cnn_inference import load_classes, load_model, predict_characters
from src.detection import detect_plate
from src.preprocessing import character_tensor, decode_upload, enhance_plate
from src.segmentation import segment_characters

ROOT = Path(__file__).resolve().parents[1]


def test_supplied_model_and_classes():
    classes = load_classes(ROOT / "data" / "classes.json")
    model = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    assert len(classes) == 36
    assert classes == list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert model.input_shape[1:] == (32, 32, 1)
    assert model.output_shape[-1] == 36


def test_character_preprocessing_tensor():
    char = np.zeros((42, 24), dtype=np.uint8)
    cv2.putText(char, "A", (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2)
    tensor = character_tensor(char)
    assert tensor.shape == (1, 32, 32, 1)
    assert tensor.dtype == np.float32
    assert 0 <= tensor.min() < tensor.max() <= 1
    assert tensor[0, 0, 0, 0] == 0


def test_controlled_segmentation_both_polarities():
    plate = np.full((80, 240, 3), 235, dtype=np.uint8)
    cv2.putText(plate, "AB12", (12, 61), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (15, 15, 15), 3)
    result = segment_characters(plate, "dark-on-light")
    assert len(result.boxes) == 4
    assert [box[0] for box in result.boxes] == sorted(box[0] for box in result.boxes)
    inverted = 255 - plate
    other = segment_characters(inverted, "light-on-dark")
    assert len(other.boxes) == 4
    assert enhance_plate(plate).shape == (80, 240)


def test_real_cnn_prediction_on_controlled_glyph():
    classes = load_classes(ROOT / "data" / "classes.json")
    model = load_model(ROOT / "models" / "platevision_cnn_v2.keras")
    char = np.zeros((40, 28), dtype=np.uint8)
    cv2.putText(char, "8", (2, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 2)
    result = predict_characters(model, classes, [char])
    assert len(result) == 1
    assert result[0]["character"] in classes
    assert 0 <= result[0]["score"] <= 1
    assert len(result[0]["top3"]) == 3


def test_invalid_upload_and_empty_character():
    with pytest.raises(ValueError):
        decode_upload(b"not an image")
    with pytest.raises(ValueError):
        decode_upload(b"")
    with pytest.raises(ValueError):
        character_tensor(np.zeros((8, 8), dtype=np.uint8))


def test_no_plate_detected():
    blank = np.zeros((200, 400, 3), dtype=np.uint8)
    assert detect_plate(blank) is None


def test_synthetic_plate_candidate_is_cropped():
    image = np.full((400, 800, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (220, 170), (600, 250), (230, 230, 230), -1)
    cv2.rectangle(image, (220, 170), (600, 250), (20, 20, 20), 3)
    cv2.putText(image, "AB1234", (240, 226), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (20, 20, 20), 3)
    candidate = detect_plate(image)
    assert candidate is not None
    assert candidate.crop.shape[1] > candidate.crop.shape[0]
    assert candidate.crop.shape[1] >= 300
    assert len(segment_characters(candidate.crop).characters) == 6
