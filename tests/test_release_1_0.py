"""Release regressions for input normalization and exports."""

import json
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from src.detection import PlateCandidate
from src.exports import annotated_png, result_csv, result_json
from src.pipeline import Analysis
from src.preprocessing import as_rgb, decode_upload


def test_color_orders_and_gray_are_explicit():
    rgb = np.array([[[25, 60, 220]]], dtype=np.uint8)
    assert np.array_equal(as_rgb(rgb), rgb)
    assert np.array_equal(as_rgb(rgb[:, :, ::-1].copy(), "BGR"), rgb)
    rgba = np.dstack((rgb, np.full((1, 1), 255, dtype=np.uint8)))
    assert np.array_equal(as_rgb(rgba, "RGBA"), rgb)
    assert as_rgb(np.zeros((4, 5), dtype=np.uint8), "GRAY").shape == (4, 5, 3)
    with pytest.raises(ValueError):
        as_rgb(rgb, "RGBA")
    with pytest.raises(ValueError):
        as_rgb(rgb.astype(np.float32))


def test_decode_png_rgba_and_grayscale():
    for mode in ("RGBA", "L"):
        image = Image.new(mode, (8, 6), (12, 34, 56, 255) if mode == "RGBA" else 127)
        stream = BytesIO()
        image.save(stream, format="PNG")
        assert decode_upload(stream.getvalue()).shape == (6, 8, 3)


def test_exif_orientation_and_corrupt_input():
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    metadata = Image.Exif()
    metadata[274] = 6
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=metadata)
    assert decode_upload(stream.getvalue()).shape == (8, 6, 3)
    with pytest.raises(ValueError, match="invalide"):
        decode_upload(b"not a photo")


def test_export_is_observation_not_claim_of_success():
    source = np.zeros((50, 120, 3), dtype=np.uint8)
    analysis = Analysis("ready", "observed", plate=source)
    result = {"analysis": analysis, "source": source, "engines": {
        "CNN V2": {"text": "ABC", "predictions": [{"character": "A", "score": 0.9}],
                    "elapsed_ms": 12.5}}, "pipeline_elapsed_ms": 4.2}
    exported = json.loads(result_json(result, "ABD"))
    assert exported["engines"]["CNN V2"]["reference_check"]["exact"] is False
    assert exported["engines"]["CNN V2"]["characters"][0]["raw_softmax"] == 0.9
    assert b"False" in result_csv(result, "ABD")
    assert annotated_png(result) is None
    analysis.candidate = PlateCandidate(source.copy(), np.array([[2, 2], [100, 2],
                                                                  [100, 40], [2, 40]], dtype=np.int32), 0.5)
    encoded = annotated_png(result)
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    assert cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR).shape == source.shape
