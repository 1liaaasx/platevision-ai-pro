"""Plate character segmentation with polarity selection."""

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import enhance_plate


@dataclass
class Segmentation:
    binary: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    characters: list[np.ndarray]
    polarity: str


def _extract(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (0.12*h <= ch <= 0.92*h and 0.008*w <= cw <= 0.35*w
                and 0.12 <= cw/max(1, ch) <= 1.2 and area >= 0.001*w*h
                and x > 0 and x+cw < w-1):
            boxes.append((x, y, cw, ch))
    return sorted(boxes, key=lambda b: b[0])


def _remove_border_components(binary: np.ndarray) -> np.ndarray:
    """Remove connected foreground touching crop edges, usually a plate frame."""
    _, labels = cv2.connectedComponents(binary)
    border = np.unique(np.concatenate((labels[0], labels[-1],
                                       labels[:, 0], labels[:, -1])))
    cleaned = binary.copy()
    cleaned[np.isin(labels, border)] = 0
    return cleaned


def segment_characters(plate: np.ndarray, polarity: str = "auto") -> Segmentation:
    """Return isolated white glyphs on black; auto evaluates both polarities."""
    if polarity not in {"auto", "dark-on-light", "light-on-dark"}:
        raise ValueError("Polarité inconnue.")
    gray = enhance_plate(plate)
    if min(gray.shape) < 16:
        return Segmentation(np.zeros_like(gray), [], [], "indéterminée")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    options = {"light-on-dark": _remove_border_components(bright),
               "dark-on-light": _remove_border_components(cv2.bitwise_not(bright))}
    if polarity == "auto":
        ranked = []
        for name, binary in options.items():
            boxes = _extract(binary)
            fill = np.count_nonzero(binary) / binary.size
            score = len(boxes) - 3 * max(0, fill - 0.5)
            ranked.append((score, -fill, name))
        polarity = max(ranked)[2]
    binary = options[polarity]
    boxes = _extract(binary)
    chars = []
    for x, y, w, h in boxes:
        pad = max(1, round(0.08 * h))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(binary.shape[1], x+w+pad), min(binary.shape[0], y+h+pad)
        chars.append(binary[y0:y1, x0:x1].copy())
    return Segmentation(binary, boxes, chars, polarity)


def draw_boxes(plate: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    if plate.ndim == 2:
        shown = cv2.cvtColor(plate, cv2.COLOR_GRAY2RGB)
    else:
        shown = plate.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(shown, (x, y), (x+w, y+h), (30, 205, 130), 2)
    return shown
