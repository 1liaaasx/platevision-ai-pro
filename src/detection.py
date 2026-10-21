"""Experimental OpenCV license plate candidate detection."""

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import grayscale
from .segmentation import segment_characters


@dataclass
class PlateCandidate:
    crop: np.ndarray
    polygon: np.ndarray
    score: float


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                     points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)


def _warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _ordered_quad(quad)
    width = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    height = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    if width < 2 or height < 2:
        return image[0:0, 0:0]
    dst = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst), (width, height))


def _small_text_candidate(work: np.ndarray) -> PlateCandidate | None:
    """Fallback for plates whose border is too weak to form a quadrilateral."""
    height, width = work.shape[:2]
    if max(width / height, height / width) > 2.5:
        return None
    gray = grayscale(work)
    kernel_width = max(9, min(31, round(width * 0.03) | 1))
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)),
    )
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    components = []
    for x, y, w, h, area in stats[1:]:
        if (max(4, 0.018 * height) <= h <= 0.16 * height
                and 2 <= w <= 1.7 * h and area >= 4):
            components.append((int(x), int(y), int(w), int(h)))
    components.sort(key=lambda box: box[0])
    best = None
    seen = set()
    for i, first in enumerate(components):
        for second in components[i+1:]:
            h = max(first[3], second[3])
            gap = second[0] - (first[0] + first[2])
            if gap > 2.5 * h:
                break
            if gap < -0.5 * h or not 0.65 <= second[3] / first[3] <= 1.55:
                continue
            x_a = first[0] + first[2] / 2
            x_b = second[0] + second[2] / 2
            y_a = first[1] + first[3] / 2
            y_b = second[1] + second[3] / 2
            slope = (y_b - y_a) / max(1, x_b - x_a)
            if abs(slope) > 0.3:
                continue
            group = [
                box for box in components
                if first[0] - 1.5*h <= box[0] <= first[0] + 12*h
                and 0.65 <= box[3] / first[3] <= 1.55
                and abs(box[1] + box[3]/2 - (y_a + slope*(box[0] + box[2]/2 - x_a))) <= 0.42*h
            ]
            if not 5 <= len(group) <= 10:
                continue
            group = sorted(group, key=lambda box: box[0])
            key = tuple(group)
            if key in seen:
                continue
            seen.add(key)
            typical_h = float(np.median([box[3] for box in group]))
            x0 = max(0, min(box[0] for box in group) - round(0.55*typical_h))
            y0 = max(0, min(box[1] for box in group) - round(0.65*typical_h))
            x1 = min(width, max(box[0]+box[2] for box in group) + round(0.55*typical_h))
            y1 = min(height, max(box[1]+box[3] for box in group) + round(0.8*typical_h))
            if y1 <= y0 or not 1.8 <= (x1-x0)/(y1-y0) <= 7.5:
                continue
            crop = work[y0:y1, x0:x1]
            evaluated = (cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                         if typical_h < 14 else crop)
            segmented = segment_characters(evaluated)
            if not 5 <= len(segmented.boxes) <= 10:
                continue
            char_h = np.array([box[3] for box in segmented.boxes], dtype=float)
            centers = np.array([box[1] + box[3]/2 for box in segmented.boxes])
            if char_h.min() < 0.45*np.median(char_h) or centers.max()-centers.min() > 0.35*evaluated.shape[0]:
                continue
            score = (len(segmented.boxes) + 0.3*len(group)
                     + 0.15*((y0+y1)/2)/height)
            if best is None or score > best[0]:
                polygon = np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]], dtype=np.int32)
                best = (score, PlateCandidate(crop, polygon, float(score/12)))
    return best[1] if best else None


def detect_plate(image: np.ndarray) -> PlateCandidate | None:
    """Find a plausible horizontal plate; return None when confidence is inadequate."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La détection attend une image RGB.")
    height, width = image.shape[:2]
    if min(height, width) < 40:
        return None
    scale = min(1.0, 1200 / max(height, width))
    work = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    gray = grayscale(work)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = work.shape[0] * work.shape[1]
    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        x, y, w, h = cv2.boundingRect(quad)
        ratio, area = w / max(1, h), w * h / image_area
        if not (1.8 <= ratio <= 6.5 and 0.001 <= area <= 0.5 and w >= 60 and h >= 15):
            continue
        crop = _warp(work, quad)
        if crop.size == 0:
            continue
        rectangularity = cv2.contourArea(contour) / max(1, w*h)
        if rectangularity < 0.45:
            continue
        score = float(rectangularity * (1 - min(abs(ratio-4.0)/6, 0.5)) * min(area/0.02, 1))
        candidates.append((score, quad, crop))
    if not candidates:
        fallback = _small_text_candidate(work)
        if fallback is None:
            return None
        if scale < 1:
            polygon = np.rint(fallback.polygon / scale).astype(np.int32)
            fallback = PlateCandidate(_warp(image, polygon), polygon, fallback.score)
        return fallback
    score, quad, crop = max(candidates, key=lambda item: item[0])
    quad = np.rint(quad.reshape(4, 2) / scale).astype(np.int32)
    if scale < 1:
        crop = _warp(image, quad)
    return PlateCandidate(crop=crop, polygon=quad, score=score)


def draw_candidate(image: np.ndarray, candidate: PlateCandidate) -> np.ndarray:
    shown = image.copy()
    cv2.polylines(shown, [candidate.polygon], True, (30, 205, 130), 3)
    return shown
