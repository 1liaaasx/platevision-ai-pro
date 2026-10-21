"""Guarded, testable inference pipeline for a single plate region."""

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from .cnn_inference import predict_characters
from .detection import PlateCandidate, detect_plate
from .preprocessing import as_rgb
from .segmentation import Segmentation, segment_characters


@dataclass
class Analysis:
    status: str
    message: str
    plate: np.ndarray | None = None
    candidate: PlateCandidate | None = None
    segmentation: Segmentation | None = None
    predictions: list[dict] | None = None
    raw_plate: np.ndarray | None = None


def region_is_plausible(plate: np.ndarray, segmentation: Segmentation) -> bool:
    """Conservative single-line geometry gate, not a plate classifier."""
    height, width = plate.shape[:2]
    if height < 16 or not 1.8 <= width / height <= 7.5:
        return False
    boxes = segmentation.boxes
    if not 3 <= len(boxes) <= 10:
        return False
    char_heights = np.array([box[3] for box in boxes], dtype=float)
    centers = np.array([box[1] + box[3] / 2 for box in boxes], dtype=float)
    return bool(char_heights.min() >= 0.45 * np.median(char_heights)
                and centers.max() - centers.min() <= 0.35 * height)


def analyze_image(image: np.ndarray, mode: str, polarity: str, classes: list[str],
                  crop_coords: tuple[int, int, int, int] | None = None,
                  model=None, model_loader=None, run_cnn: bool = True,
                  color_order: str = "RGB") -> Analysis:
    """Analyze photo, cropped plate, or manual crop. CNN is called only after validation."""
    image = as_rgb(image, color_order)
    candidate = None
    if mode == "photo":
        candidate = detect_plate(image)
        if candidate is None:
            return Analysis("no_candidate", "Aucune plaque détectée avec les paramètres actuels. Essayez le recadrage manuel ou une plaque déjà recadrée.")
        plate = candidate.crop
    elif mode == "cropped":
        plate = image
    elif mode == "manual":
        if crop_coords is None:
            raise ValueError("Définissez une région à recadrer.")
        x0, y0, x1, y1 = crop_coords
        if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
            raise ValueError("Coordonnées de recadrage invalides.")
        plate = image[y0:y1, x0:x1]
    else:
        raise ValueError("Type d'image inconnu.")
    raw_plate = plate.copy()
    height, width = plate.shape[:2]
    if height < 16 or not 1.8 <= width / height <= 7.5:
        return Analysis("unusable", "La région n'a pas des proportions de plaque horizontale exploitables. Vérifiez le recadrage.", plate, candidate, raw_plate=raw_plate)
    if max(height, width) > 1600:
        scale = 1600 / max(height, width)
        plate = cv2.resize(plate, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA)
    elif height < 40:
        factor = min(4, max(2, ceil(48 / height)))
        plate = cv2.resize(plate, None, fx=factor, fy=factor,
                           interpolation=cv2.INTER_CUBIC)
    segmentation = segment_characters(plate, polarity)
    if not segmentation.characters:
        return Analysis("unusable", "Aucun caractère exploitable n'a été extrait de cette région.",
                        plate, candidate, segmentation, raw_plate=raw_plate)
    if not region_is_plausible(plate, segmentation):
        return Analysis("unusable", "La région ne présente pas une ligne de caractères de plaque suffisamment plausible. Vérifiez le recadrage et le type d'image.",
                        plate, candidate, segmentation, raw_plate=raw_plate)
    predictions = None
    if run_cnn:
        if model is None:
            if model_loader is None:
                raise ValueError("Aucun modèle CNN disponible.")
            model = model_loader()
        predictions = predict_characters(model, classes, segmentation.characters)
    message = ("Lecture expérimentale produite, non vérifiée." if run_cnn
               else "Région et caractères prêts pour la reconnaissance ; lecture non vérifiée.")
    return Analysis("ready", message, plate, candidate,
                    segmentation, predictions, raw_plate)
