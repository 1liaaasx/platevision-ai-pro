"""Image input validation and character preprocessing."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 20_000_000


def as_rgb(image: np.ndarray, color_order: str = "RGB") -> np.ndarray:
    """Normalize explicit array color orders before entering the RGB pipeline."""
    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Image uint8 non vide attendue.")
    if image.ndim < 2 or image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ValueError("Dimensions d'image invalides ou supérieures à 20 millions de pixels.")
    if image.ndim == 2 and color_order in ("RGB", "GRAY"):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    conversions = {
        (3, "RGB"): None,
        (3, "BGR"): cv2.COLOR_BGR2RGB,
        (4, "RGBA"): cv2.COLOR_RGBA2RGB,
        (4, "BGRA"): cv2.COLOR_BGRA2RGB,
    }
    if image.ndim != 3 or (image.shape[2], color_order) not in conversions:
        raise ValueError("Ordre couleur invalide ; utilisez RGB, BGR, RGBA, BGRA ou GRAY.")
    conversion = conversions[(image.shape[2], color_order)]
    return image if conversion is None else cv2.cvtColor(image, conversion)


def decode_upload(content: bytes) -> np.ndarray:
    """Decode a user image to RGB, respecting EXIF rotation."""
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("L'image est vide ou dépasse 15 Mo.")
    try:
        with Image.open(BytesIO(content)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("L'image dépasse 20 millions de pixels.")
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Fichier image invalide ou illisible.") from exc


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raise ValueError("Image attendue en gris ou en RGB.")


def enhance_plate(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def character_tensor(character: np.ndarray) -> np.ndarray:
    """White glyph on black, centered in 32x32 and normalized for the CNN."""
    gray = grayscale(character)
    if gray.size == 0:
        raise ValueError("Caractère vide.")
    mask = (gray > 0).astype(np.uint8) * 255
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Caractère sans pixels de premier plan.")
    x, y, w, h = cv2.boundingRect(points)
    glyph = mask[y:y+h, x:x+w]
    scale = min(24 / w, 26 / h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(glyph, new_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((32, 32), dtype=np.uint8)
    ox, oy = (32 - new_size[0]) // 2, (32 - new_size[1]) // 2
    canvas[oy:oy+new_size[1], ox:ox+new_size[0]] = resized
    return (canvas.astype(np.float32) / 255.0)[None, :, :, None]


def v31_character_tensor(character: np.ndarray) -> np.ndarray:
    """Prepare a segmented bright glyph for V3.1's inverted 32x32 input.

    Training rendered a dark glyph on a 48x48 light canvas, resized it to
    32x32, then used 1 - gray/255. Segmented masks have no original canvas;
    this reconstruction preserves aspect ratio and supplies a light margin.
    """
    gray = grayscale(character)
    if gray.size == 0:
        raise ValueError("Caractère vide.")
    mask = (gray > 0).astype(np.uint8) * 255
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Caractère sans pixels de premier plan.")
    x, y, w, h = cv2.boundingRect(points)
    glyph = mask[y:y+h, x:x+w]
    scale = min(30 / w, 32 / h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(glyph, new_size,
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((48, 48), 255, dtype=np.uint8)
    ox, oy = (48 - new_size[0]) // 2, (48 - new_size[1]) // 2
    canvas[oy:oy+new_size[1], ox:ox+new_size[0]] = 255 - resized
    gray32 = cv2.resize(canvas, (32, 32), interpolation=cv2.INTER_CUBIC)
    return (1.0 - gray32.astype(np.float32) / 255.0)[None, :, :, None]
