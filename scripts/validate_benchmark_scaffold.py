"""Exercise the real-benchmark evaluator on one generated image, not real data."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .benchmark_real import evaluate


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="platevision-benchmark-smoke-") as temporary:
        root = Path(temporary)
        images = root / "images"
        images.mkdir()
        plate = np.full((90, 400, 3), 235, np.uint8)
        cv2.putText(plate, "456DEF", (18, 66), cv2.FONT_HERSHEY_SIMPLEX,
                    1.7, (20, 20, 20), 3, cv2.LINE_AA)
        vehicle = np.full((400, 640, 3), 90, np.uint8)
        vehicle[260:350, 110:510] = plate
        cv2.imwrite(str(images / "synthetic.png"), cv2.cvtColor(vehicle, cv2.COLOR_RGB2BGR))
        annotations = root / "annotations.json"
        annotations.write_text(json.dumps({"images": [{"id": "synthetic-smoke-1",
            "file": "synthetic.png", "plate_bbox": [110, 260, 510, 350],
            "text": "456DEF", "rights": "generated locally for test"}]}), encoding="utf-8")
        report = evaluate(annotations, images, ["CNN V2", "CNN V3.1"])
        assert report["summary"]["images"] == 1
        assert {"CNN V2", "CNN V3.1"} == set(report["summary"]["engines"])
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
