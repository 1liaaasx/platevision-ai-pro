# Reproducibility and provenance

## Verified local environment

Validation was performed on Windows, Python 3.12.10, TensorFlow CPU 2.21.0, Keras 3.15.1, Streamlit 1.66.0, EasyOCR 1.7.2, PyTorch 2.14.1+cpu, OpenCV 4.14.0 and NumPy 2.5.4. The local Pillow version is 12.3.0. `requirements.txt` defines the supported base ranges; `requirements-ocr.txt` enables optional OCR. The app loads only the selected model on demand and caches the loaded object in Streamlit. EasyOCR's official CRAFT/English G2 weights are external to the source folder and are never fetched during ordinary inference.

| Checkpoint | SHA-256 |
| --- | --- |
| `models/platevision_cnn_v2.keras` | `731ac81f7d9cefb878ee74545e3e076dccd103df8388a090d03a5a660ef7f15f` |
| `models/platevision_cnn_v31.keras` | `851887cd2ecc5603a359fd0f753c3629514306fc2fcb852ee69d528a3e9913a9` |

The 36-class order comes from `data/classes.json`. V3.1's `data/manifest_v31.json` records training seed `2026`, validation seed `2228`, font splits/hashes, NumPy 2.1.3 and Pillow 11.3.0 for the supplied experiment. Our local environment is different and does not contain the four exact DejaVu font hashes, so the reserved test images cannot be reproduced reliably here. No final DejaVu score was calculated. The original historical V2 training run is not included in this candidate.

## Re-run local checks

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.validate_v21
.\.venv\Scripts\python.exe -m scripts.profile_release
```

For the optional third engine, install `requirements-ocr.txt`, run `python -m scripts.download_easyocr` once and verify `availability() == 'ready'`. Private-photo diagnostics are excluded from this candidate. Do not copy private images, annotations or outputs into a repository.

## Benchmark boundaries

Historical V2 scores, supplied V3.1 validation, paired synthetic smoke cases and one private-photo observation answer different questions. They are not interchangeable. The real-photo evaluator requires independent authorized images and annotations; it writes per-image errors and aggregate detection IoU≥0.5, whole-plate exactness, edit-distance character accuracy and timings. Its empty example file deliberately fails fast. An exact reproduction of the V3.1 final synthetic test would additionally require the manifest's four font files with matching hashes, the original generation code and compatible Pillow/NumPy versions. A GPU is only relevant if retraining a model; no release validation or local inference requires one.
