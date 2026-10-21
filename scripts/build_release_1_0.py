"""Create a private-photo-free source folder and independently checkable ZIP."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
NAME = "PlateVision-AI-Pro-Academic-1.0"
FOLDER = ROOT / "delivery" / NAME
ARCHIVE = FOLDER.parent / f"{NAME}.zip"

FILES = [
    "app.py", ".gitignore", ".gitattributes", ".streamlit/config.toml", "pytest.ini",
    "requirements.txt", "requirements-ocr.txt", "README.md", "DOCUMENTATION_ML.md",
    "data/classes.json", "data/results.json",
    "data/manifest_v31.json", "data/validation_metrics_v31.json",
    "models/platevision_cnn_v2.keras", "models/platevision_cnn_v31.keras",
    "assets/cnn_v2_training.png", "assets/training_curves_v31.png",
    "assets/demo_synthetic_vehicle.png", "assets/ui_analysis_desktop.png",
    "assets/ui_comparison_desktop.png", "assets/ui_about_desktop.png",
    "assets/ui_analysis_mobile.png",
    "notebooks/PlateVision_AI_Training_V1_V2_authentic_excerpt.ipynb",
    "notebooks/PlateVision_AI_CNN_V3_Ready.ipynb",
    "notebooks/PlateVision_AI_CNN_V3_1_Ready.ipynb",
    "scripts/build_release_1_0.py", "scripts/download_easyocr.py",
    "scripts/validate_v21.py", "scripts/validate_v32.py",
]


def files_to_ship() -> list[Path]:
    paths = [ROOT / name for name in FILES]
    for directory in ("src", "tests"):
        paths.extend((ROOT / directory).glob("*.py"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release inputs: {missing}")
    return sorted(set(paths), key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh only a previously generated folder with the exact allowlist.")
    args = parser.parse_args()
    files = files_to_ship()
    if args.refresh:
        expected = {path.relative_to(ROOT).as_posix() for path in files} | {"RELEASE_MANIFEST.json"}
        existing = {path.relative_to(FOLDER).as_posix() for path in FOLDER.rglob("*") if path.is_file()}
        if (not FOLDER.is_dir() or not ARCHIVE.is_file() or not existing <= expected or
                "RELEASE_MANIFEST.json" not in existing or
                not FOLDER.resolve().is_relative_to((ROOT / "delivery").resolve())):
            raise SystemExit("Release target does not match the controlled allowlist; refresh refused.")
    elif FOLDER.exists() or ARCHIVE.exists():
        raise SystemExit("Release target already exists; review it before using --refresh.")
    else:
        FOLDER.mkdir(parents=True)
    manifest = {"release": "1.0", "archive_root": NAME, "files": {}}
    for path in files:
        relative = path.relative_to(ROOT)
        target = FOLDER / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        content = target.read_bytes()
        manifest["files"][relative.as_posix()] = {
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (ROOT / "RELEASE_MANIFEST.json").write_text(payload, encoding="utf-8")
    (FOLDER / "RELEASE_MANIFEST.json").write_text(payload, encoding="utf-8")
    with ZipFile(ARCHIVE, "w", ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(FOLDER.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(NAME) / path.relative_to(FOLDER))
    print(json.dumps({"folder": str(FOLDER), "archive": str(ARCHIVE),
                      "files": len(files) + 1, "zip_bytes": ARCHIVE.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
